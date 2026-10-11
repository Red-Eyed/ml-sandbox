"""Empirical finiteness checks of scalar losses and their input gradients."""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from time import perf_counter
from typing import Literal, assert_never

import torch
from rich.progress import Progress
from torch import Tensor

type Inputs = tuple[Tensor, ...]
type Loss = Callable[..., Tensor]
type Sampler = Callable[[torch.Generator], Inputs | Sample]
type Source = Literal["boundary", "random"]
type Issue = Literal[
    "nonfinite_loss", "nonfinite_gradient", "exception", "missing_required_gradient"
]
type Metric = Issue | Literal["disconnected"]
type FailureKind = tuple[Source, str, tuple[Issue, ...], tuple[str, str] | Absent]


@dataclass(frozen=True)
class Sample:
    """Label random inputs so their sampling strategy has its own summary."""

    inputs: Inputs
    strategy: str

    def __post_init__(self) -> None:
        """Reject empty strategy names instead of merging unidentified experiments."""
        if not self.strategy.strip():
            raise ValueError("strategy must have a nonempty name")


@dataclass(frozen=True)
class Absent:
    """Explain why a measurement or gradient is unavailable."""

    reason: str


_NO_TIME_LIMIT = Absent("no time limit")
_NO_TRIAL_LIMIT = Absent("no trial limit")


@dataclass(frozen=True)
class Raised:
    """Retain a callable's exception without retaining its traceback or graph."""

    stage: Literal["forward", "backward"]
    exception_type: str
    message: str


@dataclass(frozen=True)
class FailureExample:
    """Detached CPU evidence, indexed from zero within its sampling source.

    Input devices and requires-grad flags allow replay in the original precision.
    Gradients correspond to input positions; constants have reason-carrying absence.
    Tensor payloads are independent copies, but remain mutable PyTorch tensors.
    """

    source: Source
    strategy: str
    trial_index: int
    inputs: Inputs
    devices: tuple[torch.device, ...]
    requires_grad: tuple[bool, ...]
    loss: Tensor | Absent
    gradients: tuple[Tensor | Absent, ...]
    issues: tuple[Issue, ...]
    disconnected_inputs: tuple[int, ...]
    error: Raised | Absent

    def replay_inputs(self) -> Inputs:
        """Return fresh leaves on their original devices without changing the evidence."""
        return tuple(
            value.to(device, copy=True).requires_grad_(requires_grad)
            for value, device, requires_grad in zip(
                self.inputs, self.devices, self.requires_grad, strict=True
            )
        )


@dataclass
class FinitenessSummary:
    """Aggregate trial outcomes; issue counts overlap and rates use all trials.

    Disconnection is coverage information, not a failure unless required by the caller.
    Checked counts distinguish evaluated tensors from stages blocked by exceptions.
    Maxima include only finite elements, even in otherwise nonfinite tensors.
    """

    trials: int = 0
    failed_trials: int = 0
    losses_checked: int = 0
    gradient_trials_checked: int = 0
    nonfinite_loss: int = 0
    nonfinite_gradient: int = 0
    exception: int = 0
    disconnected: int = 0
    disconnected_inputs: dict[int, int] = field(default_factory=dict)
    missing_required_gradient: int = 0
    max_abs_loss: float | Absent = field(default_factory=lambda: Absent("no finite loss"))
    max_abs_gradient: float | Absent = field(
        default_factory=lambda: Absent("no finite gradient elements")
    )

    def rate(self, issue: Metric) -> float | Absent:
        """Return the fraction of all trials with this issue, or explain an empty run."""
        if self.trials == 0:
            return Absent("no trials")
        match issue:
            case "nonfinite_loss":
                count = self.nonfinite_loss
            case "nonfinite_gradient":
                count = self.nonfinite_gradient
            case "exception":
                count = self.exception
            case "disconnected":
                count = self.disconnected
            case "missing_required_gradient":
                count = self.missing_required_gradient
            case _:
                assert_never(issue)
        return count / self.trials


@dataclass(frozen=True)
class FinitenessReport:
    """Separate boundary coverage from Monte Carlo outcomes and bounded evidence.

    A seed reproduces a sampler's random prefix, not the number of trials completed
    under a wall-time budget. No-failure reports are observations, not guarantees.
    """

    seed: int
    duration: timedelta | Absent
    max_trials: int | Absent
    elapsed: timedelta
    stop_reason: Literal["duration", "max_trials"]
    gradient_scale: float
    required_gradients: tuple[int, ...]
    boundary_cases_requested: int
    boundary: FinitenessSummary
    random: FinitenessSummary
    strategies: dict[str, FinitenessSummary]
    examples: tuple[FailureExample, ...]

    @property
    def trials(self) -> int:
        """Return the total number of boundary and random trials."""
        return self.boundary.trials + self.random.trials


@dataclass
class _Observation:
    """Keep one trial's graph-free results until aggregation or evidence capture."""

    loss: Tensor | Absent = field(default_factory=lambda: Absent("forward not evaluated"))
    gradients: tuple[Tensor | Absent, ...] = ()
    issues: list[Issue] = field(default_factory=list)
    disconnected_inputs: list[int] = field(default_factory=list)
    error: Raised | Absent = field(default_factory=lambda: Absent("no exception"))
    max_abs_loss: float | Absent = field(default_factory=lambda: Absent("no finite loss"))
    max_abs_gradient: float | Absent = field(default_factory=lambda: Absent("no finite gradient"))


def _prepare(inputs: Inputs, required_gradients: tuple[int, ...]) -> Inputs:
    """Validate the sampler contract and isolate each trial from caller-owned tensors."""
    if not inputs or not any(value.requires_grad for value in inputs):
        raise ValueError("Each input tuple must contain at least one requires-grad tensor")
    for value in inputs:
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Sampler and boundary inputs must be finite")
        if value.requires_grad and not value.is_floating_point():
            raise ValueError("Differentiated inputs must be real floating-point tensors")
    for index in required_gradients:
        if index >= len(inputs) or not inputs[index].requires_grad:
            raise ValueError("Required gradient indices must identify requires-grad inputs")
    return tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)


def _observe(
    loss_fn: Loss, inputs: Inputs, required_gradients: tuple[int, ...], gradient_scale: float
) -> _Observation:
    """Evaluate both stages independently, retaining backward checks after nonfinite loss."""
    observation = _Observation(gradients=tuple(Absent("backward not evaluated") for _ in inputs))
    try:
        with torch.enable_grad():
            loss = loss_fn(*inputs)
        if loss.ndim != 0 or not loss.is_floating_point():
            raise ValueError("loss_fn must return a real floating-point scalar tensor")
        observation.loss = loss.detach()
        if not bool(torch.isfinite(loss)):
            observation.issues.append("nonfinite_loss")
    except Exception as error:  # noqa: BLE001 -- tested exceptions are retained as evidence
        # The loss is an intentionally untrusted callable under test. Sampler,
        # aggregation, and process-control exceptions are not swallowed here.
        observation.issues.append("exception")
        observation.error = Raised("forward", type(error).__name__, str(error))
        return observation

    _observe_backward(loss, inputs, observation, gradient_scale)
    if any(index in observation.disconnected_inputs for index in required_gradients):
        observation.issues.append("missing_required_gradient")
    return observation


def _observe_backward(
    loss: Tensor, inputs: Inputs, observation: _Observation, gradient_scale: float
) -> None:
    """Align gradients with input arguments and classify connectivity and finiteness."""
    if not loss.requires_grad:
        observation.disconnected_inputs = [
            index for index, value in enumerate(inputs) if value.requires_grad
        ]
        observation.gradients = tuple(
            Absent("loss has no autograd graph" if value.requires_grad else "constant input")
            for value in inputs
        )
        return

    differentiated = tuple(value for value in inputs if value.requires_grad)
    try:
        with torch.enable_grad():
            computed = iter(
                torch.autograd.grad(
                    loss,
                    differentiated,
                    grad_outputs=loss.new_tensor(gradient_scale),
                    allow_unused=True,
                )
            )
        gradients: list[Tensor | Absent] = []
        for index, value in enumerate(inputs):
            if not value.requires_grad:
                gradients.append(Absent("constant input"))
                continue
            gradient = next(computed)
            if gradient is None:
                gradients.append(Absent("input disconnected from loss"))
                observation.disconnected_inputs.append(index)
                continue
            gradients.append(gradient.detach())
            if (
                not bool(torch.isfinite(gradient).all())
                and "nonfinite_gradient" not in observation.issues
            ):
                observation.issues.append("nonfinite_gradient")
        observation.gradients = tuple(gradients)
    except Exception as error:  # noqa: BLE001 -- tested exceptions are retained as evidence
        observation.issues.append("exception")
        observation.error = Raised("backward", type(error).__name__, str(error))


def _finite_max(value: Tensor | float | Absent, previous: float | Absent) -> float | Absent:
    """Extend a maximum using finite elements without casting reductions to a wider dtype."""
    match value:
        case Absent():
            return previous
        case float():
            match previous:
                case Absent():
                    return value
                case float():
                    return max(value, previous)
                case _:
                    assert_never(previous)
        case Tensor():
            finite = value[torch.isfinite(value)]
            if finite.numel() == 0:
                return previous
            current = float(finite.abs().max().item())
            return _finite_max(current, previous)
        case _:
            assert_never(value)


def _measure_extrema(observation: _Observation) -> None:
    """Reduce tensors once so per-strategy and overall summaries reuse scalar measurements."""
    observation.max_abs_loss = _finite_max(observation.loss, observation.max_abs_loss)
    for gradient in observation.gradients:
        observation.max_abs_gradient = _finite_max(gradient, observation.max_abs_gradient)


def _aggregate(summary: FinitenessSummary, observation: _Observation) -> None:
    """Count trials once, issue categories independently, and finite extrema separately."""
    summary.trials += 1
    summary.failed_trials += bool(observation.issues)
    summary.nonfinite_loss += "nonfinite_loss" in observation.issues
    summary.nonfinite_gradient += "nonfinite_gradient" in observation.issues
    summary.exception += "exception" in observation.issues
    summary.disconnected += bool(observation.disconnected_inputs)
    summary.missing_required_gradient += "missing_required_gradient" in observation.issues
    for index in observation.disconnected_inputs:
        summary.disconnected_inputs[index] = summary.disconnected_inputs.get(index, 0) + 1
    match observation.loss:
        case Tensor():
            summary.losses_checked += 1
    if any(isinstance(gradient, Tensor) for gradient in observation.gradients):
        summary.gradient_trials_checked += 1
    summary.max_abs_loss = _finite_max(observation.max_abs_loss, summary.max_abs_loss)
    summary.max_abs_gradient = _finite_max(observation.max_abs_gradient, summary.max_abs_gradient)


def _snapshot(value: Tensor | Absent) -> Tensor | Absent:
    """Copy saved evidence to CPU so retained failures never retain graphs or GPU storage."""
    match value:
        case Tensor():
            return value.detach().to("cpu", copy=True)
        case Absent():
            return value
        case _:
            assert_never(value)


def _save_example(
    source: Source, strategy: str, trial_index: int, inputs: Inputs, observation: _Observation
) -> FailureExample:
    """Preserve original sampled inputs and evaluated evidence only for retained failures."""
    return FailureExample(
        source=source,
        strategy=strategy,
        trial_index=trial_index,
        inputs=tuple(value.detach().to("cpu", copy=True) for value in inputs),
        devices=tuple(value.device for value in inputs),
        requires_grad=tuple(value.requires_grad for value in inputs),
        loss=_snapshot(observation.loss),
        gradients=tuple(_snapshot(gradient) for gradient in observation.gradients),
        issues=tuple(observation.issues),
        disconnected_inputs=tuple(observation.disconnected_inputs),
        error=observation.error,
    )


def _failure_kind(
    source: Source, strategy: str, issues: tuple[Issue, ...], error: Raised | Absent
) -> FailureKind:
    """Group evidence by source, strategy, issue combination, and exception stage/type."""
    match error:
        case Raised(stage=stage, exception_type=exception_type):
            return source, strategy, issues, (stage, exception_type)
        case Absent():
            return source, strategy, issues, error
        case _:
            assert_never(error)


def _retain_example(
    examples: list[FailureExample],
    max_examples: int,
    source: Source,
    strategy: str,
    trial_index: int,
    inputs: Inputs,
    observation: _Observation,
) -> None:
    """Prefer novel failure groups over duplicates while keeping tensor storage bounded.

    When full, a new group replaces a duplicate of the most represented stored
    group. If all stored groups are distinct, the new group cannot be retained.
    This favors diversity without changing the sampler's random sequence.
    """
    if max_examples == 0 or not observation.issues:
        return
    if len(examples) < max_examples:
        examples.append(_save_example(source, strategy, trial_index, inputs, observation))
        return
    kind = _failure_kind(source, strategy, tuple(observation.issues), observation.error)
    kinds = [_failure_kind(e.source, e.strategy, e.issues, e.error) for e in examples]
    if kind in kinds:
        return
    duplicate, count = Counter(kinds).most_common(1)[0]
    if count == 1:
        return
    replace_index = len(kinds) - 1 - kinds[::-1].index(duplicate)
    examples[replace_index] = _save_example(source, strategy, trial_index, inputs, observation)


def _normalize_sample(sample: Inputs | Sample) -> Sample:
    """Give unlabeled tuples a default group while preserving caller-supplied labels."""
    match sample:
        case Sample():
            return sample
        case tuple():
            return Sample(sample, "random")
        case _:
            assert_never(sample)


def _validate_configuration(
    duration: timedelta | Absent,
    max_trials: int | Absent,
    max_examples: int,
    required_gradients: tuple[int, ...],
    gradient_scale: float,
) -> None:
    """Require explicit stopping limits and reject invalid evidence or gradient policies."""
    match duration, max_trials:
        case Absent(), Absent():
            raise ValueError("Provide duration, max_trials, or both")
    match duration:
        case timedelta():
            if duration.total_seconds() < 0:
                raise ValueError("duration must be non-negative")
    match max_trials:
        case int():
            if max_trials < 0:
                raise ValueError("max_trials must be non-negative")
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if any(index < 0 for index in required_gradients):
        raise ValueError("required_gradients must contain non-negative input indices")
    if not math.isfinite(gradient_scale):
        raise ValueError("gradient_scale must be finite")


def check_finiteness(
    loss_fn: Loss,
    sample_inputs: Sampler,
    *,
    duration: timedelta | Absent = _NO_TIME_LIMIT,
    max_trials: int | Absent = _NO_TRIAL_LIMIT,
    seed: int = 42,
    boundary_inputs: Sequence[Inputs] = (),
    max_examples: int = 10,
    required_gradients: tuple[int, ...] = (),
    gradient_scale: float = 1.0,
    quiet: bool = False,
) -> FinitenessReport:
    """Search for nonfinite scalar losses and input gradients until an explicit limit.

    Provide duration, max_trials, or both; the first reached limit stops sampling.
    Boundary cases run first and count against both limits. A completed
    trial may overrun the deadline; unfinished boundaries remain visible in the
    report. Sampling time, device synchronization, and evidence capture count
    toward elapsed time. CPU/CUDA/MPS checks synchronize when reading values.

    The sampler returns a tuple of tensors, or Sample(inputs, strategy) for named
    per-strategy summaries. It receives a private seeded CPU generator and controls bounds,
    shapes, dtype, and device. Use that generator for every random draw; loss
    randomness and state are caller-owned. Inputs are copied to fresh leaves,
    preserving requires-grad flags without modifying caller values or .grad.
    Each argument is copied independently; shared-storage aliasing is not tested.

    Backward uses gradient_scale as the incoming gradient, cast to the loss dtype,
    even for nonfinite losses. Rates
    are fractions of trials, with overlapping issue categories and stage coverage
    reported separately. Disconnected inputs are coverage information; they count
    as failures only for input positions listed in required_gradients. Every such
    position must have requires_grad=True. Missing gradients are never converted
    to zeros. Up to max_examples detached examples favor different source/strategy,
    issue combinations, and exception stages/types; all failures still count.

    Exceptions from forward/backward are evidence and do not stop the run.
    Invalid configuration or sampler inputs and sampler exceptions propagate;
    KeyboardInterrupt/SystemExit always propagate. Callables must not alter
    process-wide state. A failed accelerator context may require process restart.
    """
    _validate_configuration(duration, max_trials, max_examples, required_gradients, gradient_scale)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    boundary, random = FinitenessSummary(), FinitenessSummary()
    strategies: dict[str, FinitenessSummary] = {}
    examples: list[FailureExample] = []
    started = perf_counter()
    match duration:
        case timedelta():
            deadline: float | Absent = started + duration.total_seconds()
            progress_total = duration.total_seconds()
        case Absent():
            deadline = duration
            assert isinstance(max_trials, int)
            progress_total = float(max_trials)
    with Progress(disable=quiet, transient=True, refresh_per_second=2) as progress:
        task = progress.add_task("Checking numerical finiteness", total=progress_total)
        while True:
            if isinstance(deadline, float) and perf_counter() >= deadline:
                stop_reason: Literal["duration", "max_trials"] = "duration"
                break
            if isinstance(max_trials, int) and boundary.trials + random.trials >= max_trials:
                stop_reason = "max_trials"
                break
            if boundary.trials < len(boundary_inputs):
                source: Source = "boundary"
                summary = boundary
                sample = Sample(boundary_inputs[boundary.trials], "boundary")
            else:
                source = "random"
                summary = random
                sample = _normalize_sample(sample_inputs(generator))
            observation = _observe(
                loss_fn,
                _prepare(sample.inputs, required_gradients),
                required_gradients,
                gradient_scale,
            )
            _measure_extrema(observation)
            _retain_example(
                examples,
                max_examples,
                source,
                sample.strategy,
                summary.trials,
                sample.inputs,
                observation,
            )
            _aggregate(summary, observation)
            if source == "random":
                if sample.strategy not in strategies:
                    strategies[sample.strategy] = FinitenessSummary()
                _aggregate(strategies[sample.strategy], observation)
            completed = (
                perf_counter() - started
                if isinstance(deadline, float)
                else boundary.trials + random.trials
            )
            progress.update(task, completed=min(progress_total, completed))

    return FinitenessReport(
        seed=seed,
        duration=duration,
        max_trials=max_trials,
        elapsed=timedelta(seconds=perf_counter() - started),
        stop_reason=stop_reason,
        gradient_scale=gradient_scale,
        required_gradients=required_gradients,
        boundary_cases_requested=len(boundary_inputs),
        boundary=boundary,
        random=random,
        strategies=strategies,
        examples=tuple(examples),
    )
