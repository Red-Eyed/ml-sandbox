"""Exercise aggregation, replay, deadline behavior, and independent failure stages."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

import ml_sandbox.numerical_finiteness._check as module
from ml_sandbox.numerical_finiteness import (
    Absent,
    FinitenessReport,
    Inputs,
    Loss,
    Raised,
    Sample,
    Sampler,
    check_finiteness,
)

type Runner = Callable[..., FinitenessReport]
NO_TRIAL_LIMIT = Absent("no trial limit")


@dataclass
class ManualClock:
    """Model one second per loss evaluation without sleeping in tests."""

    seconds: float = 0.0

    def read(self) -> float:
        """Return the manually advanced monotonic time."""
        return self.seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    """Replace only the tester's deadline clock, leaving Rich's clock untouched."""
    clock = ManualClock()
    monkeypatch.setattr(module, "perf_counter", clock.read)
    return clock


@pytest.fixture
def run_trials(clock: ManualClock) -> Runner:
    """Execute an exact number of trials using a deterministic elapsed-time budget."""

    def run(
        loss_fn: Loss,
        sample_inputs: Sampler,
        *,
        trials: int = 4,
        boundary_inputs: Sequence[Inputs] = (),
        max_examples: int = 10,
        seed: int = 42,
        required_gradients: tuple[int, ...] = (),
        gradient_scale: float = 1.0,
        max_trials: int | Absent = NO_TRIAL_LIMIT,
    ) -> FinitenessReport:
        """Advance time even when the tested forward callable raises."""

        def timed_loss(*inputs: Tensor) -> Tensor:
            """Charge one simulated second for each attempted trial."""
            try:
                return loss_fn(*inputs)
            finally:
                clock.seconds += 1

        return check_finiteness(
            timed_loss,
            sample_inputs,
            duration=timedelta(seconds=trials),
            boundary_inputs=boundary_inputs,
            max_examples=max_examples,
            required_gradients=required_gradients,
            gradient_scale=gradient_scale,
            max_trials=max_trials,
            seed=seed,
            quiet=True,
        )

    return run


@pytest.fixture
def zero_sampler() -> Sampler:
    """Produce an independent differentiable scalar at a derivative singularity."""

    def sample(generator: torch.Generator) -> Inputs:
        """Return zero without consuming randomness."""
        return (torch.tensor(0.0, requires_grad=True),)

    return sample


def square(x: Tensor) -> Tensor:
    """Provide a finite loss with a connected derivative at zero."""
    return x.square()


def square_root(x: Tensor) -> Tensor:
    """Keep the derivative singularity at zero visible."""
    return x.sqrt()


def overflowing_forward(x: Tensor) -> Tensor:
    """Produce forward infinity while keeping the derivative equal to one."""
    return x + torch.tensor(float("inf"))


def both_nonfinite(x: Tensor) -> Tensor:
    """Produce infinity in both the loss and its derivative."""
    return (x + 100).exp()


def detached_loss(x: Tensor) -> Tensor:
    """Deliberately remove all autograd connectivity."""
    return x.square().detach()


def raises_forward(x: Tensor) -> Tensor:
    """Simulate a domain rejection by the callable under test."""
    raise ValueError("unsupported domain")


class BrokenBackward(torch.autograd.Function):
    """Expose a backward exception without breaking the forward computation."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        """Return a finite scalar with a custom backward edge."""
        return x.clone()

    @staticmethod
    def backward(ctx: FunctionCtx, *gradients: Tensor) -> Tensor:
        """Raise a representative backward runtime error."""
        raise RuntimeError("backward failed")


def raises_backward(x: Tensor) -> Tensor:
    """Evaluate a loss whose custom backward always raises."""
    return BrokenBackward.apply(x)


@pytest.mark.parametrize(
    "loss_fn,expected",
    [
        (square, (0, 0, 0, 0)),
        (overflowing_forward, (4, 0, 0, 0)),
        (square_root, (0, 4, 0, 0)),
        (both_nonfinite, (4, 4, 0, 0)),
        (raises_forward, (0, 0, 4, 0)),
        (raises_backward, (0, 0, 4, 0)),
        (detached_loss, (0, 0, 0, 4)),
    ],
)
def test_independent_categories_and_continuation(
    run_trials: Runner, zero_sampler: Sampler, loss_fn: Loss, expected: tuple[int, ...]
) -> None:
    """Failures do not terminate sampling, and forward/backward categories can overlap."""
    report = run_trials(loss_fn, zero_sampler)
    summary = report.random
    assert report.trials == 4
    assert report.elapsed == timedelta(seconds=4)
    assert (
        summary.nonfinite_loss,
        summary.nonfinite_gradient,
        summary.exception,
        summary.disconnected,
    ) == expected
    assert summary.failed_trials == (4 if any(expected[:3]) else 0)
    assert summary.rate("nonfinite_loss") == expected[0] / 4
    if loss_fn is overflowing_forward:
        assert summary.gradient_trials_checked == 4
        assert summary.max_abs_gradient == 1.0
        assert isinstance(summary.max_abs_loss, Absent)


def test_example_limit_does_not_limit_counts(run_trials: Runner, zero_sampler: Sampler) -> None:
    """Every failure counts after the detached evidence storage cap is reached."""
    report = run_trials(square_root, zero_sampler, trials=8, max_examples=2)
    assert report.random.nonfinite_gradient == 8
    assert len(report.examples) == 2
    assert [example.trial_index for example in report.examples] == [0, 1]
    assert report.examples[0].issues == ("nonfinite_gradient",)


def test_boundary_random_separation(run_trials: Runner) -> None:
    """A deterministic bad boundary cannot inflate the random-sample failure rate."""

    def sample(generator: torch.Generator) -> Inputs:
        """Sample strictly positive values away from the derivative singularity."""
        return (torch.rand((), generator=generator).add(1).requires_grad_(),)

    report = run_trials(
        square_root,
        sample,
        boundary_inputs=[(torch.tensor(0.0, requires_grad=True),)],
    )
    assert report.boundary_cases_requested == 1
    assert report.boundary.trials == 1
    assert report.boundary.rate("nonfinite_gradient") == 1.0
    assert report.random.trials == 3
    assert report.random.rate("nonfinite_gradient") == 0.0
    assert report.examples[0].source == "boundary"


def test_deadline_can_leave_boundaries_unchecked(run_trials: Runner, zero_sampler: Sampler) -> None:
    """Boundary evaluation uses the same time budget as random sampling."""
    boundaries = [(torch.tensor(0.0, requires_grad=True),)] * 5
    report = run_trials(square_root, zero_sampler, trials=2, boundary_inputs=boundaries)
    assert report.boundary_cases_requested == 5
    assert report.boundary.trials == 2
    assert report.random.trials == 0
    assert isinstance(report.random.rate("nonfinite_gradient"), Absent)


def test_reproducible_prefix_and_replay(run_trials: Runner) -> None:
    """A seeded sampler reproduces values; saved tensors preserve dtype and gradient flags."""

    def sample(generator: torch.Generator) -> Inputs:
        """Sample negative float64 values so every square root fails."""
        return (-torch.rand((), generator=generator, dtype=torch.float64).requires_grad_(),)

    first = run_trials(square_root, sample, seed=81)
    second = run_trials(square_root, sample, seed=81)
    for original, repeated in zip(first.examples, second.examples, strict=True):
        torch.testing.assert_close(original.inputs[0], repeated.inputs[0])
        assert not original.inputs[0].requires_grad
        replay = original.replay_inputs()
        assert replay[0].requires_grad and replay[0].is_leaf
        assert replay[0].dtype == torch.float64
        value = square_root(*replay)
        gradient = torch.autograd.grad(value, replay)[0]
        assert not bool(torch.isfinite(value))
        assert not bool(torch.isfinite(gradient))
        replay[0].detach().fill_(9)
        assert original.inputs[0].item() < 0


def test_unused_inputs_constants_and_existing_grads(run_trials: Runner) -> None:
    """Report unused differentiable inputs without accumulating or clearing caller gradients."""
    x = torch.tensor(2.0, requires_grad=True)
    x.grad = torch.tensor(99.0)
    unused = torch.tensor(3.0, requires_grad=True)
    constant = torch.tensor(5)

    def sample(generator: torch.Generator) -> Inputs:
        """Return caller-owned tensors, including a constant argument."""
        return x, unused, constant

    def loss(x: Tensor, unused: Tensor, constant: Tensor) -> Tensor:
        """Use the first input and constant, leaving the second disconnected."""
        return x.square() + constant

    report = run_trials(loss, sample, required_gradients=(1,))
    assert report.random.disconnected == 4
    assert report.random.missing_required_gradient == 4
    assert report.random.max_abs_loss == 9.0
    assert report.random.max_abs_gradient == 4.0
    assert x.grad.item() == 99.0
    assert unused.grad is None
    assert x.item() == 2.0
    example = report.examples[0]
    assert example.gradients[1] == Absent("input disconnected from loss")
    assert example.gradients[2] == Absent("constant input")
    assert example.requires_grad == (True, True, False)


def test_finite_maxima_ignore_nan_and_inf_elements(run_trials: Runner) -> None:
    """Nonfinite gradient tensors can still contribute finite elements to extrema."""

    def sample(generator: torch.Generator) -> Inputs:
        """Combine an infinite, finite, and NaN square-root derivative."""
        return (torch.tensor([0.0, 1.0, -1.0], requires_grad=True),)

    def loss(x: Tensor) -> Tensor:
        """Keep a scalar reduction while retaining all three input gradients."""
        return x.sqrt().sum()

    report = run_trials(loss, sample)
    assert report.random.nonfinite_loss == 4
    assert report.random.nonfinite_gradient == 4
    assert report.random.max_abs_gradient == 0.5
    assert isinstance(report.random.max_abs_loss, Absent)


@pytest.mark.parametrize(
    "loss_fn,stage", [(raises_forward, "forward"), (raises_backward, "backward")]
)
def test_exception_evidence(
    run_trials: Runner, zero_sampler: Sampler, loss_fn: Loss, stage: str
) -> None:
    """Keep the exception stage and message and any output produced before it."""
    example = run_trials(loss_fn, zero_sampler).examples[0]
    assert isinstance(example.error, Raised)
    assert example.error.stage == stage
    assert example.error.message
    if stage == "backward":
        assert isinstance(example.loss, Tensor)


def test_zero_budget_does_not_sample(zero_sampler: Sampler) -> None:
    """An empty run has unavailable rates and maxima instead of misleading zeros."""
    report = check_finiteness(square, zero_sampler, duration=timedelta(0), quiet=True)
    assert report.trials == 0
    assert report.examples == ()
    assert isinstance(report.random.max_abs_gradient, Absent)
    assert isinstance(report.random.rate("exception"), Absent)


@pytest.mark.parametrize(
    "inputs",
    [(), (torch.tensor(1.0),), (torch.tensor(float("nan"), requires_grad=True),)],
)
def test_invalid_sampler_inputs_propagate(run_trials: Runner, inputs: Inputs) -> None:
    """A broken sampler contract is configuration failure, not evidence about the loss."""

    def sample(generator: torch.Generator) -> Inputs:
        """Return a deliberately invalid trial tuple."""
        return inputs

    with pytest.raises(ValueError):
        run_trials(square, sample)


def test_sampler_errors_propagate(run_trials: Runner) -> None:
    """Do not turn a sampling bug into a stream of loss exceptions."""

    def sample(generator: torch.Generator) -> Inputs:
        """Simulate an unexpected error in input generation."""
        raise RuntimeError("sampler failed")

    with pytest.raises(RuntimeError, match="sampler failed"):
        run_trials(square, sample)


def test_process_control_propagates(run_trials: Runner, zero_sampler: Sampler) -> None:
    """A user interrupt must end the run rather than count as a tested exception."""

    def loss(x: Tensor) -> Tensor:
        """Simulate an interrupt during the loss computation."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_trials(loss, zero_sampler)


@pytest.mark.parametrize("invalid", ["duration", "max_examples"])
def test_invalid_configuration(zero_sampler: Sampler, invalid: str) -> None:
    """Reject negative budgets and evidence limits before starting any work."""
    with pytest.raises(ValueError):
        check_finiteness(
            square,
            zero_sampler,
            duration=timedelta(seconds=-1 if invalid == "duration" else 1),
            max_examples=-1 if invalid == "max_examples" else 10,
            quiet=True,
        )


def test_outer_no_grad_does_not_disable_checks(run_trials: Runner, zero_sampler: Sampler) -> None:
    """Explicitly enabled autograd keeps backward testing active inside no_grad callers."""
    with torch.no_grad():
        report = run_trials(square_root, zero_sampler, max_examples=0)
    assert report.random.nonfinite_gradient == 4
    assert report.examples == ()


def test_named_strategy_rates(run_trials: Runner) -> None:
    """A combined Monte Carlo rate cannot hide which distribution produced each failure."""
    labels = iter(("exact_zero", "uniform", "exact_zero", "uniform"))

    def sample(generator: torch.Generator) -> Sample:
        """Alternate a singular boundary and a safe value under distinct strategy labels."""
        strategy = next(labels)
        value = 0.0 if strategy == "exact_zero" else 1.0
        return Sample((torch.tensor(value, requires_grad=True),), strategy)

    report = run_trials(square_root, sample)
    assert report.random.rate("nonfinite_gradient") == 0.5
    assert report.strategies["exact_zero"].trials == 2
    assert report.strategies["exact_zero"].rate("nonfinite_gradient") == 1.0
    assert report.strategies["uniform"].trials == 2
    assert report.strategies["uniform"].rate("nonfinite_gradient") == 0.0
    assert sum(summary.trials for summary in report.strategies.values()) == report.random.trials


def test_diverse_evidence_replaces_duplicates(run_trials: Runner) -> None:
    """Late rare failures replace repeated evidence, including distinct exception stages."""
    values = iter((0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0))

    def sample(generator: torch.Generator) -> Inputs:
        """Feed common failures before four different rare failure categories."""
        return (torch.tensor(next(values), requires_grad=True),)

    def loss(x: Tensor) -> Tensor:
        """Select a known failure according to the explicitly sampled input value."""
        match int(x.item()):
            case 0:
                return square_root(x)
            case 1:
                return raises_forward(x)
            case 2:
                return raises_backward(x)
            case 3:
                return overflowing_forward(x)
            case _:
                return detached_loss(x)

    report = run_trials(loss, sample, trials=7, max_examples=5, required_gradients=(0,))
    assert report.random.failed_trials == 7
    assert len(report.examples) == 5
    assert {example.issues for example in report.examples} == {
        ("nonfinite_gradient",),
        ("exception",),
        ("nonfinite_loss",),
        ("missing_required_gradient",),
    }
    stages = {
        example.error.stage for example in report.examples if isinstance(example.error, Raised)
    }
    assert stages == {"forward", "backward"}
    assert report.examples[0].trial_index == 0


def test_evidence_diversity_includes_strategy(run_trials: Runner) -> None:
    """Identical numerical failures from different strategies retain separate representatives."""
    labels = iter(("common", "common", "common", "rare"))

    def sample(generator: torch.Generator) -> Sample:
        """Name each zero input by the distribution that produced it."""
        return Sample((torch.tensor(0.0, requires_grad=True),), next(labels))

    report = run_trials(square_root, sample, max_examples=2)
    assert len(report.examples) == 2
    assert {example.strategy for example in report.examples} == {"common", "rare"}
    assert report.random.nonfinite_gradient == 4


def test_optional_disconnection_is_coverage(run_trials: Runner, zero_sampler: Sampler) -> None:
    """An intentionally detached loss reports missing coverage without numerical failure."""
    report = run_trials(detached_loss, zero_sampler)
    assert report.random.disconnected == 4
    assert report.random.disconnected_inputs == {0: 4}
    assert report.random.failed_trials == 0
    assert report.random.missing_required_gradient == 0
    assert report.random.gradient_trials_checked == 0
    assert isinstance(report.random.max_abs_gradient, Absent)
    assert report.examples == ()


def test_required_disconnection_is_a_failure(run_trials: Runner, zero_sampler: Sampler) -> None:
    """A caller's explicit connectivity requirement turns missing coverage into failure."""
    report = run_trials(detached_loss, zero_sampler, required_gradients=(0,))
    assert report.random.failed_trials == 4
    assert report.random.missing_required_gradient == 4
    assert report.examples[0].disconnected_inputs == (0,)


def test_trial_limit_without_deadline(zero_sampler: Sampler) -> None:
    """A fixed count run works without an implicit wall-time limit."""
    report = check_finiteness(square, zero_sampler, max_trials=7, quiet=True)
    assert report.trials == 7
    assert report.stop_reason == "max_trials"
    assert isinstance(report.duration, Absent)


@pytest.mark.parametrize(
    "trials,max_trials,expected,reason", [(5, 2, 2, "max_trials"), (2, 5, 2, "duration")]
)
def test_first_limit_stops_run(
    run_trials: Runner,
    zero_sampler: Sampler,
    trials: int,
    max_trials: int,
    expected: int,
    reason: str,
) -> None:
    """Time and trial limits compete without letting failures alter the stopping rule."""
    report = run_trials(square_root, zero_sampler, trials=trials, max_trials=max_trials)
    assert report.trials == expected
    assert report.stop_reason == reason


def test_trial_limit_counts_boundaries(run_trials: Runner, zero_sampler: Sampler) -> None:
    """Fixed-count limits include deterministic boundaries as well as random trials."""
    report = run_trials(
        square,
        zero_sampler,
        trials=5,
        max_trials=2,
        boundary_inputs=[(torch.tensor(0.0, requires_grad=True),)] * 3,
    )
    assert report.boundary.trials == 2
    assert report.boundary_cases_requested == 3
    assert report.random.trials == 0


def test_fixed_count_reproducibility() -> None:
    """Seeded fixed-count runs produce the same sampled failure inputs."""

    def sample(generator: torch.Generator) -> Inputs:
        """Draw negative values so every input is preserved as failure evidence."""
        return (-torch.rand((), generator=generator).requires_grad_(),)

    first = check_finiteness(square_root, sample, max_trials=6, seed=81, quiet=True)
    second = check_finiteness(square_root, sample, max_trials=6, seed=81, quiet=True)
    assert first.trials == second.trials == 6
    for original, repeated in zip(first.examples, second.examples, strict=True):
        torch.testing.assert_close(original.inputs[0], repeated.inputs[0])


def test_gradient_scale_can_expose_overflow() -> None:
    """A finite float16 loss and unscaled gradient can overflow under loss scaling."""

    def sample(generator: torch.Generator) -> Inputs:
        """Provide a float16 input with finite square loss and derivative."""
        return (torch.tensor(20.0, dtype=torch.float16, requires_grad=True),)

    unscaled = check_finiteness(square, sample, max_trials=1, quiet=True)
    scaled = check_finiteness(square, sample, max_trials=1, gradient_scale=2048.0, quiet=True)
    assert unscaled.random.max_abs_loss == 400.0
    assert unscaled.random.max_abs_gradient == 40.0
    assert unscaled.random.failed_trials == 0
    assert scaled.random.nonfinite_loss == 0
    assert scaled.random.nonfinite_gradient == 1
    assert scaled.gradient_scale == 2048.0


def test_completed_trial_can_overrun_deadline(clock: ManualClock, zero_sampler: Sampler) -> None:
    """The checker stops between trials and truthfully reports elapsed overruns."""

    def loss(x: Tensor) -> Tensor:
        """Simulate one evaluation taking longer than the entire requested budget."""
        clock.seconds += 3
        return x.square()

    report = check_finiteness(loss, zero_sampler, duration=timedelta(seconds=1), quiet=True)
    assert report.trials == 1
    assert report.elapsed == timedelta(seconds=3)
    assert report.stop_reason == "duration"


@pytest.mark.parametrize("required", [(-1,), (1,)])
def test_invalid_required_indices(zero_sampler: Sampler, required: tuple[int, ...]) -> None:
    """Required connectivity indices must identify actual differentiated arguments."""
    with pytest.raises(ValueError, match="indices"):
        check_finiteness(
            square, zero_sampler, max_trials=1, required_gradients=required, quiet=True
        )


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_nonfinite_scale_is_invalid(zero_sampler: Sampler, scale: float) -> None:
    """The input gradient scale must itself start as a finite Python value."""
    with pytest.raises(ValueError, match="gradient_scale"):
        check_finiteness(square, zero_sampler, max_trials=1, gradient_scale=scale, quiet=True)


def test_missing_limits_are_invalid(zero_sampler: Sampler) -> None:
    """The checker never starts an implicitly unbounded run."""
    with pytest.raises(ValueError, match="Provide duration"):
        check_finiteness(square, zero_sampler, quiet=True)


def test_negative_trial_limit_is_invalid(zero_sampler: Sampler) -> None:
    """Reject a negative trial limit before evaluation."""
    with pytest.raises(ValueError, match="max_trials"):
        check_finiteness(square, zero_sampler, max_trials=-1, quiet=True)


def test_zero_trial_limit(zero_sampler: Sampler) -> None:
    """A zero count produces an empty report with a precise stopping reason."""
    report = check_finiteness(square, zero_sampler, max_trials=0, quiet=True)
    assert report.trials == 0
    assert report.stop_reason == "max_trials"


def test_empty_strategy_is_invalid() -> None:
    """Unidentified named samples cannot silently merge into an aggregate."""
    with pytest.raises(ValueError, match="strategy"):
        Sample((torch.tensor(0.0, requires_grad=True),), " ")
