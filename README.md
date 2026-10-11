# ML Sandbox

ML Sandbox is a Jupyter-first collection of small experiments that reveal the tensors and
state hidden by high-level deep-learning APIs. Each notebook starts with one concrete question,
implements the mechanism with basic PyTorch tensor operations, changes one variable, and checks
the result.

The notebooks are the project. The installable `ml_sandbox` package provides display tooling and
reusable numerical diagnostics. Core ML behavior stays visible in notebook cells where it can
be inspected and modified.

## Current Scope

The repository grows one reviewed experiment at a time. Each completed notebook stands alone and
keeps the mechanism being taught visible in its cells rather than moving it into a training
framework.

## Experiments

1. [Where does a parameter gradient live?](01_parameters/parameter_and_gradient.ipynb) inspects
   parameter values, the autograd graph, hand-derived gradients, per-example contributions,
   accumulation across backward passes, and resetting `.grad` to `None`.
2. [Can a finite loss still have nonfinite gradients?](05_numerical_stability/numerical_stability.ipynb)
   compares square-root losses, checks boundary cases, and demonstrates a time-budgeted
   Monte Carlo tester that aggregates failures instead of stopping at the first one.

## Reuse the Package

Install this checkout into another Python 3.12 environment:

```bash
uv pip install --editable /path/to/ml-sandbox
```

`check_finiteness` searches for nonfinite scalar losses and input gradients. Supply finite tensor
inputs within your intended bounds, with the shapes, dtype, and device you want to test. A sampler
can return a tuple of tensors or a labeled `Sample` for per-strategy statistics:

```python
from datetime import timedelta

import torch
from torch import Tensor

from ml_sandbox.numerical_finiteness import Sample, check_finiteness


def loss_fn(x: Tensor) -> Tensor:
    """Return a loss whose derivative is singular at zero."""
    return x.sqrt()


def sample_inputs(generator: torch.Generator) -> Sample:
    """Draw a float32 input in [0, 1] using the supplied private generator."""
    x = torch.rand((), generator=generator, dtype=torch.float32).requires_grad_()
    return Sample(inputs=(x,), strategy="uniform")


report = check_finiteness(
    loss_fn,
    sample_inputs,
    duration=timedelta(seconds=10),
    boundary_inputs=[(torch.tensor(0.0, requires_grad=True),)],
    required_gradients=(0,),
    max_examples=10,
    seed=42,
)

print(report.boundary.nonfinite_gradient)
print(report.strategies["uniform"].rate("nonfinite_gradient"))
print(report.random.failed_trials)
```

Use `max_trials=10_000` instead of `duration` for a reproducible fixed-count run; supplying both
stops at the first reached limit. Boundary cases count toward both limits. A single trial can
overrun the time budget, which is checked between trials. Pass `quiet=True` to hide progress.

The report separates deterministic boundaries, combined random outcomes, and each named sampling
strategy. It includes trial counts, overlapping issue rates, stage coverage, finite absolute
maxima, stopping reason, and bounded replayable failure evidence. When storage is full, new failure
groups replace duplicate examples where possible; every failure still contributes to counts.
Rates describe the selected sampling distribution, and zero observed failures is not a guarantee.

Disconnected differentiated inputs are coverage information. They count as failures only for input
indices listed in `required_gradients`; these inputs must have `requires_grad=True`. Constants can
remain nondifferentiable. The tester uses fresh input copies and `autograd.grad`, so caller `.grad`
fields are preserved. To check model parameter gradients, expose parameters as differentiated
arguments rather than relying on parameters captured by a closure.

Set `gradient_scale` to test an incoming gradient other than one, such as a mixed-precision loss
scale. The scale is cast to the loss dtype; report maxima and saved gradients reflect that scale.
Forward/backward exceptions are recorded and sampling continues. Sampler errors, invalid inputs,
and user interrupts propagate. Reproducible inputs require using the supplied generator and a
deterministic loss. A saved failure can be replayed with `report.examples[0].replay_inputs()`.

## Run the Experiments

Install the locked environment and the project in editable mode, then launch JupyterLab:

```bash
./setup.sh
uv run jupyter lab
```

Run the complete validation suite:

```bash
./checks.sh
```

The checks lint and type-check the package, execute every notebook currently present, enforce the
notebook-size and no-image policies, and verify that the installable package builds. They also pass
when the repository intentionally contains no notebooks.

Altair renders charts as compact inline SVG that remains visible in static GitHub previews. Rich
renders exact tensor values and tables as HTML-capable notebook output. The repository contains no
generated image files or JavaScript-dependent chart outputs.
