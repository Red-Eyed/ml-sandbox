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

3. [Which prediction should match this text box?](06_hungarian_matcher/matcher_lab.ipynb)
   is a live oriented-box Hungarian matcher lab. Move, resize, and rotate boxes; change
   confidences and costs; inspect assignments, the cost matrix, and pair contributions.

## Interactive Hungarian Matcher Lab

In VS Code, open `06_hungarian_matcher/matcher_lab.ipynb`, select the repository's
`.venv/bin/python` notebook kernel, and run the **first code cell**. It launches the
live editor independently of the later examples. Keep the kernel running. The Microsoft
Python and Jupyter extensions are required; the installed ipympl backend supports
[VS Code notebooks](https://matplotlib.org/ipympl/).

For JupyterLab, run `./setup.sh`, then
`uv run jupyter lab 06_hungarian_matcher/matcher_lab.ipynb` and run the first code cell
or choose **Run All Cells**. The locked environment includes NumPy, SciPy, Matplotlib,
ipympl, and ipywidgets. Use this project's Python 3.12 kernel. The live UI requires
JupyterLab's widget frontend; GitHub displays the saved static SVG examples.

- Click and drag a green GT or blue prediction. Select overlapping boxes by identifier.
- Drag square corner handles to resize along rotated local axes, keeping the opposite
  corner fixed. Drag the yellow circle to rotate about the center. Angles are
  counterclockwise degrees in the editor, radians in the backend; y increases upward.
- Add/remove either box type, edit numeric geometry, and adjust each prediction's
  independent foreground-confidence slider. Numeric edits apply on Enter or blur.
- Enable classification, normalized L1, true rotated IoU, Gaussian ProbIoU, or Hausdorff distance and adjust
  their weights. Similarities support `1-S` and `-log(max(S, epsilon))`; classification
  always uses `-log(max(p, epsilon))`. L1 and Hausdorff normalization scales stay fixed during zoom.
- Inspect the combined matrix or an enabled component's unweighted transformed costs.
  Magenta outlines always show the combined assignment. Pair selectors distinguish raw
  metrics, transformed costs, weighted contributions, and the total.
- Use starter scenes, hide arrows, fit the viewport, or reset the matcher baseline.
  Dashed boxes and the status line identify unmatched predictions and targets.

The explicit Phase 1 baseline uses classification 1 + L1 5 + rotated IoU 2. It is an OBB
prototype baseline, **not standard DETR's GIoU configuration**. Rectangular matching always
assigns `min(P, G)` pairs; there are no implicit thresholds or dummy assignments. Empty
scenes work. Tied optima may change identity. With one foreground class, classification
is constant across each row: it can select rows when P > G but cannot change a square
assignment by itself. Gaussian tails can underflow at extreme separation; the log floor
is visible and configurable. Invalid geometry or nonfinite costs produce a message.

Choose **Geometry only: Hausdorff** to test the WACV 2025 paper's sampled symmetric
Hausdorff matching distance with classification excluded. **Hausdorff points** selects
4 corners, 8 corners/midpoints, or 32 evenly sampled boundary points (8 per edge).
The score is the larger of the two directed maximum nearest-point distances; lower
is better. It is an unbounded distance and ignores the similarity-transform selector.
It is an exact distance between sampled point sets, approximating boundary distance.
Changing point count changes the approximation; it need not change scores monotonically.

Normalization follows the [official RHINO HausdorffCost](https://github.com/SIAnalytics/RHINO/blob/main/mmrotate/models/task_modules/assigners/match_cost.py):
divide `cx,w` by **x scale** and `cy,h` by **y scale**, preserve the angle, then construct
rotated boundary points. The defaults are 100 for both. Unequal scales differ from
scaling world-space corners after rotation. These shared controls also normalize L1.
The selected-pair inspector displays this normalized distance as both raw metric and
unweighted cost, then separately shows its weighted contribution.

Try **Oversized competing prediction**, a constructed illustration of the screenshot
failure, and switch between **Geometry only: ProbIoU** and **Geometry only: Hausdorff**.
The latter chooses P1 → G1 and P2 → G2. The former chooses P3 → G1 and P2 → G2.
Try 4/8/32 points while moving and rotating the same boxes. This tests the geometric
term alone; RHINO's full configuration also uses classification and a KLD term, and
its adaptive denoising is outside this lab. See [the paper](https://arxiv.org/html/2305.07598v5).
The backend uses NumPy for matching; this addition does not introduce a differentiable
training loss. Restart an already running notebook kernel after updating the package.

Motion computations are limited to 30 requests per second, with a final evaluation on
release. This small-scene prototype redraws the scene and heatmap together; frame rate
depends on scene size and frontend latency. Continuous fields, GIoU/DIoU, side-by-side
matchers, stability sweeps, random generators, and scene export are deferred to later phases.
More intuitive matching does not establish better detection accuracy.

To execute the lab and refresh its saved SVG examples without persisting widget state:

```bash
uv run 06_hungarian_matcher/execute.py
```

The numerical core and mouse-event/widget tests are co-located in
`src/ml_sandbox/matcher_lab/` and run through `./checks.sh`. Re-running the live
initialization cell closes its previous app instance. Use `lab.close()` to release it
explicitly. See the [ProbIoU paper](https://arxiv.org/abs/2106.06072),
[SciPy assignment semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html),
and [ipympl documentation](https://matplotlib.org/ipympl/) for the underlying definitions.

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
renders exact tensor values and tables as HTML-capable notebook output. Saved matcher-lab examples also use vector SVG. Its live Jupyter controls use widgets at
runtime; the saved examples remain readable without JavaScript or widget state.
