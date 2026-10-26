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
live editor. The notebook contains only launch instructions and this editor. Keep the kernel running. The Microsoft
Python and Jupyter extensions are required; the installed ipympl backend supports
[VS Code notebooks](https://matplotlib.org/ipympl/).

For JupyterLab, run `./setup.sh`, then
`uv run jupyter lab 06_hungarian_matcher/matcher_lab.ipynb` and run the first code cell
or choose **Run All Cells**. The locked environment includes NumPy, SciPy, Matplotlib,
ipympl, and ipywidgets. Use this project's Python 3.11 kernel. The live UI requires
a widget frontend in VS Code or JupyterLab; GitHub displays the launch instructions only.

- Click and drag a green GT or blue prediction. Select overlapping boxes by identifier.
- Drag square corner handles to resize along rotated local axes, keeping the opposite
  corner fixed. Drag the yellow circle to rotate about the center. Angles are
  counterclockwise degrees in the editor, radians in the backend; y increases upward.
- Add/remove either box type, edit numeric geometry, and adjust each prediction's
  independent foreground-confidence slider. Numeric edits apply on Enter or blur.
- Enable classification, normalized L1, true rotated IoU, Gaussian ProbIoU, Hausdorff, or Gaussian distances and adjust
  their weights. Similarities support `1-S` and `-log(max(S, epsilon))`; classification
  always uses `-log(max(p, epsilon))`. L1 and Hausdorff normalization scales stay fixed during zoom.
- Inspect the combined matrix or any geometry-only metric's unweighted transformed costs.
  Magenta outlines show the inspected method's own assignment. Pair selectors distinguish
  raw metrics, transformed costs, weighted contributions, and the total.
- Compare all ten geometry-only metrics simultaneously on the same editable boxes.
  Click a metric panel (or use **Inspect**) to focus its own assignment and cost matrix.
  **Combined** retains the separately configurable weighted matcher. Geometry-only panels
  ignore its weights and confidences; they share the representation and distance settings.
  Totals use different units and cannot rank metrics.
- Use starter scenes, hide arrows, fit the viewport, or reset the matcher baseline.
  Dashed boxes and the status line identify unmatched predictions and targets.

The editor starts with **DETR-style: L1 + ProbIoU**: normalized L1 × 5 plus
ProbIoU cost × 2, excluding classification. The default similarity transform is `1-S`.
Original DETR uses classification + L1 + GIoU; this preset substitutes ProbIoU for geometry.
The separate Phase 1 baseline uses classification 1 + L1 5 + rotated IoU 2. It is an OBB
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

Gaussian comparison presets now include **KL P → GT**, **KL GT → P**,
**Symmetric KL (mean)**, **Gaussian W2**, **Bhattacharyya**, and **Hellinger**.
All use the same `box_to_gaussian` converter as ProbIoU. **Gaussian covariance**
selects the uniform-rectangle convention `R diag(w²,h²) Rᵀ / 12` (default) or the
half-extent convention `R diag(w²,h²) Rᵀ / 4`. Presets restore the default convention;
changing the dropdown applies one shared representation to every enabled Gaussian term.

The converter validates geometry and stores immutable means, major/minor standard
deviations, and principal-axis orientation. Distances work from covariance factors;
they do not invert the dense matrix. The pair inspector shows both means and dense
covariances plus the retained principal standard deviations. Extreme aspect ratios
can lose their small eigenvalue in a dense display matrix; factor calculations retain it.
Unrepresentable variances and nonfinite distances are rejected, without covariance jitter.
Gaussian approximations lose the orientation of square boxes regardless of distance.

Symmetric KL is `(KL(P || GT) + KL(GT || P)) / 2`, explicitly the mean, and is a
symmetric divergence rather than a triangle-inequality metric. KL and Bhattacharyya
are dimensionless. **Gaussian W2 is unsquared and uses scene units**. Hellinger is
unsquared and bounded in `[0,1]`; it equals `1 - ProbIoU` mathematically, so it is not
an independent geometric representation. These distance costs receive no similarity
transform, clipping, or hidden normalization. The x/y scale controls apply only to
L1/Hausdorff. Raw totals from different methods are not directly comparable.

The public API accepts the lab's Cartesian `Box`, or `NormalizedBox` for detector
records `(cx, cy, w, h, cosine, sine)` in `[0,1]`:

```python
from ml_sandbox.matcher_lab import NormalizedBox, normalized_box_to_gaussian
from ml_sandbox.matcher_lab.gaussian_kl import symmetric_kl

encoded = NormalizedBox("P1", 0.25, 0.5, 0.1, 0.1, 0.7, 0.7)
gaussian = normalized_box_to_gaussian(encoded, image_width=800, image_height=400)
```

The adapter decodes `(2*cosine-1, 2*sine-1)`, normalizes that direction, rejects the
undefined `(0.5,0.5)` pair, and denormalizes center/dimensions before rotating.
It requires explicit image dimensions and preserves the caller's axis convention.
The returned mean is `(200,200)` and the physical rectangle has sides 80 and 40.
These NumPy diagnostics support matching experiments, not autograd training losses.

Motion computations are limited to 30 requests per second, with a final evaluation on
release. Plot objects are reused, layout is recalculated on resize, and pending redraws are
coalesced. Frame rate depends on scene size and frontend latency. Continuous fields,
GIoU/DIoU, stability sweeps, random generators, and scene export are deferred to later phases.
More intuitive matching does not establish better detection accuracy.

To verify that the interactive editor initializes and save the notebook without runtime outputs:

```bash
uv run 06_hungarian_matcher/execute.py
```

The numerical core and mouse-event/widget tests are co-located in
`src/ml_sandbox/matcher_lab/` and run through `./checks.sh`. Re-running the live
initialization cell closes its previous app instance. Use `lab.close()` to release it
explicitly. The live frontend uses [ipympl](https://matplotlib.org/ipympl/).

### Matching method references

| Method | Primary reference | Relationship to this lab |
|---|---|---|
| Normalized L1 | Carion et al., **End-to-End Object Detection with Transformers** (ECCV 2020): [paper](https://arxiv.org/abs/2005.12872), [official matcher](https://github.com/facebookresearch/detr/blob/main/models/matcher.py) | Sum of absolute center/size errors. Configurable scene scales and the optional periodic angle term are lab adaptations. |
| Rotated IoU | [MMCV rotated IoU definition](https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.box_iou_rotated.html); Rezatofighi et al., **Generalized Intersection over Union** (CVPR 2019), [IoU definition](https://arxiv.org/abs/1902.09630) | Actual oriented-rectangle intersection/union, implemented here with polygon clipping. The lab uses Cartesian y-up coordinates and counterclockwise radians; convert conventions when comparing with image-coordinate libraries. GIoU itself is not implemented. |
| ProbIoU | Llerena et al., **Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection**: [2021 preprint](https://arxiv.org/abs/2106.06072), [TIP 2024 publication](https://doi.org/10.1109/TIP.2023.3348697) | Gaussian covariance from rotated dimensions; similarity is one minus Hellinger distance. |
| Gaussian Bhattacharyya distance | [ProbIoU paper, Gaussian distance derivation](https://arxiv.org/abs/2106.06072) | The unbounded distance underlying ProbIoU; also selectable directly as a geometry-only cost. |
| Gaussian KL, both directions | Yang et al., **Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence** (NeurIPS 2021): [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html) | Closed-form Gaussian KL. The symmetric mean is a lab-defined comparison of both directions; no detection-loss transforms are applied. |
| Gaussian Wasserstein W2 | Yang et al., **Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss** (ICML 2021): [paper](https://proceedings.mlr.press/v139/yang21l.html) | Raw unsquared W2, evaluated by factor alignment. The covariance convention is explicit and the paper's extra loss transforms are not reproduced. |
| Hellinger distance | [ProbIoU paper](https://arxiv.org/abs/2106.06072) | Unsquared Gaussian Hellinger, equal to one minus ProbIoU. |
| Sampled Hausdorff | Lee et al., **Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer** (WACV 2025): [paper, §3.4 and appendix C](https://arxiv.org/html/2305.07598v5), [official RHINO cost](https://github.com/SIAnalytics/RHINO/blob/main/mmrotate/models/task_modules/assigners/match_cost.py) | Symmetric maximum nearest-point distance, with RHINO parameter normalization and 4/8/32 boundary samples. The standalone preset excludes the other RHINO costs and adaptive denoising. |
| Classification: negative log probability | [PyTorch negative log-likelihood definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html); [official DETR matching classification cost](https://github.com/facebookresearch/detr/blob/main/models/matcher.py) | The lab deliberately uses `-log(max(p, epsilon))`. Original DETR matching uses `-p`; these are different formulations. |
| Rectangular assignment | [SciPy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Reference solver, assigning exactly the smaller matrix dimension without implicit thresholds or dummy pairs. |

The selectable `1-S` and `-log(max(S, epsilon))` similarity-to-cost transforms are lab
analysis settings. They are not claimed to reproduce each paper's full matcher or training loss.

## Reuse the Package

Install this checkout into another Python 3.11 environment:

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
renders exact tensor values and tables as HTML-capable notebook output. The interactive matcher lab uses Jupyter widgets at
runtime; the saved examples remain readable without JavaScript or widget state.
