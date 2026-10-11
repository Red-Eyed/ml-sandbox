"""ProbIoU on single-angle direction vectors, with a numerical baseline.

The baseline evaluates the published Ultralytics equations (eps=1e-7,
floor=0, CIoU=False), rather than depending on the detector package.
Source: https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.probiou
"""

import math

import torch
from torch import Tensor


def _validate(first: Tensor, second: Tensor) -> None:
    """Reject invalid geometry before either numerical path evaluates it."""
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("Inputs must have the same device and dtype")
    if first.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Inputs must be floating-point tensors")
    for boxes in (first, second):
        if boxes.ndim == 0 or boxes.shape[-1] != 6:
            raise ValueError("Expected (..., 6): cx, cy, w, h, cos, sin")
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError("All coordinates must be finite")
        if not bool((boxes[..., 2:4] > 0).all()):
            raise ValueError("Widths and heights must be strictly positive")
        if not bool((boxes[..., 4:6].abs().amax(dim=-1) > 0).all()):
            raise ValueError("Direction vectors must be nonzero")


def _unit_direction(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize a nonzero direction without overflowing its squared norm."""
    direction = boxes[..., 4:6]
    scale = direction.abs().amax(dim=-1, keepdim=True).detach()
    direction = direction / scale
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    return direction[..., 0], direction[..., 1]


def stable_probiou(first: Tensor, second: Tensor) -> Tensor:
    """Return Gaussian OBB similarity in [0, 1], with broadcast leading dimensions.

    Coordinates are (cx, cy, w, h, cos(theta), sin(theta)); positive sides
    and finite, nonzero directions are required. Directions are normalized,
    so gradients include that normalization. No inputs are mutated.

    Float64 arithmetic is used on CPU/CUDA, returning the input dtype. MPS
    raises ValueError because it cannot perform the required float64 work.
    Pairwise side ratios below 1e-30 or offsets above 1e30 times the largest
    side are rejected rather than silently regularized. Invalid geometry
    raises ValueError; incompatible broadcasting follows PyTorch semantics.

    Equality returns exactly 1 with the zero norm subgradient. This is a
    chosen subgradient at a nondifferentiable point, not a smooth derivative.
    Translation residuals beyond 32 are saturated, where exp(-BD) is already
    below float64 resolution. No epsilon plateau is introduced near equality.
    Gradients cast back to the input dtype can overflow if the mathematical
    gradient is outside that dtype's range. Validation synchronizes devices.
    """
    _validate(first, second)
    if first.device.type == "mps":
        raise ValueError("Stable ProbIoU requires float64; use CPU or CUDA")
    first64, second64 = first.double(), second.double()
    scale = torch.maximum(first64[..., 2:4].amax(dim=-1), second64[..., 2:4].amax(dim=-1)).detach()
    sides1, sides2 = first64[..., 2:4] / scale[..., None], second64[..., 2:4] / scale[..., None]
    delta = (second64[..., :2] - first64[..., :2]) / scale[..., None]
    if not bool((torch.minimum(sides1, sides2) >= 1e-30).all()):
        raise ValueError("Pairwise dimension ratio is below the supported 1e-30")
    if not bool((delta.abs() <= 1e30).all()):
        raise ValueError("Pairwise center offset exceeds the supported 1e30 scale")
    p, r = sides1[..., 0] / math.sqrt(12), sides1[..., 1] / math.sqrt(12)
    q, s = sides2[..., 0] / math.sqrt(12), sides2[..., 1] / math.sqrt(12)
    c1, s1 = _unit_direction(first64)
    c2, s2 = _unit_direction(second64)
    cosine, sine = c1 * c2 + s1 * s2, c1 * s2 - s1 * c2
    root_product = p * r * q * s
    shape_residual = torch.stack(
        (p * r - q * s, cosine * (p * s - r * q), sine * (p * q - r * s)), dim=-1
    ) / (2 * root_product.sqrt()[..., None])
    shape_squared = shape_residual.square().sum(dim=-1)
    average_determinant = root_product * (1 + shape_squared)
    dx, dy = delta[..., 0] * c1 + delta[..., 1] * s1, -delta[..., 0] * s1 + delta[..., 1] * c1
    qx, qy = dx * cosine + dy * sine, -dx * sine + dy * cosine
    translation_residual = torch.stack((r * dx, p * dy, s * qx, q * qy), dim=-1)
    translation_residual = (
        translation_residual / (4 * average_determinant.sqrt()[..., None])
    ).clamp(-32, 32)
    # Both ratios have removable singularities. Safe denominators also keep
    # the inactive torch.where branch finite during backward.
    safe_shape = torch.where(shape_squared > 0, shape_squared, torch.ones_like(shape_squared))
    shape_factor = torch.where(
        shape_squared > 0, 0.5 * torch.log1p(shape_squared) / safe_shape, 0.5
    ).sqrt()
    residual = torch.cat((translation_residual, shape_residual * shape_factor[..., None]), dim=-1)
    root_distance = torch.linalg.vector_norm(residual, dim=-1)
    distance = root_distance.square()
    safe_distance = torch.where(distance > 0, distance, torch.ones_like(distance))
    factor = torch.where(distance > 0, -torch.expm1(-distance) / safe_distance, 1.0).sqrt()
    return (1 - root_distance * factor).clamp(0, 1).to(first.dtype)


def ultralytics_probiou(first: Tensor, second: Tensor, eps: float = 1e-7) -> Tensor:
    """Evaluate Ultralytics' standard equations in the input dtype.

    Adapt six-coordinate inputs through atan2 to its xywhr geometry. Leading
    dimensions broadcast and output has no trailing singleton. The shared
    input validator is stricter than upstream; arithmetic, epsilon placement,
    determinant clamps, and distance clamps preserve the baseline behavior.
    This is an equation reproduction, not an installed-package integration.
    Nonfinite forward/backward values are deliberately retained for diagnosis.
    """
    _validate(first, second)
    a, b, c = _baseline_covariance(first)
    d, e, f = _baseline_covariance(second)
    dx, dy = first[..., 0] - second[..., 0], first[..., 1] - second[..., 1]
    determinant = (a + d) * (b + e) - (c + f).square()
    translation = ((a + d) * dy.square() + (b + e) * dx.square()) / (determinant + eps) / 4
    cross = -(c + f) * dx * dy / (determinant + eps) / 2
    product = (a * b - c.square()).clamp_min(0) * (d * e - f.square()).clamp_min(0)
    shape = (determinant / (4 * product.sqrt() + eps) + eps).log() / 2
    distance = (translation + cross + shape).clamp(eps, 100)
    return 1 - (1 - (-distance).exp() + eps).sqrt()


def _baseline_covariance(boxes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Expand rotated covariance as the angle-format baseline does."""
    angle = torch.atan2(boxes[..., 5], boxes[..., 4])
    cosine, sine = angle.cos(), angle.sin()
    w, h = boxes[..., 2].square() / 12, boxes[..., 3].square() / 12
    return (
        w * cosine.square() + h * sine.square(),
        w * sine.square() + h * cosine.square(),
        (w - h) * cosine * sine,
    )
