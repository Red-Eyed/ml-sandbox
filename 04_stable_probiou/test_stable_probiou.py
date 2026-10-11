"""Forward properties and backward diagnostics for both ProbIoU paths."""

import math
from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
from stable_probiou import stable_probiou, ultralytics_probiou
from torch import Tensor


class StressProbe(TypedDict):
    """Geometry and independent forward/backward status for a stress tile."""

    scale: int
    aspect: int
    path: str
    dtype: str
    status: str


Score = Callable[[Tensor, Tensor], Tensor]
PATHS = [stable_probiou, ultralytics_probiou]


def box(w: float = 2, h: float = 1, angle: float = 0.3, x: float = 0) -> Tensor:
    """Build an independent float64 leaf geometry."""
    return torch.tensor([x, 0.0, w, h, math.cos(angle), math.sin(angle)], dtype=torch.float64)


def matrix_reference(first: Tensor, second: Tensor) -> Tensor:
    """Compute independent BD using float64 matrix solves and log determinants."""
    matrices = []
    for boxes in (first, second):
        direction = boxes[..., 4:6] / torch.linalg.vector_norm(
            boxes[..., 4:6], dim=-1, keepdim=True
        )
        c, s = direction[..., 0], direction[..., 1]
        rotation = torch.stack((c, -s, s, c), dim=-1).reshape(*boxes.shape[:-1], 2, 2)
        covariance = rotation @ torch.diag_embed(boxes[..., 2:4].square() / 12) @ rotation.mT
        matrices.append(covariance)
    average = (matrices[0] + matrices[1]) / 2
    delta = first[..., :2] - second[..., :2]
    translation = (delta * torch.linalg.solve(average, delta[..., None]).squeeze(-1)).sum(-1) / 8
    shape = (
        torch.linalg.slogdet(average).logabsdet
        - (
            torch.linalg.slogdet(matrices[0]).logabsdet
            + torch.linalg.slogdet(matrices[1]).logabsdet
        )
        / 2
    ) / 2
    return 1 - (-torch.expm1(-(translation + shape).clamp_min(0))).sqrt()


@pytest.mark.parametrize("score", PATHS)
def test_forward_against_independent_reference(score: Score) -> None:
    """Check random ordinary geometry, allowing the baseline's epsilon bias."""
    generator = torch.Generator().manual_seed(81)
    first = torch.randn(200, 6, generator=generator, dtype=torch.float64)
    second = torch.randn(200, 6, generator=generator, dtype=torch.float64)
    first[:, 2:4] = first[:, 2:4].exp() + 0.5
    second[:, 2:4] = second[:, 2:4].exp() + 0.5
    actual = score(first, second)
    assert bool(torch.isfinite(actual).all())
    tolerance = 2e-6 if score is ultralytics_probiou else 2e-12
    torch.testing.assert_close(actual, matrix_reference(first, second), atol=tolerance, rtol=0)
    torch.testing.assert_close(actual, score(second, first), atol=2e-12, rtol=0)


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_identity_forward_backward(score: Score, dtype: torch.dtype) -> None:
    """Equality has a finite zero gradient; only the stable score is exactly one."""
    first, second = box().to(dtype).requires_grad_(), box().to(dtype).requires_grad_()
    value = score(first, second)
    gradients = torch.autograd.grad(value, (first, second))
    assert bool(torch.isfinite(value))
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    if score is stable_probiou:
        assert value.item() == 1
    elif dtype == torch.float64:
        assert 0.999 < value.item() < 1


@pytest.mark.parametrize("score", PATHS)
def test_gradcheck_both_inputs(score: Score) -> None:
    """Numerical perturbations exercise all twelve independent coordinates."""
    first, second = box(x=0.2).requires_grad_(), box(1.4, 0.8, -0.4).requires_grad_()
    assert torch.autograd.gradcheck(score, (first, second), atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("score", PATHS)
def test_backward_independent_reference(score: Score) -> None:
    """Compare both argument gradients to a differentiable matrix formula."""
    first, second = box(x=0.2).requires_grad_(), box(1.4, 0.8, -0.4).requires_grad_()
    actual = torch.autograd.grad(score(first, second), (first, second))
    expected = torch.autograd.grad(matrix_reference(first, second), (first, second))
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("offset", [1e-10, 1e-6, 1e-3, 1.0, 100.0])
def test_near_identity_and_saturation(score: Score, offset: float) -> None:
    """Check analytic center-shift scores and one-sided gradient near the cusp."""
    first, second = box(angle=0).requires_grad_(), box(angle=0, x=offset).requires_grad_()
    value = score(first, second)
    gradients = torch.autograd.grad(value, (first, second))
    assert bool(torch.isfinite(value))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    if score is stable_probiou:
        distance = 1.5 * (offset / 2) ** 2
        expected = 1 - math.sqrt(-math.expm1(-distance))
        assert value.item() == pytest.approx(expected, abs=1e-14)
        if offset < 1e-3:
            assert gradients[1][0].item() == pytest.approx(-math.sqrt(1.5) / 2, rel=1e-6)
    elif offset <= 1e-6:
        assert gradients[1][0].item() == 0


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("representation", ["antipodal", "swapped", "square"])
def test_equivalent_geometry(score: Score, representation: str) -> None:
    """Equivalent Gaussian representations preserve score and finite backward."""
    first = box(angle=0)
    second = first.clone()
    if representation == "antipodal":
        second[4:6] *= -1
    elif representation == "swapped":
        second[2:4] = first[2:4].flip(0)
        second[4:6] = torch.tensor([0.0, 1.0])
    else:
        first[3] = first[2]
        second = box(2, 2, 0.8)
    first.requires_grad_()
    second.requires_grad_()
    value = score(first, second)
    assert value.item() == pytest.approx(score(first, first).item(), abs=1e-12)
    assert all(bool(torch.isfinite(g).all()) for g in torch.autograd.grad(value, (first, second)))


@pytest.mark.parametrize("score", PATHS)
def test_broadcast_backward_empty_and_noncontiguous(score: Score) -> None:
    """Broadcast reductions, strided tensors, and empty batches keep their shapes."""
    first = torch.stack([box(x=0.2), box(x=0.4)]).requires_grad_()
    second = torch.stack([box(1.2), box(1.5), box(1.8)]).requires_grad_()
    values = score(first[:, None, :], second[None, :, :])
    assert values.shape == (2, 3)
    gradients = torch.autograd.grad(values.sum(), (first, second))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    strided = torch.stack((first, first), dim=-1)[..., 0]
    torch.testing.assert_close(score(strided, box()), score(first, box()))
    assert score(torch.empty(0, 6, dtype=torch.float64), box()).shape == (0,)


@pytest.mark.parametrize("scale", [1e-30, 1e-15, 1.0, 1e15, 1e30])
@pytest.mark.parametrize("aspect", [1.0, 1e4, 1e12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stable_scale_and_aspect_forward_backward(
    scale: float, aspect: float, dtype: torch.dtype
) -> None:
    """The stable path remains finite and obeys scale equivariance for gradients."""
    first, second = box(2, 2 / aspect, x=0.1), box(1.4, 1.4 / aspect, 0.31)
    first[:4] *= scale
    second[:4] *= scale
    first, second = first.to(dtype).requires_grad_(), second.to(dtype).requires_grad_()
    if not bool((first[2:4] > 0).all()):
        pytest.skip("Geometry underflows the input dtype before evaluation")
    value = stable_probiou(first, second)
    gradients = torch.autograd.grad(value, (first, second))
    assert 0 <= value.item() <= 1
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    if dtype == torch.float64:
        unscaled1, unscaled2 = (
            box(2, 2 / aspect, x=0.1).requires_grad_(),
            box(1.4, 1.4 / aspect, 0.31).requires_grad_(),
        )
        reference = stable_probiou(unscaled1, unscaled2)
        reference_gradients = torch.autograd.grad(reference, (unscaled1, unscaled2))
        torch.testing.assert_close(value, reference, atol=2e-14, rtol=1e-12)
        for gradient, expected in zip(gradients, reference_gradients, strict=True):
            multiplier = torch.tensor([scale] * 4 + [1.0] * 2, dtype=torch.float64)
            torch.testing.assert_close(gradient * multiplier, expected, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("index,value", [(2, 0), (3, -1), (0, float("inf")), (5, float("nan"))])
def test_invalid_geometry(score: Score, index: int, value: float) -> None:
    """Both paths reject invalid inputs instead of manufacturing scores."""
    invalid = box()
    invalid[index] = value
    with pytest.raises(ValueError):
        score(invalid.requires_grad_(), box())


@pytest.mark.parametrize("score", PATHS)
def test_invalid_shape_dtype_direction(score: Score) -> None:
    """Enforce six floating coordinates and nonzero directions."""
    for invalid in (torch.ones(5), torch.ones(6, dtype=torch.int64), box()):
        if invalid.dtype == torch.float64:
            invalid[4:6] = 0
        with pytest.raises(ValueError):
            score(invalid, invalid)
    with pytest.raises(ValueError):
        score(box().float(), box())


def test_baseline_known_failure_forward_backward() -> None:
    """Preserve the baseline's determinant cancellation and overflow diagnostics."""
    first = box(1, 1e-12, 0).float().requires_grad_()
    second = first.detach().clone().requires_grad_()
    score = ultralytics_probiou(first, second)
    gradients = torch.autograd.grad(score, (first, second))
    assert not all(bool(torch.isfinite(g).all()) for g in gradients)
    large = box(1e20, 1e20).float().requires_grad_()
    value = ultralytics_probiou(large, large.detach())
    assert not bool(torch.isfinite(value))
    assert not bool(torch.isfinite(torch.autograd.grad(value, large)[0]).all())


def test_stable_rejects_outside_documented_domain() -> None:
    """An unsupported dynamic range fails explicitly rather than giving NaNs."""
    with pytest.raises(ValueError, match="dimension ratio"):
        stable_probiou(box(1, 1e-40), box())
    with pytest.raises(ValueError, match="center offset"):
        stable_probiou(box(x=1e40), box())


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_random_forward_backward_properties(score: Score, dtype: torch.dtype) -> None:
    """Exercise every coordinate over a seeded batch in every supported dtype."""
    generator = torch.Generator().manual_seed(901)
    first = torch.randn(64, 6, generator=generator, dtype=torch.float64)
    second = torch.randn(64, 6, generator=generator, dtype=torch.float64)
    first[:, 2:4] = first[:, 2:4].abs() + 1
    second[:, 2:4] = second[:, 2:4].abs() + 1
    first, second = first.to(dtype).requires_grad_(), second.to(dtype).requires_grad_()
    values = score(first, second)
    gradients = torch.autograd.grad(values.sum(), (first, second))
    assert bool(torch.isfinite(values).all())
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    tolerance = 0.02 if dtype in (torch.float16, torch.bfloat16) else 2e-6
    assert bool(((values >= -tolerance) & (values <= 1)).all())
    reverse_gradients = torch.autograd.grad(score(second, first).sum(), (first, second))
    for gradient, reverse in zip(gradients, reverse_gradients, strict=True):
        torch.testing.assert_close(gradient, reverse, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("score", PATHS)
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
@pytest.mark.parametrize("aspect", [1.0, 1e6, 1e12])
def test_both_paths_extreme_diagnostics(score: Score, scale: float, aspect: float) -> None:
    """Probe forward and backward separately; baseline failures remain visible."""
    first = box(1, 1 / aspect, 0).float()
    first[:4] *= scale
    first.requires_grad_()
    second = first.detach().clone().requires_grad_()
    value = score(first, second)
    gradients = torch.autograd.grad(value, (first, second))
    if score is stable_probiou:
        assert value.item() == 1
        assert all(bool(torch.isfinite(g).all()) for g in gradients)
    else:
        # These probes characterize upstream behavior; finite output is not
        # required where its expanded covariance arithmetic breaks down.
        assert value.shape == torch.Size([])
        assert all(g.shape == first.shape for g in gradients)


@pytest.mark.parametrize("score", PATHS)
def test_common_translation_and_rotation(score: Score) -> None:
    """Global rigid transforms preserve ordinary forward values and gradients."""
    first, second = box(x=0.2).requires_grad_(), box(1.4, 0.8, -0.4).requires_grad_()
    angle = 0.7
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    transformed = []
    for boxes in (first, second):
        center = rotation @ boxes[:2] + torch.tensor([10.0, -3.0], dtype=torch.float64)
        transformed.append(torch.cat((center, boxes[2:4], rotation @ boxes[4:6])))
    original = score(first, second)
    changed = score(*transformed)
    torch.testing.assert_close(original, changed, atol=2e-12, rtol=0)
    original_gradients = torch.autograd.grad(original, (first, second))
    changed_gradients = torch.autograd.grad(changed, (first, second))
    for gradient, expected in zip(changed_gradients, original_gradients, strict=True):
        torch.testing.assert_close(gradient, expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("aspect", [1.0, 1e6, 1e12, 1e24])
def test_thin_box_unsaturated_backward(aspect: float) -> None:
    """Thin-axis gradients agree with an analytic result before saturation."""
    first, second = box(1, 1 / aspect, 0), box(1, 1 / aspect, 0)
    second[1] = 0.1 / aspect
    first.requires_grad_()
    second.requires_grad_()
    value = stable_probiou(first, second)
    gradient = torch.autograd.grad(value, second)[0][1]
    distance = 1.5 * 0.1**2
    expected_score = 1 - math.sqrt(-math.expm1(-distance))
    expected_gradient = (
        -math.exp(-distance) * 3 * 0.1 * aspect / (2 * math.sqrt(-math.expm1(-distance)))
    )
    assert value.item() == pytest.approx(expected_score, abs=2e-14)
    assert gradient.item() == pytest.approx(expected_gradient, rel=2e-12)


@pytest.mark.parametrize("score", PATHS)
def test_direction_rescaling_and_input_ownership(score: Score) -> None:
    """Non-unit directions preserve geometry and neither path mutates inputs."""
    first, second = box(x=0.2), box(1.4, 0.8, -0.4)
    original1, original2 = first.clone(), second.clone()
    reference = score(first, second)
    torch.testing.assert_close(first, original1, atol=0, rtol=0)
    torch.testing.assert_close(second, original2, atol=0, rtol=0)
    first[4:6] *= 5
    second[4:6] *= 0.2
    torch.testing.assert_close(score(first, second), reference, atol=2e-14, rtol=0)


@pytest.mark.parametrize("score", PATHS)
def test_gradcheck_broadcast_reduction(score: Score) -> None:
    """Finite differences verify accumulated gradients for a broadcast target."""
    first = torch.stack((box(x=0.1), box(1.8, 0.6, -0.2))).requires_grad_()
    second = box(1.5, 0.8, 0.6).requires_grad_()
    assert torch.autograd.gradcheck(score, (first, second), atol=2e-6, rtol=2e-4)


def notebook_stress_probes() -> list[StressProbe]:
    """Measure independent forward/backward status on the notebook stress grid.

    Return counts-ready chart records; baseline failures are retained rather
    than repaired. This is diagnostic infrastructure, not the demonstrated
    Gaussian arithmetic.
    """
    records: list[StressProbe] = []
    for dtype in (torch.float32, torch.float64):
        for scale_exponent in (-30, -15, -6, 0, 6, 15, 30):
            for aspect_exponent in (0, 2, 4, 6, 8, 10, 12):
                scale = 10.0**scale_exponent
                aspect = 10.0**aspect_exponent
                for name, score in [
                    ("stable", stable_probiou),
                    ("Ultralytics", ultralytics_probiou),
                ]:
                    first = box(scale, scale / aspect, 0).to(dtype).requires_grad_()
                    second = first.detach().clone().requires_grad_()
                    value = score(first, second)
                    gradients = torch.autograd.grad(value, (first, second))
                    forward = bool(torch.isfinite(value))
                    backward = all(bool(torch.isfinite(g).all()) for g in gradients)
                    status = "finite both" if forward and backward else "backward nonfinite"
                    if not forward:
                        status = "forward nonfinite"
                    records.append(
                        {
                            "scale": scale_exponent,
                            "aspect": aspect_exponent,
                            "path": name,
                            "dtype": str(dtype).removeprefix("torch."),
                            "status": status,
                        }
                    )
    return records
