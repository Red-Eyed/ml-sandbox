"""Independent references and invariants for conversion and Gaussian distances."""

from dataclasses import replace
from math import log, pi, sqrt

import numpy as np
import pytest
import torch
from scipy.linalg import sqrtm
from torch.distributions import MultivariateNormal, kl_divergence

from .gaussian import (
    CovarianceConvention,
    Gaussian2D,
    NormalizedBox,
    box_to_gaussian,
    normalized_box_to_gaussian,
)
from .gaussian_kl import gaussian_kl, symmetric_kl
from .gaussian_overlap import gaussian_bhattacharyya, gaussian_hellinger, gaussian_probiou
from .gaussian_wasserstein import gaussian_wasserstein
from .geometry import Box


@pytest.fixture
def boxes() -> tuple[Box, Box]:
    """Provide distinct nonsingular rotated boxes for independent numerical references."""
    return Box("P", 2, 3, 14, 2, 0.7), Box("G", 5, 8, 9, 3, -0.2)


@pytest.mark.parametrize(
    "convention,divisor",
    [(CovarianceConvention.UNIFORM, 12), (CovarianceConvention.HALF_EXTENT, 4)],
)
def test_converter_matches_covariance_formula(
    boxes: tuple[Box, Box], convention: CovarianceConvention, divisor: int
) -> None:
    """The factor, mean, and analytic log determinant match the rotated-box definition."""
    box = boxes[0]
    gaussian = box_to_gaussian(box, convention=convention)
    reference = box.rotation @ np.diag([box.width**2, box.height**2]) @ box.rotation.T / divisor
    np.testing.assert_allclose(gaussian.mean, box.center)
    np.testing.assert_allclose(gaussian.covariance, reference)
    assert gaussian.log_determinant == pytest.approx(np.linalg.slogdet(reference)[1])
    factor = gaussian.factor
    factor[:] = 0
    np.testing.assert_allclose(gaussian.covariance, reference)


def test_equivalent_boxes_have_same_gaussian(boxes: tuple[Box, Box]) -> None:
    """Half turns and width/height swaps preserve the represented distribution."""
    a = boxes[0]
    gaussian = box_to_gaussian(a)
    for b in (
        replace(a, angle=a.angle + pi),
        replace(a, width=a.height, height=a.width, angle=a.angle + pi / 2),
    ):
        np.testing.assert_allclose(box_to_gaussian(b).covariance, gaussian.covariance)
    square = Box("square", 0, 0, 4, 4, 0.4)
    np.testing.assert_array_equal(
        box_to_gaussian(square).covariance, box_to_gaussian(replace(square, angle=1.2)).covariance
    )


def test_normalized_adapter_uses_image_geometry() -> None:
    """Shifted nonunit angle vectors are normalized after denormalizing nonsquare image dimensions."""
    encoded = NormalizedBox("P", 0.25, 0.5, 0.1, 0.2, 0.7, 0.7)
    converted = normalized_box_to_gaussian(encoded, image_width=800, image_height=400)
    expected = box_to_gaussian(Box("P", 200, 200, 80, 80, pi / 4))
    np.testing.assert_allclose(converted.mean, expected.mean)
    np.testing.assert_allclose(converted.covariance, expected.covariance)
    angled = replace(encoded, height=0.1)
    actual = normalized_box_to_gaussian(angled, image_width=800, image_height=400)
    reference = box_to_gaussian(Box("P", 200, 200, 80, 40, pi / 4))
    np.testing.assert_allclose(actual.covariance, reference.covariance)


@pytest.mark.parametrize("values", [(0.5, 0.5), (float("nan"), 0.5), (2.0, 0.5)])
def test_reject_invalid_orientation(values: tuple[float, float]) -> None:
    """Undefined and out-of-range detector encodings cannot enter Gaussian calculations."""
    with pytest.raises(ValueError):
        box = NormalizedBox("P", 0, 0, 0.1, 0.1, *values)
        normalized_box_to_gaussian(box, image_width=800, image_height=400)


@pytest.mark.parametrize("image_width", [0, -1, float("inf"), float("nan")])
def test_reject_invalid_image_dimensions(image_width: float) -> None:
    """Image-normalized inputs require explicit positive finite physical scales."""
    with pytest.raises(ValueError):
        normalized_box_to_gaussian(
            NormalizedBox("P", 0, 0, 0.1, 0.1, 1, 0.5), image_width=image_width, image_height=400
        )


def test_kl_matches_pytorch(boxes: tuple[Box, Box]) -> None:
    """Both KL directions agree with torch.distributions on independent dense covariances."""
    a, b = (box_to_gaussian(box) for box in boxes)
    reference_a = MultivariateNormal(
        torch.tensor(a.mean), covariance_matrix=torch.tensor(a.covariance)
    )
    reference_b = MultivariateNormal(
        torch.tensor(b.mean), covariance_matrix=torch.tensor(b.covariance)
    )
    forward = float(kl_divergence(reference_a, reference_b))
    reverse = float(kl_divergence(reference_b, reference_a))
    assert gaussian_kl(a, b) == pytest.approx(forward)
    assert gaussian_kl(b, a) == pytest.approx(reverse)
    assert symmetric_kl(a, b) == pytest.approx((forward + reverse) / 2)
    assert forward != pytest.approx(reverse)
    assert symmetric_kl(b, a) == pytest.approx(symmetric_kl(a, b))


def test_wasserstein_matches_dense_reference(boxes: tuple[Box, Box]) -> None:
    """Factor alignment agrees with the standard square-root covariance expression."""
    a, b = (box_to_gaussian(box) for box in boxes)
    root_a = sqrtm(a.covariance)
    covariance_term = np.trace(
        a.covariance + b.covariance - 2 * sqrtm(root_a @ b.covariance @ root_a)
    )
    expected = sqrt(float(np.sum((np.array(a.mean) - b.mean) ** 2) + covariance_term))
    assert gaussian_wasserstein(a, b) == pytest.approx(expected)
    assert gaussian_wasserstein(b, a) == pytest.approx(expected)


def test_analytic_translation_and_variance() -> None:
    """Axis-aligned analytic cases expose KL direction, covariance scale, and unsquared W2."""
    a = box_to_gaussian(Box("A", 0, 0, 4, 4))
    b = box_to_gaussian(Box("B", 2, 0, 4, 4))
    assert gaussian_kl(a, b) == pytest.approx(1.5)
    assert symmetric_kl(a, b) == pytest.approx(1.5)
    assert gaussian_wasserstein(a, b) == pytest.approx(2)
    assert gaussian_bhattacharyya(a, b) == pytest.approx(0.375)
    alternative = box_to_gaussian(Box("B", 0, 0, 8, 8))
    assert gaussian_kl(a, alternative) == pytest.approx(0.25 - 1 + log(4))
    assert gaussian_wasserstein(a, alternative) == pytest.approx(sqrt(8 / 3))
    half_a = box_to_gaussian(Box("A", 0, 0, 4, 4), convention=CovarianceConvention.HALF_EXTENT)
    half_b = box_to_gaussian(Box("B", 2, 0, 4, 4), convention=CovarianceConvention.HALF_EXTENT)
    assert gaussian_kl(half_a, half_b) == pytest.approx(0.5)


def test_identity_and_thin_rotated_boxes() -> None:
    """Distances use factor precision even when dense covariance loses its small eigenvalue."""
    a = box_to_gaussian(Box("A", 0, 0, 100, 1e-8, 0.7))
    assert gaussian_kl(a, a) == 0
    assert symmetric_kl(a, a) == 0
    assert gaussian_wasserstein(a, a) == 0
    assert gaussian_bhattacharyya(a, a) == pytest.approx(0, abs=1e-12)
    assert gaussian_hellinger(a, a) == pytest.approx(0, abs=1e-6)
    b = replace(a, id="B", mean=(1, 0))
    assert np.isfinite(gaussian_kl(a, b))
    assert gaussian_wasserstein(a, b) == pytest.approx(1)


@pytest.mark.parametrize("scale", [1e-3, 5.0])
def test_distances_under_rigid_motion_and_units(boxes: tuple[Box, Box], scale: float) -> None:
    """KL/Bhattacharyya are unit invariant; W2 scales with common coordinate units."""
    a, b = (box_to_gaussian(box) for box in boxes)
    rotation = Box("frame", 0, 0, 1, 1, 0.5).rotation
    transformed = []
    for original in boxes:
        center = rotation @ original.center * scale + np.array([7, -11]) * scale
        transformed.append(
            box_to_gaussian(
                replace(
                    original,
                    cx=float(center[0]),
                    cy=float(center[1]),
                    width=original.width * scale,
                    height=original.height * scale,
                    angle=original.angle + 0.5,
                )
            )
        )
    c, d = transformed
    assert gaussian_kl(c, d) == pytest.approx(gaussian_kl(a, b))
    assert gaussian_bhattacharyya(c, d) == pytest.approx(gaussian_bhattacharyya(a, b))
    assert gaussian_wasserstein(c, d) == pytest.approx(scale * gaussian_wasserstein(a, b))
    assert gaussian_hellinger(c, d) == pytest.approx(gaussian_hellinger(a, b))
    assert 0 <= gaussian_probiou(c, d) <= 1


@pytest.mark.parametrize("std", [0, -1, 1e-200, 1e200])
def test_reject_unrepresentable_gaussian(std: float) -> None:
    """Invalid covariance factors are rejected without a silent numerical floor."""
    with pytest.raises(ValueError):
        Gaussian2D("G", (0, 0), (std, std), 0)
