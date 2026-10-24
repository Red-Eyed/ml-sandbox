"""Analytic, SciPy-reference, and matching checks for sampled Hausdorff geometry."""

from dataclasses import replace
from math import pi, sqrt

import numpy as np
import pytest
from scipy.spatial.distance import directed_hausdorff

from .assignment import solve_assignment
from .costs import CostTerm, MatcherConfig, Metric, Transform, evaluate_costs
from .geometry import Box, corners
from .hausdorff import boundary_points, hausdorff_distance
from .presets import scene_presets


@pytest.fixture
def box() -> Box:
    """Provide a thin rotated rectangle for sampling and invariance checks."""
    return Box("A", 12, -7, 30, 2, 0.4)


@pytest.mark.parametrize("num_points", [4, 8, 32])
def test_identity_and_equivalent_representations(box: Box, num_points: int) -> None:
    """Corner order, half turns, and width/height swaps do not change the rectangle."""
    equivalent = replace(box, id="B", width=box.height, height=box.width, angle=box.angle + pi / 2)
    assert hausdorff_distance(box, box, num_points=num_points) == 0
    assert hausdorff_distance(box, equivalent, num_points=num_points) == pytest.approx(0, abs=1e-12)
    assert hausdorff_distance(
        box, replace(box, angle=box.angle + pi), num_points=num_points
    ) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("num_points", [4, 8, 32])
def test_translated_box_has_analytic_distance(box: Box, num_points: int) -> None:
    """A translated copy has distance equal to its center displacement, without overlap requirements."""
    translated = replace(box, id="B", cx=box.cx + 3, cy=box.cy + 4)
    assert hausdorff_distance(box, translated, num_points=num_points) == pytest.approx(5)
    distant = replace(box, id="B", cx=box.cx + 3000, cy=box.cy + 4000)
    assert hausdorff_distance(box, distant, num_points=num_points) == pytest.approx(5000)


@pytest.mark.parametrize("num_points", [4, 8, 32])
def test_matches_scipy_in_both_directions(box: Box, num_points: int) -> None:
    """Symmetric point-set distance agrees with an independent SciPy solver."""
    other = Box("B", -2, 3, 5, 8, -0.7)
    a, b = boundary_points(box, num_points), boundary_points(other, num_points)
    expected = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
    assert hausdorff_distance(box, other, num_points=num_points) == pytest.approx(expected)
    assert hausdorff_distance(other, box, num_points=num_points) == pytest.approx(expected)


def test_boundary_samples_are_even_per_edge() -> None:
    """Eight points consist of corners and midpoints, without repeating endpoints."""
    box = Box("A", 0, 0, 4, 2)
    points = boundary_points(box, 8)
    np.testing.assert_allclose(points[::2], corners(box))
    np.testing.assert_allclose(points[1::2], (corners(box) + np.roll(corners(box), -1, axis=0)) / 2)
    assert len(np.unique(points, axis=0)) == 8
    assert hausdorff_distance(box, Box("B", 0, 0, 2, 2)) == pytest.approx(1)
    assert hausdorff_distance(Box("A", 0, 0, 4, 4), Box("B", 0, 0, 2, 2)) == pytest.approx(sqrt(2))


def test_rhino_normalizes_parameters_before_rotation(box: Box) -> None:
    """Unequal image scales match RHINO's convention, distinct from scaling final corners."""
    other = Box("B", 4, 6, 13, 3, -0.9)
    normalized_a = replace(
        box, cx=box.cx / 200, cy=box.cy / 100, width=box.width / 200, height=box.height / 100
    )
    normalized_b = replace(
        other,
        cx=other.cx / 200,
        cy=other.cy / 100,
        width=other.width / 200,
        height=other.height / 100,
    )
    expected = hausdorff_distance(normalized_a, normalized_b, num_points=32)
    assert hausdorff_distance(box, other, num_points=32, x_scale=200, y_scale=100) == pytest.approx(
        expected
    )


def test_rigid_motion_and_common_scale(box: Box) -> None:
    """A common rigid motion preserves distance, and common units scale it linearly."""
    other = Box("B", 4, 6, 13, 3, -0.9)
    rotation = Box("frame", 0, 0, 1, 1, 0.6).rotation
    transformed = []
    for original in (box, other):
        center = rotation @ original.center + [50, -90]
        transformed.append(
            replace(original, cx=float(center[0]), cy=float(center[1]), angle=original.angle + 0.6)
        )
    expected = hausdorff_distance(box, other, num_points=32)
    assert hausdorff_distance(*transformed, num_points=32) == pytest.approx(expected)
    assert hausdorff_distance(box, other, num_points=32, x_scale=10, y_scale=10) == pytest.approx(
        expected / 10
    )


@pytest.mark.parametrize("count", [0, 3, 7, 16, 33])
def test_invalid_sampling_count(box: Box, count: int) -> None:
    """Unsupported UI point counts fail clearly at both backend boundaries."""
    with pytest.raises(ValueError, match="point count"):
        boundary_points(box, count)
    with pytest.raises(ValueError, match="point count"):
        MatcherConfig((), hausdorff_points=count)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale(box: Box, scale: float) -> None:
    """Invalid normalization cannot silently produce a usable cost."""
    with pytest.raises(ValueError, match="scales"):
        hausdorff_distance(box, box, x_scale=scale)


def test_thin_boxes_remain_finite() -> None:
    """Strictly positive tiny dimensions work without covariance regularization."""
    box = Box("A", 0, 0, 100, 1e-9, 0.3)
    assert hausdorff_distance(box, replace(box, id="B", cy=2), num_points=32) == pytest.approx(2)


def test_geometry_only_assignment_rejects_oversized_competitor() -> None:
    """The screenshot-inspired pathology distinguishes Hausdorff from ProbIoU-only matching."""
    scene = scene_presets()["Oversized competing prediction"]
    config = MatcherConfig((CostTerm(Metric.HAUSDORFF, 1),))
    evaluation = evaluate_costs(scene.predictions, scene.targets, config)
    assert solve_assignment(evaluation.total).pairs == ((0, 0), (1, 1))
    logged = evaluate_costs(
        scene.predictions, scene.targets, replace(config, transform=Transform.LOG)
    )
    np.testing.assert_array_equal(logged.total, evaluation.total)
    gaussian = evaluate_costs(
        scene.predictions, scene.targets, MatcherConfig((CostTerm(Metric.PROBIOU, 1),))
    )
    assert solve_assignment(gaussian.total).pairs == ((1, 1), (2, 0))


def test_empty_hausdorff_cost_matrix() -> None:
    """The new metric preserves rectangular empty-scene solver semantics."""
    evaluation = evaluate_costs([], [], MatcherConfig((CostTerm(Metric.HAUSDORFF, 1),)))
    assert evaluation.total.shape == (0, 0)
    assert solve_assignment(evaluation.total).pairs == ()
