"""Analytic and invariant checks for oriented polygon intersection and editing."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from .geometry import Box, contains, corners, resize_from_corner, rotated_iou


@pytest.fixture
def box() -> Box:
    """Provide a non-square rotated rectangle away from the origin."""
    return Box("G1", 11, -4, 12, 3, 0.4)


@pytest.mark.parametrize("width,height", [(0, 1), (-1, 2), (1, float("nan"))])
def test_invalid_dimensions(width: float, height: float) -> None:
    """Reject degenerate and nonfinite dimensions before numerical operations."""
    with pytest.raises(ValueError):
        Box("G1", 0, 0, width, height)


@pytest.mark.parametrize("width,height,angle", [(12, 3, 0.4), (100, 1e-6, 0.0), (1e-8, 1e-8, 1.2)])
def test_identical_boxes(width: float, height: float, angle: float) -> None:
    """Preserve exact identity at small scales and large aspect ratios."""
    a = Box("A", 0, 0, width, height, angle)
    assert rotated_iou(a, replace(a, id="B")) == pytest.approx(1, abs=1e-8)


def test_analytic_intersections() -> None:
    """Match analytic axis-aligned overlap and perpendicular thin-rectangle overlap."""
    a = Box("A", 0, 0, 4, 2)
    assert rotated_iou(a, Box("B", 2, 0, 4, 2)) == pytest.approx(1 / 3)
    assert rotated_iou(a, Box("B", 4, 0, 4, 2)) == 0
    assert rotated_iou(a, Box("B", 40, 0, 4, 2)) == 0
    assert rotated_iou(a, replace(a, id="B", angle=pi / 2)) == pytest.approx(1 / 3)


def test_rigid_motion_and_symmetry(box: Box) -> None:
    """A common rotation and translation preserve IoU and operand order does not matter."""
    b = Box("B", 12, -3, 9, 2, -0.2)
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    angle = np.arctan2(0.8, 0.6)
    transformed = []
    for original in (box, b):
        center = rotation @ original.center + np.array([1000, -300])
        transformed.append(
            replace(
                original, cx=float(center[0]), cy=float(center[1]), angle=original.angle + angle
            )
        )
    expected = rotated_iou(box, b)
    assert rotated_iou(b, box) == pytest.approx(expected)
    assert rotated_iou(*transformed) == pytest.approx(expected)
    assert 0 <= expected <= 1


def test_local_resize_fixes_opposite_corner(box: Box) -> None:
    """A rotated corner resize changes local side lengths without moving its anchor."""
    anchor = corners(box)[0]
    point = anchor + np.array([18, 5]) @ box.rotation.T
    resized = resize_from_corner(box, 2, point, 0.01)
    np.testing.assert_allclose(corners(resized)[0], anchor)
    assert resized.width == pytest.approx(18)
    assert resized.height == pytest.approx(5)
    assert resized.angle == box.angle
    crossed = resize_from_corner(box, 2, anchor - box.rotation[:, 0], 0.01)
    assert crossed.width == pytest.approx(0.01)
    assert crossed.height == pytest.approx(0.01)


def test_oriented_hit_testing(box: Box) -> None:
    """Body selection respects the rectangular region rather than its enclosing AABB."""
    assert contains(box, box.center)
    assert contains(box, box.center + np.array([1.0, 0.5]) @ box.rotation.T)
    assert not contains(box, box.center + np.array([0.0, 2.0]) @ box.rotation.T)
