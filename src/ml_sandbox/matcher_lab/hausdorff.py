"""Sampled symmetric Hausdorff box cost from Lee et al., WACV 2025.

The mathematical definition and normalization follow RHINO's HausdorffCost:
https://github.com/SIAnalytics/RHINO/blob/main/mmrotate/models/task_modules/assigners/match_cost.py
This NumPy implementation is independent of its detector and training framework.
"""

from dataclasses import replace
from math import isfinite

import numpy as np

from .geometry import Box, FloatArray, corners

SUPPORTED_POINT_COUNTS = (4, 8, 32)


def boundary_points(box: Box, num_points: int) -> FloatArray:
    """Sample equal numbers of points per edge, including each corner once.

    Four points gives the corner set. Eight adds edge midpoints; thirty-two
    divides each edge into eight segments. Other point counts raise ValueError.
    These finite sets approximate the boundary, not the filled rectangle.
    """
    if num_points not in SUPPORTED_POINT_COUNTS:
        raise ValueError("Hausdorff point count must be 4, 8, or 32")
    vertices = corners(box)
    fractions = np.arange(num_points // 4, dtype=np.float64) / (num_points // 4)
    edges = np.roll(vertices, -1, axis=0) - vertices
    return (vertices[:, None, :] + fractions[None, :, None] * edges[:, None, :]).reshape(-1, 2)


def normalized_box(box: Box, x_scale: float, y_scale: float) -> Box:
    """Apply RHINO's parameter normalization before constructing rotated corners."""
    return replace(
        box,
        cx=box.cx / x_scale,
        cy=box.cy / y_scale,
        width=box.width / x_scale,
        height=box.height / y_scale,
    )


def hausdorff_distance(
    a: Box,
    b: Box,
    *,
    num_points: int = 4,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> float:
    """Return max(max_a min_b ||a-b||, max_b min_a ||a-b||) for sampled boundaries.

    Lower is better; the distance is unbounded and receives no similarity
    transformation. Positive finite scales divide cx/w and cy/h respectively,
    preserving angle as in the reference implementation. With unequal scales,
    this differs from scaling world-space corners after rotation. Scales of one
    return a distance in original scene units. Nonfinite results raise ValueError.
    """
    if any(not isfinite(value) or value <= 0 for value in (x_scale, y_scale)):
        raise ValueError("Hausdorff normalization scales must be finite and positive")
    points_a = boundary_points(normalized_box(a, x_scale, y_scale), num_points)
    points_b = boundary_points(normalized_box(b, x_scale, y_scale), num_points)
    delta = points_a[:, None, :] - points_b[None, :, :]
    distances = np.hypot(delta[..., 0], delta[..., 1])
    distance = float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))
    if not isfinite(distance):
        raise ValueError("Hausdorff distance is outside float64 resolution")
    return distance
