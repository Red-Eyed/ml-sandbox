"""Explicit rectangular matching semantics independent of geometry and presentation."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import FloatArray


@dataclass(frozen=True)
class Assignment:
    """Zero-based pairs and unmatched indices in original matrix order."""

    pairs: tuple[tuple[int, int], ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_targets: tuple[int, ...]
    total: float


def solve_assignment(cost: FloatArray) -> Assignment:
    """Assign min(rows, columns) pairs, including zero-overlap or low-confidence pairs.

    Reject nonfinite or non-matrix input even if one dimension is empty. Costs
    are not thresholded; SciPy resolves tied optima without a stability guarantee.
    """
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Assignment requires a finite two-dimensional cost matrix")
    rows, columns = linear_sum_assignment(cost)
    pairs = tuple((int(i), int(j)) for i, j in zip(rows, columns, strict=True))
    total = float(cost[rows, columns].sum())
    if not isfinite(total):
        raise ValueError("Assignment total exceeds float64 range")
    used_rows, used_columns = set(rows), set(columns)
    return Assignment(
        pairs,
        tuple(i for i in range(cost.shape[0]) if i not in used_rows),
        tuple(j for j in range(cost.shape[1]) if j not in used_columns),
        total,
    )
