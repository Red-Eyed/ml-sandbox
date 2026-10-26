"""Numerical backend for the interactive oriented-box Hungarian matcher laboratory.

Import ``MatcherLab`` from ``.widgets`` only when starting the interactive UI;
backend imports do not initialize Matplotlib, widgets, or a notebook backend.
"""

from .assignment import Assignment, solve_assignment
from .costs import CostTerm, MatcherConfig, Metric, Prediction, Transform, evaluate_costs
from .gaussian import (
    CovarianceConvention,
    Gaussian2D,
    NormalizedBox,
    box_to_gaussian,
    normalized_box_to_gaussian,
)
from .geometry import Box
from .presets import Scene, baseline

__all__ = [
    "Assignment",
    "Box",
    "CostTerm",
    "CovarianceConvention",
    "Gaussian2D",
    "MatcherConfig",
    "Metric",
    "NormalizedBox",
    "Prediction",
    "Scene",
    "Transform",
    "baseline",
    "box_to_gaussian",
    "evaluate_costs",
    "normalized_box_to_gaussian",
    "solve_assignment",
]
