"""Numerical backend for the interactive oriented-box Hungarian matcher laboratory.

Import ``MatcherLab`` from ``.widgets`` only when starting the interactive UI;
backend imports do not initialize Matplotlib, widgets, or a notebook backend.
"""

from .assignment import Assignment, solve_assignment
from .costs import CostTerm, MatcherConfig, Metric, Prediction, Transform, evaluate_costs
from .geometry import Box
from .presets import Scene, baseline

__all__ = [
    "Assignment",
    "Box",
    "CostTerm",
    "MatcherConfig",
    "Metric",
    "Prediction",
    "Scene",
    "Transform",
    "baseline",
    "evaluate_costs",
    "solve_assignment",
]
