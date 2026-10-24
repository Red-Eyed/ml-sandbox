"""Rectangular solver semantics and a noninteractive end-to-end backend smoke test."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from .assignment import solve_assignment
from .costs import CostTerm, MatcherConfig, Metric, evaluate_costs
from .presets import baseline, scene_presets


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (0, 2), (2, 0), (0, 0)])
def test_rectangular_and_empty(shape: tuple[int, int]) -> None:
    """Always produce min(P,G) disjoint pairs, explicitly reporting each unmatched index."""
    result = solve_assignment(np.zeros(shape))
    assert len(result.pairs) == min(shape)
    rows, columns = [i for i, _ in result.pairs], [j for _, j in result.pairs]
    assert len(set(rows)) == len(rows)
    assert len(set(columns)) == len(columns)
    assert sorted([*rows, *result.unmatched_predictions]) == list(range(shape[0]))
    assert sorted([*columns, *result.unmatched_targets]) == list(range(shape[1]))


def test_global_optimum_against_enumeration() -> None:
    """Choose a global one-to-one optimum even when independent row minima conflict."""
    cost = np.array([[4.0, 1, 3], [2, 0, 5], [3, 2, 2]])
    result = solve_assignment(cost)
    expected = min(sum(cost[i, j] for i, j in enumerate(order)) for order in permutations(range(3)))
    assert result.pairs == ((0, 1), (1, 0), (2, 2))
    assert result.total == expected == 5


@pytest.mark.parametrize("cost", [np.array([1.0]), np.array([[np.inf]]), np.array([[np.nan]])])
def test_reject_invalid_matrix(cost: np.ndarray) -> None:
    """Never pass malformed or nonfinite costs to the solver."""
    with pytest.raises(ValueError, match="finite two-dimensional"):
        solve_assignment(cost)


def test_confidence_selects_rows_but_not_square_permutation() -> None:
    """Single-class classification can discard a row, but adds a constant to a square optimum."""
    scene = scene_presets()["Neighboring text lines"]
    geometry = MatcherConfig((CostTerm(Metric.L1, 5),))
    combined = MatcherConfig((CostTerm(Metric.L1, 5), CostTerm(Metric.CLASSIFICATION, 1)))
    predictions = scene.predictions[:2]
    assert solve_assignment(evaluate_costs(predictions, scene.targets, geometry).total).pairs == (
        solve_assignment(evaluate_costs(predictions, scene.targets, combined).total).pairs
    )
    duplicate = replace(
        scene.predictions[0], box=replace(scene.predictions[0].box, id="P4"), confidence=0.99
    )
    config = MatcherConfig((CostTerm(Metric.CLASSIFICATION, 1),))
    evaluation = evaluate_costs((scene.predictions[0], duplicate), scene.targets[:1], config)
    assert solve_assignment(evaluation.total).pairs == ((1, 0),)


def test_zero_overlap_still_matches() -> None:
    """Zero geometric overlap is a cost, never an implicit unmatched constraint."""
    scene = scene_presets()["Separated boxes"]
    evaluation = evaluate_costs(
        scene.predictions, scene.targets, MatcherConfig((CostTerm(Metric.IOU, 1),))
    )
    assert evaluation.total[0, 0] == 1
    assert solve_assignment(evaluation.total).pairs == ((0, 0),)


def test_backend_smoke_all_presets() -> None:
    """Every starter scene has finite costs, valid indices, and the expected cardinality."""
    for scene in scene_presets().values():
        evaluation = evaluate_costs(scene.predictions, scene.targets, baseline())
        assert np.isfinite(evaluation.total).all()
        result = solve_assignment(evaluation.total)
        assert len(result.pairs) == min(len(scene.predictions), len(scene.targets))
