"""Verify independent assignments and shared settings in metric comparison."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from .comparison import GEOMETRIC_METRICS, MetricFailure, MetricMatch, compare_metrics
from .costs import CostEvaluation, CostTerm, MatcherConfig, Metric, Prediction, evaluate_costs
from .geometry import Box
from .presets import Scene, baseline, scene_presets


def test_each_panel_solves_its_own_cost() -> None:
    """The oversized competitor exposes genuine disagreements between method optima."""
    scene = scene_presets()["Oversized competing prediction"]
    results = compare_metrics(scene, baseline())
    assert tuple(result.metric for result in results) == GEOMETRIC_METRICS
    for result in results:
        assert isinstance(result, MetricMatch)
        reference = evaluate_costs(
            scene.predictions, scene.targets, MatcherConfig((CostTerm(result.metric, 1),))
        )
        np.testing.assert_array_equal(result.component.cost, reference.total)
        assert result.assignment.total == pytest.approx(
            sum(reference.total[i, j] for i, j in result.assignment.pairs)
        )
    pairs = {
        result.metric: result.assignment.pairs
        for result in results
        if isinstance(result, MetricMatch)
    }
    assert pairs[Metric.HAUSDORFF] == ((0, 0), (1, 1))
    assert pairs[Metric.PROBIOU] == ((1, 1), (2, 0))


def test_comparison_ignores_weights_and_confidence() -> None:
    """Geometry-only panels remain independent of the separately configured matcher."""
    scene = scene_presets()["Oversized competing prediction"]
    edited = replace(
        scene, predictions=tuple(replace(p, confidence=0.01) for p in scene.predictions)
    )
    before = compare_metrics(scene, baseline())
    after = compare_metrics(edited, MatcherConfig(()))
    for first, second in zip(before, after, strict=True):
        assert isinstance(first, MetricMatch) and isinstance(second, MetricMatch)
        np.testing.assert_array_equal(first.component.cost, second.component.cost)
        assert first.assignment == second.assignment


def test_metric_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed Gaussian method must not erase the polygon-based methods."""
    original = evaluate_costs

    def reject_kl(
        predictions: Sequence[Prediction], targets: Sequence[Box], config: MatcherConfig
    ) -> CostEvaluation:
        """Fail one enabled metric while retaining the real evaluator otherwise."""
        if any(term.metric == Metric.KL_FORWARD for term in config.terms):
            raise ValueError("KL unavailable")
        return original(predictions, targets, config)

    monkeypatch.setattr("ml_sandbox.matcher_lab.comparison.evaluate_costs", reject_kl)
    results = compare_metrics(scene_presets()["Rotated crossing"], baseline())
    failures = [result for result in results if isinstance(result, MetricFailure)]
    assert failures == [MetricFailure(Metric.KL_FORWARD, "KL unavailable")]
    assert len([result for result in results if isinstance(result, MetricMatch)]) == 9


def test_empty_comparison() -> None:
    """Every method reports the correct empty rectangular assignment."""
    results = compare_metrics(Scene((), ()), baseline())
    assert len(results) == 10
    for result in results:
        assert isinstance(result, MetricMatch)
        assert result.component.cost.shape == (0, 0)
        assert result.assignment.pairs == ()
