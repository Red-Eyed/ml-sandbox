"""Check cost direction, Gaussian reference values, and explicit numerical floors."""

from dataclasses import replace
from math import exp, log, pi, sqrt
from typing import TypedDict
from unittest.mock import patch

import numpy as np
import pytest

from .costs import (
    CostTerm,
    MatcherConfig,
    Metric,
    Prediction,
    Transform,
    bhattacharyya,
    evaluate_costs,
    log_probiou,
    probiou,
)
from .geometry import Box
from .presets import baseline, scene_presets


@pytest.mark.parametrize("angle", [0.0, 0.4, pi / 2])
def test_gaussian_reference(angle: float) -> None:
    """Equal isotropic covariances give D=3d²/(2w²), independent of angle."""
    a = Box("A", 0, 0, 4, 4, angle)
    b = Box("B", 2, 0, 4, 4, -angle)
    expected = 3 * 2**2 / (2 * 4**2)
    assert bhattacharyya(a, b) == pytest.approx(expected)
    assert probiou(a, b) == pytest.approx(1 - sqrt(1 - exp(-expected)))
    assert bhattacharyya(b, a) == pytest.approx(expected)
    assert probiou(a, a) == pytest.approx(1, abs=1e-7)


def test_gaussian_covariance_reference() -> None:
    """Compare anisotropic rotated boxes to an independent direct matrix formula."""
    a, b = Box("A", 2, 3, 14, 2, 0.7), Box("B", 5, 8, 9, 3, -0.2)
    covariance_a = a.rotation @ np.diag([a.width**2, a.height**2]) @ a.rotation.T / 12
    covariance_b = b.rotation @ np.diag([b.width**2, b.height**2]) @ b.rotation.T / 12
    mean = (covariance_a + covariance_b) / 2
    delta = a.center - b.center
    expected = (
        float(delta @ np.linalg.solve(mean, delta)) / 8
        + log(np.linalg.det(mean) / sqrt(np.linalg.det(covariance_a) * np.linalg.det(covariance_b)))
        / 2
    )
    assert bhattacharyya(a, b) == pytest.approx(expected)
    assert bhattacharyya(b, a) == pytest.approx(expected)
    shift = np.array([1e5, -2e5])
    assert bhattacharyya(
        replace(a, cx=a.cx + shift[0], cy=a.cy + shift[1]),
        replace(b, cx=b.cx + shift[0], cy=b.cy + shift[1]),
    ) == pytest.approx(expected)


def test_thin_and_distant_boxes() -> None:
    """Retain thin-box precision and avoid 1-sqrt cancellation before underflow."""
    a = Box("A", 0, 0, 100, 1e-6)
    assert probiou(a, a) == pytest.approx(1, abs=1e-6)
    assert 0 < probiou(a, replace(a, id="B", cx=500)) < 1e-15
    assert log_probiou(1e6) == pytest.approx(-1e6 - log(2))
    assert probiou(a, replace(a, id="B", cx=1e9)) == 0


def test_explicit_cost_transform_and_decomposition() -> None:
    """Raw probability, similarity, transformed cost, and weighted contribution remain separate."""
    prediction = Prediction(Box("P1", 0, 0, 4, 2), 0)
    target = Box("G1", 20, 0, 4, 2)
    config = MatcherConfig(
        (CostTerm(Metric.IOU, 2), CostTerm(Metric.CLASSIFICATION, 3)),
        transform=Transform.LOG,
        epsilon=1e-6,
    )
    evaluation = evaluate_costs([prediction], [target], config)
    assert evaluation.components[0].raw[0, 0] == 0
    assert evaluation.components[0].cost[0, 0] == pytest.approx(-log(1e-6))
    assert evaluation.total[0, 0] == pytest.approx(-5 * log(1e-6))
    np.testing.assert_allclose(evaluation.total, sum(c.weighted for c in evaluation.components))


def test_disabled_costs_do_no_work() -> None:
    """Neither missing nor zero-weight metrics call the mathematical evaluator."""
    scene = scene_presets()["Perfect match"]
    with patch("ml_sandbox.matcher_lab.costs.raw_metric", side_effect=AssertionError):
        evaluation = evaluate_costs(
            scene.predictions, scene.targets, MatcherConfig((CostTerm(Metric.PROBIOU, 0),))
        )
    assert evaluation.total[0, 0] == 0
    assert not evaluation.components


def test_l1_scaling_and_periodic_angle() -> None:
    """L1 uses explicit coordinate scales, not box dimensions or current viewport."""
    a = Box("P1", 0, 0, 4, 2, 0)
    b = Box("G1", 10, 20, 6, 3, pi)
    config = MatcherConfig((CostTerm(Metric.L1, 1),), x_scale=10, y_scale=20, angle_weight=2)
    assert evaluate_costs([Prediction(a, 0.5)], [b], config).total[0, 0] == pytest.approx(2.25)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_reject_invalid_confidence(confidence: float) -> None:
    """Invalid probabilities cannot enter a cost matrix."""
    with pytest.raises(ValueError):
        Prediction(Box("P1", 0, 0, 1, 1), confidence)


def test_reject_duplicate_scene_identities() -> None:
    """A stable identity must resolve to exactly one box across the whole scene."""
    box = Box("same", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="unique"):
        evaluate_costs([Prediction(box, 0.5)], [box], baseline())


class ConfigChanges(TypedDict, total=False):
    """Describe the numeric configuration fields varied by validation tests."""

    epsilon: float
    x_scale: float
    y_scale: float
    angle_weight: float


@pytest.mark.parametrize(
    "changes",
    [
        {"epsilon": 0},
        {"epsilon": 1},
        {"x_scale": 0},
        {"angle_weight": -1},
        {"y_scale": float("inf")},
    ],
)
def test_reject_invalid_config(changes: ConfigChanges) -> None:
    """Bad numerical controls are rejected even before a nonempty scene exists."""
    with pytest.raises(ValueError):
        replace(baseline(), **changes)


@pytest.mark.parametrize("angle,scale", [(0.6, 1.0), (-0.9, 100.0), (1.7, 1e-4)])
def test_gaussian_rigid_rotation_and_scaling(angle: float, scale: float) -> None:
    """Gaussian distance is invariant to a shared rotation, translation, and change of units."""
    a, b = Box("A", 2, 3, 14, 2, 0.7), Box("B", 5, 8, 9, 3, -0.2)
    rotation = Box("frame", 0, 0, 1, 1, angle).rotation
    transformed = []
    for box in (a, b):
        center = (rotation @ box.center) * scale + np.array([7, -11]) * scale
        transformed.append(
            replace(
                box,
                cx=float(center[0]),
                cy=float(center[1]),
                width=box.width * scale,
                height=box.height * scale,
                angle=box.angle + angle,
            )
        )
    assert bhattacharyya(*transformed) == pytest.approx(bhattacharyya(a, b))
    assert 0 <= probiou(*transformed) <= 1


def test_gaussian_conversion_shared_once_per_box() -> None:
    """Enabling several Gaussian methods reuses one converted distribution per scene box."""
    from .gaussian import box_to_gaussian

    scene = scene_presets()["Neighboring text lines"]
    config = MatcherConfig(
        tuple(
            CostTerm(metric, 1)
            for metric in (
                Metric.KL_FORWARD,
                Metric.KL_REVERSE,
                Metric.SYMMETRIC_KL,
                Metric.PROBIOU,
            )
        )
    )
    with patch("ml_sandbox.matcher_lab.costs.box_to_gaussian", wraps=box_to_gaussian) as convert:
        evaluation = evaluate_costs(scene.predictions, scene.targets, config)
    assert convert.call_count == len(scene.predictions) + len(scene.targets)
    assert len(evaluation.components) == 4
    forward, reverse, symmetric, _ = evaluation.components
    np.testing.assert_allclose(symmetric.raw, (forward.raw + reverse.raw) / 2)


def test_no_gaussian_conversion_for_disabled_terms() -> None:
    """Non-Gaussian matching does not instantiate even zero-weight Gaussian components."""
    scene = scene_presets()["Perfect match"]
    config = MatcherConfig((CostTerm(Metric.IOU, 1), CostTerm(Metric.SYMMETRIC_KL, 0)))
    with patch("ml_sandbox.matcher_lab.costs.box_to_gaussian", side_effect=AssertionError):
        assert evaluate_costs(scene.predictions, scene.targets, config).total[0, 0] == 0


@pytest.mark.parametrize(
    "metric",
    [
        Metric.KL_FORWARD,
        Metric.KL_REVERSE,
        Metric.SYMMETRIC_KL,
        Metric.WASSERSTEIN,
        Metric.BHATTACHARYYA,
        Metric.HELLINGER,
    ],
)
def test_gaussian_distance_costs_are_not_similarity_transformed(metric: Metric) -> None:
    """Distances keep their direction and native scale for either similarity control setting."""
    scene = scene_presets()["Neighboring text lines"]
    config = MatcherConfig((CostTerm(metric, 1),))
    base = evaluate_costs(scene.predictions, scene.targets, config)
    logged = evaluate_costs(
        scene.predictions, scene.targets, replace(config, transform=Transform.LOG)
    )
    np.testing.assert_array_equal(base.total, logged.total)
    np.testing.assert_array_equal(base.total, base.components[0].raw)
    assert np.isfinite(base.total).all()
    assert (base.total >= 0).all()
