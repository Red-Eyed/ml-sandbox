"""Pure pairwise Phase 1 costs, with raw metrics and weighted contributions kept distinct."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, log

import numpy as np

from .geometry import Box, FloatArray, angle_distance, rotated_iou


class Metric(StrEnum):
    """The small set of interchangeable Phase 1 pairwise cost definitions."""

    L1 = "L1"
    IOU = "Rotated IoU"
    PROBIOU = "ProbIoU"
    CLASSIFICATION = "Classification"


class Transform(StrEnum):
    """Alternative lower-is-better conversions for bounded similarities."""

    LINEAR = "1 - S"
    LOG = "-log(max(S, epsilon))"


@dataclass(frozen=True)
class Prediction:
    """A foreground prediction whose confidence is validated independently of its box."""

    box: Box
    confidence: float

    def __post_init__(self) -> None:
        """Allow both endpoints; the explicit matcher epsilon handles log zero."""
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Prediction confidence must be finite and in [0, 1]")


@dataclass(frozen=True)
class CostTerm:
    """An enabled cost and its finite nonnegative weight; absent terms do no work."""

    metric: Metric
    weight: float

    def __post_init__(self) -> None:
        """Reject weights that could reverse or invalidate the cost direction."""
        if not isfinite(self.weight) or self.weight < 0:
            raise ValueError("Cost weights must be finite and nonnegative")


@dataclass(frozen=True)
class MatcherConfig:
    """Explicit normalization lengths and common similarity conversion.

    L1 sums |delta cx|/x_scale + |delta cy|/y_scale + |delta w|/x_scale
    + |delta h|/y_scale, optionally adding angle_weight * periodic angle/pi.
    Scales are fixed scene units, independent of zoom or box count. Epsilon is
    an explicit log floor, never a cap on L1 or Bhattacharyya distance.
    """

    terms: tuple[CostTerm, ...]
    transform: Transform = Transform.LINEAR
    epsilon: float = 1e-12
    x_scale: float = 100.0
    y_scale: float = 100.0
    angle_weight: float = 0.0

    def __post_init__(self) -> None:
        """Validate all numerical controls even for empty scenes."""
        if not isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("Epsilon must be finite and strictly between zero and one")
        if any(not isfinite(v) or v <= 0 for v in (self.x_scale, self.y_scale)):
            raise ValueError("L1 normalization lengths must be finite and positive")
        if not isfinite(self.angle_weight) or self.angle_weight < 0:
            raise ValueError("L1 angle weight must be finite and nonnegative")
        if len({term.metric for term in self.terms}) != len(self.terms):
            raise ValueError("Enable each metric at most once")


@dataclass(frozen=True)
class Component:
    """One metric matrix, its transformed cost, and its weighted contribution."""

    metric: Metric
    raw: FloatArray
    cost: FloatArray
    weighted: FloatArray


@dataclass(frozen=True)
class CostEvaluation:
    """Pairwise matrices with rows in prediction order and columns in target order."""

    total: FloatArray
    components: tuple[Component, ...]


def bhattacharyya(a: Box, b: Box) -> float:
    """Return Gaussian Bhattacharyya distance for covariance R diag(w²,h²) Rᵀ/12.

    Evaluate in a's local frame and common units, solving with Cholesky rather
    than inverting a covariance. The determinant of each box is known from its
    dimensions. Unrepresentable covariance or distances raise ValueError.
    """
    scale = max(a.width, a.height, b.width, b.height)
    da = np.array([a.width / scale, a.height / scale], dtype=np.float64)
    db = np.array([b.width / scale, b.height / scale], dtype=np.float64)
    relative = a.rotation.T @ b.rotation
    covariance_a = np.diag(da * da / 12)
    covariance_b = (relative * (db * db / 12)) @ relative.T
    mean = (covariance_a + covariance_b) / 2
    delta = ((b.center - a.center) / scale) @ a.rotation
    try:
        root = np.linalg.cholesky(mean)
        whitened = np.linalg.solve(root, delta)
    except np.linalg.LinAlgError as error:
        raise ValueError("Gaussian covariance is outside float64 resolution") from error
    logdet_mean = 2 * float(np.log(np.diag(root)).sum())
    logdet_a = 2 * (log(a.width) + log(a.height) - 2 * log(scale)) - 2 * log(12)
    logdet_b = 2 * (log(b.width) + log(b.height) - 2 * log(scale)) - 2 * log(12)
    distance = float(whitened @ whitened) / 8 + (logdet_mean - (logdet_a + logdet_b) / 2) / 2
    if not isfinite(distance) or distance < -1e-10:
        raise ValueError("Bhattacharyya distance is outside float64 resolution")
    return max(distance, 0.0)


def log_probiou(distance: float) -> float:
    """Return log(1 - Hellinger) without cancellation for well-separated boxes."""
    return -distance - float(np.log1p(np.sqrt(-np.expm1(-distance))))


def probiou(a: Box, b: Box) -> float:
    """Return Gaussian ProbIoU in [0,1]; extreme separation can underflow to zero."""
    return float(np.exp(log_probiou(bhattacharyya(a, b))))


def raw_metric(metric: Metric, prediction: Prediction, target: Box, config: MatcherConfig) -> float:
    """Evaluate one raw value, before similarity conversion or weighting."""
    box = prediction.box
    match metric:
        case Metric.L1:
            return (
                (abs(box.cx - target.cx) + abs(box.width - target.width)) / config.x_scale
                + (abs(box.cy - target.cy) + abs(box.height - target.height)) / config.y_scale
                + config.angle_weight * angle_distance(box.angle, target.angle)
            )
        case Metric.IOU:
            return rotated_iou(box, target)
        case Metric.PROBIOU:
            return probiou(box, target)
        case Metric.CLASSIFICATION:
            return prediction.confidence


def transform_cost(metric: Metric, raw: FloatArray, config: MatcherConfig) -> FloatArray:
    """Convert a raw metric to cost; classification always uses negative log p."""
    match metric:
        case Metric.CLASSIFICATION:
            return -np.log(np.maximum(raw, config.epsilon))
        case Metric.L1:
            return raw.copy()
        case Metric.IOU | Metric.PROBIOU:
            match config.transform:
                case Transform.LINEAR:
                    return 1 - raw
                case Transform.LOG:
                    return -np.log(np.maximum(raw, config.epsilon))


def evaluate_costs(
    predictions: Sequence[Prediction], targets: Sequence[Box], config: MatcherConfig
) -> CostEvaluation:
    """Evaluate enabled nonzero terms only, rejecting duplicates or nonfinite costs.

    All-zero configurations intentionally yield a tied zero matrix. No confidence
    threshold, dummy assignment, or implicit metric normalization is applied.
    """
    identifiers = [p.box.id for p in predictions] + [t.id for t in targets]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Scene identifiers must be unique across predictions and targets")
    shape = (len(predictions), len(targets))
    total = np.zeros(shape, dtype=np.float64)
    components: list[Component] = []
    for term in config.terms:
        if term.weight == 0:
            continue
        raw = np.array(
            [[raw_metric(term.metric, p, t, config) for t in targets] for p in predictions],
            dtype=np.float64,
        ).reshape(shape)
        cost = transform_cost(term.metric, raw, config)
        weighted = term.weight * cost
        components.append(Component(term.metric, raw, cost, weighted))
        total += weighted
    if not np.isfinite(total).all():
        raise ValueError("Matching costs must be finite; inspect geometry and weights")
    return CostEvaluation(total, tuple(components))
