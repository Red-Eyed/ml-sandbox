"""Independent geometry-only assignments for a shared editable scene."""

from dataclasses import dataclass, replace

from .assignment import Assignment, solve_assignment
from .costs import Component, CostTerm, MatcherConfig, Metric, evaluate_costs
from .presets import Scene

GEOMETRIC_METRICS = tuple(metric for metric in Metric if metric != Metric.CLASSIFICATION)


@dataclass(frozen=True)
class MetricMatch:
    """One unweighted metric's costs and its own rectangular assignment."""

    metric: Metric
    component: Component
    assignment: Assignment


@dataclass(frozen=True)
class MetricFailure:
    """An unavailable metric with the numerical failure retained for display."""

    metric: Metric
    reason: str


def compare_metrics(scene: Scene, config: MatcherConfig) -> tuple[MetricMatch | MetricFailure, ...]:
    """Evaluate every geometric metric using shared settings, ignoring matcher weights.

    The normal path converts each box to a Gaussian once. If joint evaluation
    fails, retry separately so a failed method cannot hide valid comparisons.
    Totals are never used to rank metrics with different units.
    """
    comparison_config = replace(
        config, terms=tuple(CostTerm(metric, 1) for metric in GEOMETRIC_METRICS)
    )
    try:
        evaluation = evaluate_costs(scene.predictions, scene.targets, comparison_config)
        return tuple(
            MetricMatch(component.metric, component, solve_assignment(component.cost))
            for component in evaluation.components
        )
    except ValueError:
        return tuple(
            _compare_metric(scene, comparison_config, metric) for metric in GEOMETRIC_METRICS
        )


def _compare_metric(
    scene: Scene, config: MatcherConfig, metric: Metric
) -> MetricMatch | MetricFailure:
    """Retain one method's failure without discarding other methods' results."""
    try:
        evaluation = evaluate_costs(
            scene.predictions, scene.targets, replace(config, terms=(CostTerm(metric, 1),))
        )
        component = evaluation.components[0]
        return MetricMatch(metric, component, solve_assignment(component.cost))
    except ValueError as error:
        return MetricFailure(metric, str(error))
