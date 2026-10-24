"""Small reproducible scenes and explicit Phase 1 matcher configurations."""

from dataclasses import dataclass
from math import radians

from .costs import CostTerm, MatcherConfig, Metric, Prediction
from .geometry import Box


@dataclass(frozen=True)
class Scene:
    """Shared immutable targets and predictions, with stable globally unique identifiers."""

    targets: tuple[Box, ...]
    predictions: tuple[Prediction, ...]


def baseline() -> MatcherConfig:
    """Return the Phase 1 baseline: classification 1, normalized L1 5, rotated IoU 2.

    This is an explicit OBB prototype baseline, not DETR's class + L1 + GIoU.
    GIoU and multiclass classification are outside Phase 1.
    """
    return MatcherConfig(
        (CostTerm(Metric.CLASSIFICATION, 1), CostTerm(Metric.L1, 5), CostTerm(Metric.IOU, 2))
    )


def matcher_presets() -> dict[str, MatcherConfig]:
    """Return independent named configurations without hidden normalization."""
    return {
        "Baseline: class + L1 + rotated IoU": baseline(),
        "Class + ProbIoU": MatcherConfig(
            (CostTerm(Metric.CLASSIFICATION, 1), CostTerm(Metric.PROBIOU, 2))
        ),
        "Geometry only: ProbIoU": MatcherConfig((CostTerm(Metric.PROBIOU, 1),)),
        "Geometry only: Hausdorff": MatcherConfig((CostTerm(Metric.HAUSDORFF, 1),)),
        "Geometry only: rotated IoU": MatcherConfig((CostTerm(Metric.IOU, 1),)),
    }


def scene_presets() -> dict[str, Scene]:
    """Return starter pathologies; every box remains independently editable."""
    a = Box("G1", 35, 45, 42, 7, radians(12))
    b = Box("G2", 58, 61, 38, 7, radians(12))
    predictions = (
        Prediction(Box("P1", 38, 47, 40, 8, radians(16)), 0.45),
        Prediction(Box("P2", 56, 57, 38, 8, radians(5)), 0.92),
        Prediction(Box("P3", 70, 25, 30, 10, radians(-25)), 0.75),
    )
    return {
        "Neighboring text lines": Scene((a, b), predictions),
        "Perfect match": Scene((a,), (Prediction(Box("P1", 35, 45, 42, 7, radians(12)), 0.9),)),
        "Separated boxes": Scene(
            (Box("G1", 25, 50, 20, 8),), (Prediction(Box("P1", 75, 50, 20, 8), 0.9),)
        ),
        "Confidence versus localization": Scene(
            (a,),
            (
                Prediction(Box("P1", 35, 45, 42, 7, radians(12)), 0.08),
                Prediction(Box("P2", 50, 55, 42, 7, radians(12)), 0.98),
            ),
        ),
        "More targets than predictions": Scene((a, b), predictions[:1]),
        "Rotated crossing": Scene(
            (Box("G1", 50, 50, 60, 6, radians(40)),),
            (Prediction(Box("P1", 50, 50, 60, 6, radians(-40)), 0.8),),
        ),
        "Oversized competing prediction": Scene(
            (Box("G1", 40, 52, 42, 7, radians(12)), Box("G2", 62, 68, 38, 7, radians(12))),
            (
                Prediction(Box("P1", 32, 68, 40, 7, radians(16)), 1.0),
                Prediction(Box("P2", 60, 64, 38, 8, radians(5)), 1.0),
                Prediction(Box("P3", 70, 50, 20, 103, radians(-25)), 1.0),
            ),
        ),
        "Empty scene": Scene((), ()),
    }
