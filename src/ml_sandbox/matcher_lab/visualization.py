"""Shared scene and matrix rendering for the live canvas and static SVG preview."""

from dataclasses import dataclass
from html import escape
from io import StringIO

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Annotation

from .assignment import Assignment, solve_assignment
from .comparison import MetricFailure, MetricMatch, compare_metrics
from .costs import (
    GAUSSIAN_METRICS,
    CostEvaluation,
    MatcherConfig,
    Metric,
    evaluate_costs,
    gaussian_metric,
    raw_metric,
)
from .gaussian import CovarianceConvention, Gaussian2D, box_to_gaussian
from .geometry import Box, FloatArray, corners
from .presets import Scene

GT_COLOR = "#15803d"
PRED_COLOR = "#2563eb"
MATCH_COLOR = "#a21caf"


@dataclass(frozen=True)
class Absent:
    """A reason-carrying absence for selection and pointer state."""

    reason: str


NO_SELECTION = Absent("No selection")


def find_box(scene: Scene, identifier: str) -> Box:
    """Find a stable scene identity, raising ValueError for a stale selection."""
    for box in (*scene.targets, *(p.box for p in scene.predictions)):
        if box.id == identifier:
            return box
    raise ValueError(f"Unknown box: {identifier}")


def handle_points(axes: Axes, box: Box) -> FloatArray:
    """Return four resize handles and a rotation handle 24 screen pixels above the box."""
    top = box.center + box.rotation[:, 1] * box.height / 2
    center_pixels = axes.transData.transform(box.center)
    top_pixels = axes.transData.transform(top)
    direction = top_pixels - center_pixels
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        direction = np.array([0.0, 1.0])
        norm = 1.0
    rotation = axes.transData.inverted().transform(top_pixels + 24 * direction / norm)
    return np.vstack([corners(box), rotation])


def _update_scene_artists(
    axes: Axes,
    scene: Scene,
    assignment: Assignment,
    selected: str | Absent,
    arrows: bool,
    compact: bool,
    limits: tuple[float, float, float, float],
) -> bool:
    """Reuse scene artists while their counts and roles match the current scene."""
    boxes = (*scene.targets, *(p.box for p in scene.predictions))
    pair_count = len(assignment.pairs) if arrows else 0
    if (
        len(axes.patches) != 2 * len(boxes)
        or len(axes.texts) != len(boxes) + pair_count
        or len(axes.lines) != (3 if isinstance(selected, str) else 0)
        or not all(isinstance(patch, Polygon) for patch in axes.patches)
        or not all(isinstance(text, Annotation) for text in axes.texts)
        or not boxes
    ):
        return False
    if tuple(axes.get_xlim()) != limits[:2]:
        axes.set_xlim(limits[:2])
    if tuple(axes.get_ylim()) != limits[2:]:
        axes.set_ylim(limits[2:])
    unmatched = {scene.predictions[i].box.id for i in assignment.unmatched_predictions}
    unmatched.update(scene.targets[j].id for j in assignment.unmatched_targets)
    captions = [target.id for target in scene.targets] + [
        p.box.id if compact else f"{p.box.id}  p={p.confidence:.2f}" for p in scene.predictions
    ]
    for index, (box, caption) in enumerate(zip(boxes, captions, strict=True)):
        points = corners(box)
        color = GT_COLOR if index < len(scene.targets) else PRED_COLOR
        fill, outline = axes.patches[2 * index : 2 * index + 2]
        assert isinstance(fill, Polygon) and isinstance(outline, Polygon)
        fill.set_xy(points)
        fill.set_facecolor(color)
        fill.set_edgecolor(color)
        outline.set_xy(points)
        outline.set_edgecolor(color)
        outline.set_linewidth(2.5 if box.id == selected else 1.5)
        outline.set_linestyle("--" if box.id in unmatched else "-")
        label = axes.texts[index]
        assert isinstance(label, Annotation)
        label.xy = (box.cx, box.cy)
        label.set_color(color)
        label.set_text(caption + ("\nunmatched" if box.id in unmatched and not compact else ""))
    if arrows:
        for annotation, (i, j) in zip(axes.texts[len(boxes) :], assignment.pairs, strict=True):
            assert isinstance(annotation, Annotation)
            target, prediction = scene.targets[j], scene.predictions[i].box
            annotation.xy = (target.cx, target.cy)
            annotation.set_position((prediction.cx, prediction.cy))
    if isinstance(selected, str):
        box = find_box(scene, selected)
        handles = handle_points(axes, box)
        top = box.center + box.rotation[:, 1] * box.height / 2
        axes.lines[0].set_data([top[0], handles[4, 0]], [top[1], handles[4, 1]])
        axes.lines[1].set_data(handles[:4, 0], handles[:4, 1])
        axes.lines[2].set_data(handles[4:5, 0], handles[4:5, 1])
    return True


def draw_scene(
    axes: Axes,
    scene: Scene,
    assignment: Assignment,
    *,
    selected: str | Absent = NO_SELECTION,
    arrows: bool = True,
    limits: tuple[float, float, float, float] = (0, 100, 0, 100),
    compact: bool = False,
) -> None:
    """Render oriented geometry, reusing artists until scene topology changes."""
    if _update_scene_artists(axes, scene, assignment, selected, arrows, compact, limits):
        return
    axes.clear()
    axes.set(xlim=limits[:2], ylim=limits[2:], xlabel="x (scene units)", ylabel="y (scene units)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title("Drag body • resize squares • rotate circle", fontsize=11)
    if compact:
        axes.set(xlabel="", ylabel="", xticks=[], yticks=[])
    axes.grid(alpha=0.15)
    unmatched = {scene.predictions[i].box.id for i in assignment.unmatched_predictions}
    unmatched.update(scene.targets[j].id for j in assignment.unmatched_targets)
    for box, color, caption in (
        *((t, GT_COLOR, t.id) for t in scene.targets),
        *(
            (p.box, PRED_COLOR, p.box.id if compact else f"{p.box.id}  p={p.confidence:.2f}")
            for p in scene.predictions
        ),
    ):
        is_selected = box.id == selected
        axes.add_patch(
            Polygon(
                corners(box), closed=True, facecolor=color, edgecolor=color, alpha=0.16, linewidth=1
            )
        )
        axes.add_patch(
            Polygon(
                corners(box),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2.5 if is_selected else 1.5,
                linestyle="--" if box.id in unmatched else "-",
            )
        )
        label = caption + ("\nunmatched" if box.id in unmatched and not compact else "")
        axes.annotate(
            label,
            (box.cx, box.cy),
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=7 if compact else 9,
            fontweight="bold",
        )
    if arrows:
        for i, j in assignment.pairs:
            axes.annotate(
                "",
                (scene.targets[j].cx, scene.targets[j].cy),
                (scene.predictions[i].box.cx, scene.predictions[i].box.cy),
                arrowprops={"arrowstyle": "->", "color": MATCH_COLOR, "lw": 1.8},
            )
    if isinstance(selected, str):
        box = find_box(scene, selected)
        handles = handle_points(axes, box)
        top = box.center + box.rotation[:, 1] * box.height / 2
        axes.plot([top[0], handles[4, 0]], [top[1], handles[4, 1]], color="#111827", lw=1)
        axes.plot(handles[:4, 0], handles[:4, 1], "s", color="#111827", mfc="white", ms=7)
        axes.plot(handles[4, 0], handles[4, 1], "o", color="#111827", mfc="#fde047", ms=8)


def _update_matrix_artists(axes: Axes, values: FloatArray, assignment: Assignment) -> bool:
    """Update matrix colors, values, and match outlines without rebuilding the axes."""
    if (
        values.size == 0
        or len(axes.collections) != 1
        or not isinstance(axes.collections[0], QuadMesh)
        or len(axes.texts) != values.size
        or len(axes.patches) != len(assignment.pairs)
        or not all(isinstance(patch, Rectangle) for patch in axes.patches)
    ):
        return False
    mesh = axes.collections[0]
    assert isinstance(mesh, QuadMesh)
    previous_values = mesh.get_array()
    if previous_values is None or previous_values.shape != values.shape:
        return False
    mesh.set_array(values)
    low, high = float(values.min()), float(values.max())
    mesh.set_clim(low, high)
    for text, value in zip(axes.texts, values.flat, strict=True):
        text.set_text(f"{value:.4g}")
        text.set_color("black" if high > low and (value - low) / (high - low) > 0.65 else "white")
    for patch, (i, j) in zip(axes.patches, assignment.pairs, strict=True):
        assert isinstance(patch, Rectangle)
        patch.set_xy((j - 0.47, i - 0.47))
    return True


def draw_matrix(
    axes: Axes,
    scene: Scene,
    values: FloatArray,
    assignment: Assignment,
    title: str,
) -> None:
    """Draw a labeled matrix, reusing cells while its dimensions remain unchanged."""
    if _update_matrix_artists(axes, values, assignment):
        axes.set_title(title, fontsize=11)
        axes.set_xticks(range(len(scene.targets)), [t.id for t in scene.targets])
        axes.set_yticks(range(len(scene.predictions)), [p.box.id for p in scene.predictions])
        return
    axes.clear()
    axes.set_title(title, fontsize=11)
    axes.set_xlabel("Ground truth")
    axes.set_ylabel("Prediction")
    if values.size == 0:
        axes.text(
            0.5,
            0.5,
            f"Empty {values.shape[0]} × {values.shape[1]} cost matrix",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        axes.set_xticks([])
        axes.set_yticks([])
        return
    # Vector cells keep SVG previews free of the raster embedded by imshow.
    axes.pcolormesh(
        np.arange(values.shape[1] + 1) - 0.5,
        np.arange(values.shape[0] + 1) - 0.5,
        values,
        cmap="viridis",
        shading="flat",
        rasterized=False,
    )
    axes.set_xlim(-0.5, values.shape[1] - 0.5)
    axes.set_ylim(values.shape[0] - 0.5, -0.5)
    axes.set_xticks(range(len(scene.targets)), [t.id for t in scene.targets])
    axes.set_yticks(range(len(scene.predictions)), [p.box.id for p in scene.predictions])
    low, high = float(values.min()), float(values.max())
    for (i, j), value in np.ndenumerate(values):
        color = "black" if high > low and (value - low) / (high - low) > 0.65 else "white"
        axes.text(j, i, f"{value:.4g}", ha="center", va="center", color=color, fontsize=10)
    for i, j in assignment.pairs:
        axes.add_patch(
            Rectangle(
                (j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=MATCH_COLOR, linewidth=3
            )
        )


def pair_html(
    scene: Scene,
    config: MatcherConfig,
    evaluation: CostEvaluation,
    prediction_id: str,
    target_id: str,
) -> str:
    """Describe one pair, separating raw values, transformed costs, and weighted costs.

    Disabled metrics are evaluated here for inspection only and marked disabled.
    They never enter the matching total.
    """
    if not prediction_id or not target_id:
        return "<p>Add a prediction and a target to inspect a pair.</p>"
    i = next(i for i, p in enumerate(scene.predictions) if p.box.id == prediction_id)
    j = next(j for j, t in enumerate(scene.targets) if t.id == target_id)
    prediction, target = scene.predictions[i], scene.targets[j]
    components = {c.metric: c for c in evaluation.components}
    try:
        gaussian_pair: tuple[Gaussian2D, Gaussian2D] | Absent = (
            box_to_gaussian(prediction.box, convention=config.gaussian_convention),
            box_to_gaussian(target, convention=config.gaussian_convention),
        )
    except ValueError as error:
        gaussian_pair = Absent(str(error))
    rows = []
    for metric in Metric:
        try:
            if metric in GAUSSIAN_METRICS:
                match gaussian_pair:
                    case Absent(reason=reason):
                        raise ValueError(reason)
                    case (gaussian_prediction, gaussian_target):
                        value = gaussian_metric(metric, gaussian_prediction, gaussian_target)
            else:
                value = raw_metric(metric, prediction, target, config)
            raw = f"{value:.6g}"
        except ValueError as error:
            raw = f"unavailable: {escape(str(error))}"
        if metric in components:
            component = components[metric]
            cost, weighted = f"{component.cost[i, j]:.6g}", f"{component.weighted[i, j]:.6g}"
        else:
            cost, weighted = "disabled", "0"
        label = "p (foreground)" if metric == Metric.CLASSIFICATION else metric.value
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{raw}</td><td>{cost}</td><td>{weighted}</td></tr>"
        )
    match gaussian_pair:
        case Absent(reason=reason):
            gaussian_details = f"<p>Gaussian conversion unavailable: {escape(reason)}</p>"
        case (gaussian_prediction, gaussian_target):
            gaussian_details = gaussian_pair_html(
                gaussian_prediction,
                gaussian_target,
                config.gaussian_convention,
            )
    return (
        f"<b>{escape(prediction_id)} → {escape(target_id)}</b>"
        "<table style='border-spacing:12px 3px'><tr><th>Metric</th><th>Raw</th>"
        "<th>Cost</th><th>Weighted</th></tr>"
        + "".join(rows)
        + f"</table><b>Pair total: {evaluation.total[i, j]:.6g}</b>"
        + gaussian_details
    )


def gaussian_pair_html(
    prediction: Gaussian2D,
    target: Gaussian2D,
    convention: CovarianceConvention,
) -> str:
    """Show converter output and authoritative factor scales beside dense covariances."""
    rows = []
    for gaussian in (prediction, target):
        covariance = np.array2string(gaussian.covariance, precision=6, suppress_small=False)
        scales = np.array2string(np.array(gaussian.standard_deviations), precision=6)
        mean = np.array2string(np.array(gaussian.mean), precision=6)
        rows.append(
            f"<tr><td>{escape(gaussian.id)}</td><td><pre>{mean}</pre></td>"
            f"<td><pre>{covariance}</pre></td><td><pre>{scales}</pre></td></tr>"
        )
    return (
        f"<p><b>Shared box → Gaussian converter:</b> {escape(convention.value)}</p>"
        "<table style='border-spacing:12px 3px'><tr><th>Box</th><th>Mean μ</th>"
        "<th>Covariance Σ (scene units²)</th><th>Principal std (scene units)</th></tr>"
        + "".join(rows)
        + "</table><p>Distances use the principal-axis factor; dense covariance "
        "display can lose the small eigenvalue for extreme aspect ratios.</p>"
    )


def static_preview(scene: Scene, config: MatcherConfig) -> str:
    """Return self-contained SVG without a widget backend or raster image payloads."""
    evaluation = evaluate_costs(scene.predictions, scene.targets, config)
    assignment = solve_assignment(evaluation.total)
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasSVG(figure)
    scene_axes, matrix_axes = figure.subplots(1, 2)
    draw_scene(scene_axes, scene, assignment)
    draw_matrix(
        matrix_axes, scene, evaluation.total, assignment, "Combined cost · outlined = matched"
    )
    output = StringIO()
    figure.savefig(output, format="svg", metadata={"Date": None})
    return output.getvalue()


def draw_metric_panel(
    axes: Axes,
    scene: Scene,
    result: MetricMatch | MetricFailure,
    *,
    limits: tuple[float, float, float, float] = (0, 100, 0, 100),
    arrows: bool = True,
    focused: bool = False,
) -> None:
    """Show one independent assignment with common coordinates and a focus border."""
    match result:
        case MetricFailure(reason=reason):
            axes.clear()
            axes.set(xticks=[], yticks=[])
            axes.text(
                0.5,
                0.5,
                f"Unavailable\n{reason}",
                ha="center",
                va="center",
                wrap=True,
                fontsize=7,
                transform=axes.transAxes,
            )
        case MetricMatch(assignment=assignment):
            draw_scene(axes, scene, assignment, limits=limits, arrows=arrows, compact=True)
            pairs = (
                " · ".join(
                    f"{scene.predictions[i].box.id}→{scene.targets[j].id}"
                    for i, j in assignment.pairs
                )
                or "No pairs"
            )
            axes.set_xlabel(f"{pairs}\nΣ cost = {assignment.total:.4g}", fontsize=7)
    axes.set_title(result.metric.value, fontsize=9, fontweight="bold")
    axes.set_facecolor("#faf5ff" if focused else "white")
    for spine in axes.spines.values():
        spine.set_color(MATCH_COLOR if focused else "#9ca3af")
        spine.set_linewidth(2 if focused else 0.7)


def comparison_preview(scene: Scene, config: MatcherConfig) -> str:
    """Export all geometry-only assignments as one static, vector-only SVG grid."""
    results = compare_metrics(scene, config)
    figure = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasSVG(figure)
    axes = figure.subplots(2, 5).flat
    for axis, result in zip(axes, results, strict=True):
        draw_metric_panel(axis, scene, result)
    figure.suptitle(
        "Same boxes · independent Hungarian assignments · totals have different units", fontsize=12
    )
    output = StringIO()
    figure.savefig(output, format="svg", metadata={"Date": None})
    return output.getvalue()
