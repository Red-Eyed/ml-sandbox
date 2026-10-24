"""Shared scene and matrix rendering for the live canvas and static SVG preview."""

from dataclasses import dataclass
from html import escape
from io import StringIO

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from .assignment import Assignment, solve_assignment
from .costs import CostEvaluation, MatcherConfig, Metric, bhattacharyya, evaluate_costs, raw_metric
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


def draw_scene(
    axes: Axes,
    scene: Scene,
    assignment: Assignment,
    *,
    selected: str | Absent = NO_SELECTION,
    arrows: bool = True,
    limits: tuple[float, float, float, float] = (0, 100, 0, 100),
) -> None:
    """Render oriented geometry, unmatched status, and optional editing handles."""
    axes.clear()
    axes.set(xlim=limits[:2], ylim=limits[2:], xlabel="x (scene units)", ylabel="y (scene units)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title("Drag body • resize squares • rotate circle", fontsize=11)
    axes.grid(alpha=0.15)
    unmatched = {scene.predictions[i].box.id for i in assignment.unmatched_predictions}
    unmatched.update(scene.targets[j].id for j in assignment.unmatched_targets)
    for box, color, caption in (
        *((t, GT_COLOR, t.id) for t in scene.targets),
        *((p.box, PRED_COLOR, f"{p.box.id}  p={p.confidence:.2f}") for p in scene.predictions),
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
        label = caption + ("\nunmatched" if box.id in unmatched else "")
        axes.annotate(
            label,
            (box.cx, box.cy),
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=9,
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


def draw_matrix(
    axes: Axes,
    scene: Scene,
    values: FloatArray,
    assignment: Assignment,
    title: str,
) -> None:
    """Draw a labeled matrix with selected solver entries outlined in magenta."""
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
    rows = []
    for metric in Metric:
        try:
            raw = f"{raw_metric(metric, prediction, target, config):.6g}"
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
    try:
        distance = f"{bhattacharyya(prediction.box, target):.6g}"
    except ValueError as error:
        distance = f"unavailable: {escape(str(error))}"
    return (
        f"<b>{escape(prediction_id)} → {escape(target_id)}</b>"
        "<table style='border-spacing:12px 3px'><tr><th>Metric</th><th>Raw</th>"
        "<th>Cost</th><th>Weighted</th></tr>"
        + "".join(rows)
        + f"</table><b>Pair total: {evaluation.total[i, j]:.6g}</b>"
        f" · Gaussian Bhattacharyya distance: {distance}"
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
