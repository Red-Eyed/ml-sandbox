"""Thin stateful Jupyter UI for editing boxes and observing rectangular assignment."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from html import escape
from math import atan2, degrees, radians
from time import monotonic
from typing import Any, Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._base import _AxesBase
from matplotlib.backend_bases import Event, MouseButton, MouseEvent, RendererBase
from matplotlib.transforms import Bbox

from .assignment import Assignment, solve_assignment
from .comparison import GEOMETRIC_METRICS, MetricFailure, MetricMatch, compare_metrics
from .costs import (
    CostEvaluation,
    CostTerm,
    MatcherConfig,
    Metric,
    Prediction,
    Transform,
    evaluate_costs,
)
from .gaussian import CovarianceConvention
from .geometry import Box, FloatArray, contains, resize_from_corner
from .presets import Scene, baseline, scene_presets
from .visualization import (
    Absent,
    draw_matrix,
    draw_metric_panel,
    draw_scene,
    find_box,
    handle_points,
    pair_html,
)


def mouse_callback(callback: Callable[[MouseEvent], None]) -> Callable[[Event], None]:
    """Narrow Matplotlib's generic callback contract at the third-party boundary."""

    def dispatch(event: Event) -> None:
        """Forward mouse events only; ignore other Matplotlib event variants."""
        match event:
            case MouseEvent():
                callback(event)

    return dispatch


@dataclass(frozen=True)
class Drag:
    """Pointer gesture bound to its original box, preserving the opposite resize corner."""

    box: Box
    start: FloatArray
    kind: Literal["move", "resize", "rotate"]
    handle: int = 0


@dataclass(frozen=True)
class TermControl:
    """The checkbox and weight slider that configure one cost term."""

    enabled: widgets.Checkbox
    weight: widgets.FloatSlider


class MatcherLab:
    """Own one editable scene, configuration, figure, and widget callback lifecycle.

    Construct after enabling ``%matplotlib widget``. Display ``.view`` to start;
    ``close()`` disconnects callbacks and closes this application's resources.
    Mouse motion is rate-limited; release always applies the final position.
    """

    def __init__(self, scene: Scene, config: MatcherConfig | None = None) -> None:
        """Initialize all controls and validate the scene before displaying the UI."""
        self.scene = scene
        self.config = config if config is not None else baseline()
        self.selected: str | Absent = Absent("No box selected")
        self.drag: Drag | Absent = Absent("No active gesture")
        self.updating = False
        self.closed = False
        self.last_motion = 0.0
        self.frame_pending = False
        self.minimum_size = 0.01
        self.limits = (0.0, 100.0, 0.0, 100.0)
        self.evaluation = evaluate_costs(scene.predictions, scene.targets, self.config)
        self.assignment = solve_assignment(self.evaluation.total)
        self.evaluation_error: str | Absent = Absent("Evaluation succeeded")
        self.comparisons = compare_metrics(scene, self.config)
        with plt.ioff():
            self.figure = plt.figure(figsize=(10.5, 9), layout="constrained")
            grid = self.figure.add_gridspec(5, 4)
            self.scene_axes = self.figure.add_subplot(grid[:3, :2])
            self.matrix_axes = self.figure.add_subplot(grid[3:, :2])
            self.matrix_axes.set_axes_locator(self._matrix_bounds)
            self.metric_axes = {
                metric: self.figure.add_subplot(grid[index // 2, 2 + index % 2])
                for index, metric in enumerate(GEOMETRIC_METRICS)
            }
        if not isinstance(self.figure.canvas, widgets.Widget):
            plt.close(self.figure)
            raise TypeError("Enable %matplotlib widget before constructing MatcherLab")
        self.figure.canvas.toolbar_visible = False
        self.figure.canvas.header_visible = False
        self.figure.canvas.footer_visible = False
        self._build_controls()
        self._load_config_controls()
        self._rebuild_scene_controls()
        self.connections = [
            self.figure.canvas.mpl_connect("draw_event", self._frame_drawn),
            self.figure.canvas.mpl_connect("resize_event", self._canvas_resized),
            self.figure.canvas.mpl_connect("button_press_event", mouse_callback(self._press)),
            self.figure.canvas.mpl_connect("motion_notify_event", mouse_callback(self._motion)),
            self.figure.canvas.mpl_connect("button_release_event", mouse_callback(self._release)),
        ]
        self.refresh()

    def _frame_drawn(self, event: Event) -> None:
        """Release frame backpressure and freeze the completed layout between resizes."""
        self.frame_pending = False
        self.figure.set_layout_engine("none")

    def _canvas_resized(self, event: Event) -> None:
        """Recalculate subplot positions once after an actual canvas resize."""
        self.figure.set_layout_engine("constrained")

    def _request_frame(self) -> None:
        """Coalesce redraw requests until the frontend consumes the current frame."""
        if self.frame_pending:
            return
        self.frame_pending = True
        self.figure.canvas.draw_idle()

    def _matrix_bounds(self, axes: _AxesBase, renderer: RendererBase) -> Bbox:
        """Keep small cost matrices compact instead of stretching into the subplot slot."""
        slot = axes.get_position(original=True)
        figure_width, figure_height = self.figure.get_size_inches()
        columns = max(1, len(self.scene.targets))
        rows = max(1, len(self.scene.predictions))
        width = min(slot.width, max(2.8, columns * 0.8) / figure_width)
        height = min(slot.height, max(0.65, rows * 0.32) / figure_height)
        return Bbox.from_bounds(slot.x0 + (slot.width - width) / 2, slot.y1 - height, width, height)

    def _build_controls(self) -> None:
        """Compose the scene editor, matcher settings, matrix selector, and pair inspector."""
        self.status = widgets.HTML()
        self.details = widgets.HTML()
        self.formula = widgets.HTML()
        self._build_scene_controls()
        self._build_cost_controls()
        self._build_inspection_controls()
        buttons = [
            self._button("Add GT", partial(self._add, False)),
            self._button("Add prediction", partial(self._add, True)),
            self._button("Remove selected", self._remove),
            self._button("Fit scene", self._fit),
            self._button("Reset baseline", self._reset),
        ]
        self.view = widgets.VBox(
            [
                widgets.HTML(
                    "<h3>Hungarian matcher · oriented text boxes</h3>"
                    "<p>Green: GT · Blue: prediction · Magenta: assignment. "
                    "Dashed outline: unmatched. Edit boxes on the large canvas; all ten "
                    "geometry-only panels update together. Click a panel to inspect its own "
                    "assignment and costs. Totals have different units and cannot rank metrics.</p>"
                ),
                widgets.HBox(
                    [self.scene_choice, self.show_arrows, self.matrix_choice],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                self.figure.canvas,
                self.status,
                widgets.HBox(buttons, layout=widgets.Layout(flex_flow="row wrap")),
                self.selection,
                widgets.HBox(
                    list(self.numeric.values()), layout=widgets.Layout(flex_flow="row wrap")
                ),
                self.minimum_control,
                widgets.HTML("<b>Independent foreground confidences</b>"),
                self.confidences,
                widgets.HTML("<b>Weighted matcher (Combined view only)</b>"),
                *[
                    widgets.HBox([term.enabled, term.weight])
                    for term in self.term_controls.values()
                ],
                widgets.HTML("<b>Shared metric settings</b>"),
                widgets.HBox([self.transform, self.epsilon]),
                self.hausdorff_points,
                self.gaussian_convention,
                widgets.HBox(
                    [self.x_scale, self.y_scale, self.angle_weight],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                self.formula,
                widgets.HBox(
                    [self.pair_prediction, self.pair_target],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                self.details,
            ]
        )

    def _build_scene_controls(self) -> None:
        """Build geometry selection and scene-level display controls."""
        self.selection = widgets.Dropdown(description="Edit box", options=[("None", "")])
        self.selection.observe(self._select, names="value")
        self.numeric = {
            name: widgets.FloatText(description=name, layout=widgets.Layout(width="175px"))
            for name in ("cx", "cy", "width", "height", "angle °")
        }
        for control in self.numeric.values():
            control.observe(self._numeric_changed, names="value")
        self.confidences = widgets.VBox()
        self.show_arrows = widgets.Checkbox(value=True, description="Show assignment arrows")
        self.show_arrows.observe(self._view_changed, names="value")
        self.scene_choice = widgets.Dropdown(
            description="Scene", options=["Custom", *scene_presets()]
        )
        self.scene_choice.observe(self._scene_changed, names="value")

    def _build_cost_controls(self) -> None:
        """Build independent term toggles, weights, and numerical parameters."""
        self.term_controls = {
            metric: TermControl(
                widgets.Checkbox(
                    description=metric.value, indent=False, layout=widgets.Layout(width="205px")
                ),
                widgets.FloatSlider(
                    min=0,
                    max=10,
                    step=0.1,
                    readout_format=".2f",
                    continuous_update=True,
                    layout=widgets.Layout(width="260px"),
                ),
            )
            for metric in Metric
        }
        for term in self.term_controls.values():
            term.enabled.observe(self._config_changed, names="value")
            term.weight.observe(self._config_changed, names="value")
        self.transform = widgets.Dropdown(
            description="Similarity",
            options=[t.value for t in Transform],
            layout=widgets.Layout(width="350px"),
        )
        self.epsilon = widgets.FloatLogSlider(
            description="Log floor ε", base=10, min=-15, max=-2, value=1e-12, readout_format=".0e"
        )
        self.x_scale = widgets.FloatText(description="x scale", value=100)
        self.y_scale = widgets.FloatText(description="y scale", value=100)
        self.angle_weight = widgets.FloatSlider(description="L1 angle", min=0, max=5, step=0.1)
        self.hausdorff_points = widgets.Dropdown(
            description="Hausdorff points",
            options=[4, 8, 32],
            value=4,
            style={"description_width": "initial"},
        )
        self.gaussian_convention = widgets.Dropdown(
            description="Gaussian covariance",
            options=[(c.value, c.value) for c in CovarianceConvention],
            value=CovarianceConvention.UNIFORM.value,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )
        self.minimum_control = widgets.BoundedFloatText(
            description="Min side", value=0.01, min=1e-9, max=1e6
        )
        self.minimum_control.observe(self._minimum_changed, names="value")
        for control in (
            self.transform,
            self.epsilon,
            self.x_scale,
            self.y_scale,
            self.angle_weight,
            self.hausdorff_points,
            self.gaussian_convention,
        ):
            control.observe(self._config_changed, names="value")

    def _build_inspection_controls(self) -> None:
        """Build selectors for the matrix and detailed pair inspection."""
        self.matrix_choice = widgets.Dropdown(
            description="Inspect",
            options=["Combined", *(metric.value for metric in GEOMETRIC_METRICS)],
            layout=widgets.Layout(width="290px"),
        )
        self.matrix_choice.observe(self._view_changed, names="value")
        self.pair_prediction = widgets.Dropdown(description="Pair pred", options=[("None", "")])
        self.pair_target = widgets.Dropdown(description="Pair GT", options=[("None", "")])
        self.pair_prediction.observe(self._view_changed, names="value")
        self.pair_target.observe(self._view_changed, names="value")

    def _button(self, label: str, callback: Any) -> widgets.Button:
        """Bind a command button to this application's state transition."""
        button = widgets.Button(description=label)
        button.on_click(callback)
        return button

    def _load_config_controls(self) -> None:
        """Reflect the configuration without triggering intermediate recomputations."""
        self.updating = True
        weights = {term.metric: term.weight for term in self.config.terms}
        for metric, term in self.term_controls.items():
            term.enabled.value = metric in weights
            term.weight.max = max(10, weights.get(metric, 1))
            term.weight.value = weights.get(metric, 1)
            term.weight.disabled = not term.enabled.value
        self.transform.value = self.config.transform.value
        self.epsilon.min = min(-15, float(np.log10(self.config.epsilon)))
        self.epsilon.max = max(-2, float(np.log10(self.config.epsilon)))
        self.epsilon.value = self.config.epsilon
        self.x_scale.value, self.y_scale.value = self.config.x_scale, self.config.y_scale
        self.angle_weight.max = max(5, self.config.angle_weight)
        self.angle_weight.value = self.config.angle_weight
        self.hausdorff_points.value = self.config.hausdorff_points
        self.gaussian_convention.value = self.config.gaussian_convention.value
        self.updating = False

    def _rebuild_scene_controls(self) -> None:
        """Reconcile identities and confidence sliders after additions, removals, or scene loads."""
        self.updating = True
        self.scene_choice.value = next(
            (name for name, scene in scene_presets().items() if scene == self.scene), "Custom"
        )
        identifiers = [t.id for t in self.scene.targets] + [
            p.box.id for p in self.scene.predictions
        ]
        if self.selected not in identifiers:
            self.selected = Absent("Selection removed or scene replaced")
        self.selection.options = [
            ("None", ""),
            *((identifier, identifier) for identifier in identifiers),
        ]
        self.selection.value = self.selected if isinstance(self.selected, str) else ""
        old_sliders = self.confidences.children
        sliders = []
        for prediction in self.scene.predictions:
            slider = widgets.FloatSlider(
                description=prediction.box.id,
                min=0,
                max=1,
                step=0.01,
                value=prediction.confidence,
                continuous_update=True,
            )
            slider.observe(partial(self._confidence_changed, prediction.box.id), names="value")
            sliders.append(slider)
        self.confidences.children = tuple(sliders)
        for slider in old_sliders:
            slider.close()
        self.pair_prediction.options = [(p.box.id, p.box.id) for p in self.scene.predictions] or [
            ("None", "")
        ]
        self.pair_target.options = [(t.id, t.id) for t in self.scene.targets] or [("None", "")]
        self.updating = False
        self._sync_numeric()

    def _sync_numeric(self) -> None:
        """Reflect the selected geometry without feeding changes back into the scene."""
        self.updating = True
        enabled = isinstance(self.selected, str)
        for control in self.numeric.values():
            control.disabled = not enabled
        if isinstance(self.selected, str):
            box = find_box(self.scene, self.selected)
            values = (box.cx, box.cy, box.width, box.height, degrees(box.angle))
            for control, value in zip(self.numeric.values(), values, strict=True):
                control.value = value
        self.updating = False

    def _mark_scene_custom(self) -> None:
        """Keep the scene selector honest after any user edit."""
        self.updating = True
        self.scene_choice.value = "Custom"
        self.updating = False

    def _replace_box(self, box: Box) -> None:
        """Replace geometry while preserving identity and independent prediction confidence."""
        self._mark_scene_custom()
        self.scene = Scene(
            tuple(box if t.id == box.id else t for t in self.scene.targets),
            tuple(replace(p, box=box) if p.box.id == box.id else p for p in self.scene.predictions),
        )

    def _select(self, change: Any) -> None:
        """Select through the dropdown and expose matching mouse handles."""
        if self.updating:
            return
        self.selected = change["new"] or Absent("No box selected")
        self._sync_numeric()
        self.refresh()

    def _numeric_changed(self, change: Any) -> None:
        """Apply numeric edits immediately, retaining valid geometry on bad input."""
        if self.updating or isinstance(self.selected, Absent):
            return
        try:
            values = {name: control.value for name, control in self.numeric.items()}
            box = Box(
                self.selected,
                values["cx"],
                values["cy"],
                values["width"],
                values["height"],
                radians(values["angle °"]),
            )
            self._replace_box(box)
            self.refresh()
        except ValueError as error:
            self._sync_numeric()
            self._show_error(error)

    def _confidence_changed(self, identifier: str, change: Any) -> None:
        """Update only one prediction's foreground probability."""
        if self.updating:
            return
        self._mark_scene_custom()
        self.scene = replace(
            self.scene,
            predictions=tuple(
                replace(p, confidence=float(change["new"])) if p.box.id == identifier else p
                for p in self.scene.predictions
            ),
        )
        self.refresh()

    def _config_changed(self, change: Any) -> None:
        """Validate widget input at the UI boundary before evaluating a new matcher."""
        if self.updating:
            return
        try:
            self.config = MatcherConfig(
                tuple(
                    CostTerm(metric, term.weight.value)
                    for metric, term in self.term_controls.items()
                    if term.enabled.value
                ),
                Transform(self.transform.value),
                self.epsilon.value,
                self.x_scale.value,
                self.y_scale.value,
                self.angle_weight.value,
                self.hausdorff_points.value,
                CovarianceConvention(self.gaussian_convention.value),
            )
        except ValueError as error:
            self._load_config_controls()
            self._show_error(error)
            return
        for term in self.term_controls.values():
            term.weight.disabled = not term.enabled.value
        self.refresh()

    def _scene_changed(self, change: Any) -> None:
        """Replace the scene reproducibly, canceling any active gesture."""
        if self.updating or change["new"] == "Custom":
            return
        self.scene = scene_presets()[change["new"]]
        self.drag = Absent("Scene replaced")
        self._rebuild_scene_controls()
        self._fit()

    def _minimum_changed(self, change: Any) -> None:
        """Set the visible minimum size used only by mouse resizing."""
        self.minimum_size = float(change["new"])

    def _view_changed(self, change: Any) -> None:
        """Redraw presentation controls without re-evaluating pairwise costs."""
        if not self.updating:
            self._draw()

    def _add(self, prediction: bool, button: Any) -> None:
        """Add a uniquely named box at the viewport center without renumbering existing boxes."""
        prefix = "P" if prediction else "G"
        identifiers = {t.id for t in self.scene.targets} | {
            p.box.id for p in self.scene.predictions
        }
        index = 1
        while f"{prefix}{index}" in identifiers:
            index += 1
        x0, x1, y0, y1 = self.limits
        box = Box(f"{prefix}{index}", (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 12)
        if prediction:
            self.scene = replace(
                self.scene, predictions=(*self.scene.predictions, Prediction(box, 0.5))
            )
        else:
            self.scene = replace(self.scene, targets=(*self.scene.targets, box))
        self.selected = box.id
        self._rebuild_scene_controls()
        self.refresh()

    def _remove(self, button: Any) -> None:
        """Remove the selected identity and leave all remaining identities unchanged."""
        if isinstance(self.selected, Absent):
            return
        self.scene = Scene(
            tuple(t for t in self.scene.targets if t.id != self.selected),
            tuple(p for p in self.scene.predictions if p.box.id != self.selected),
        )
        self.drag = Absent("Selected box removed")
        self._rebuild_scene_controls()
        self.refresh()

    def _fit(self, button: Any = None) -> None:
        """Frame all corners with a margin, without changing L1 normalization units."""
        from .geometry import corners

        boxes = (*self.scene.targets, *(p.box for p in self.scene.predictions))
        if boxes:
            points = np.vstack([corners(box) for box in boxes])
            low, high = points.min(axis=0), points.max(axis=0)
            side = max(float(np.max(high - low)) * 1.35, 1.0)
            center = (low + high) / 2
            self.limits = (
                float(center[0] - side / 2),
                float(center[0] + side / 2),
                float(center[1] - side / 2),
                float(center[1] + side / 2),
            )
        else:
            self.limits = (0.0, 100.0, 0.0, 100.0)
        self.refresh()

    def _reset(self, button: Any) -> None:
        """Restore the declared baseline while preserving the user's scene."""
        self.config = baseline()
        self._load_config_controls()
        self.refresh()

    def _press(self, event: MouseEvent) -> None:
        """Hit-test handles in screen pixels, then oriented bodies in scene coordinates."""
        if event.button != MouseButton.LEFT:
            return
        for metric, axes in self.metric_axes.items():
            if event.inaxes is axes:
                self.matrix_choice.value = metric.value
                return
        if event.inaxes is not self.scene_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata], dtype=np.float64)
        if isinstance(self.selected, str):
            box = find_box(self.scene, self.selected)
            pixels = self.scene_axes.transData.transform(handle_points(self.scene_axes, box))
            distances = np.linalg.norm(pixels - np.array([event.x, event.y]), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 11:
                self.drag = Drag(box, point, "rotate" if nearest == 4 else "resize", nearest)
                return
        boxes = (*self.scene.targets, *(p.box for p in self.scene.predictions))
        # Repeated clicks on overlaps are supplemented by the explicit identity selector.
        for box in reversed(boxes):
            if contains(box, point):
                self.selected = box.id
                self.drag = Drag(box, point, "move")
                self.updating = True
                self.selection.value = box.id
                self.updating = False
                self._sync_numeric()
                self._draw()
                return
        self.selected = Absent("Clicked empty canvas")
        self.selection.value = ""

    def _apply_pointer(self, event: MouseEvent) -> None:
        """Apply a gesture in the original local frame, including release outside the axes."""
        if isinstance(self.drag, Absent):
            return
        point = np.asarray(
            self.scene_axes.transData.inverted().transform((event.x, event.y)), dtype=np.float64
        )
        drag = self.drag
        match drag.kind:
            case "move":
                delta = point - drag.start
                box = replace(
                    drag.box, cx=drag.box.cx + float(delta[0]), cy=drag.box.cy + float(delta[1])
                )
            case "resize":
                box = resize_from_corner(drag.box, drag.handle, point, self.minimum_size)
            case "rotate":
                initial = drag.start - drag.box.center
                current = point - drag.box.center
                angle = (
                    drag.box.angle + atan2(current[1], current[0]) - atan2(initial[1], initial[0])
                )
                box = replace(drag.box, angle=(angle + np.pi) % (2 * np.pi) - np.pi)
        self._replace_box(box)
        self._sync_numeric()
        self.refresh()

    def _motion(self, event: MouseEvent) -> None:
        """Limit interactive recomputation to at most thirty requests per second."""
        now = monotonic()
        if self.frame_pending or isinstance(self.drag, Absent) or now - self.last_motion < 1 / 30:
            return
        self.last_motion = now
        self._apply_pointer(event)

    def _release(self, event: MouseEvent) -> None:
        """Always evaluate the final mouse position before ending a gesture."""
        self._apply_pointer(event)
        self.drag = Absent("Pointer released")

    def _show_error(self, error: ValueError) -> None:
        """Surface a recoverable invalid edit without hiding unexpected programming errors."""
        self.status.value = f"<b style='color:#b91c1c'>Invalid input: {escape(str(error))}</b>"

    def refresh(self) -> None:
        """Recompute assignment and synchronize every visible diagnostic."""
        self.comparisons = compare_metrics(self.scene, self.config)
        try:
            evaluation = evaluate_costs(self.scene.predictions, self.scene.targets, self.config)
            assignment = solve_assignment(evaluation.total)
        except ValueError as error:
            self.evaluation_error = str(error)
            self._draw()
            return
        self.evaluation_error = Absent("Evaluation succeeded")
        self.evaluation, self.assignment = evaluation, assignment
        self._draw()

    def _draw(self) -> None:
        """Render independent metric panels and the currently focused diagnostic."""
        for result in self.comparisons:
            draw_metric_panel(
                self.metric_axes[result.metric],
                self.scene,
                result,
                limits=self.limits,
                arrows=self.show_arrows.value,
                focused=self.matrix_choice.value == result.metric.value,
            )
        config = self.config
        evaluation, assignment = self.evaluation, self.assignment
        error = self.evaluation_error
        title = "Combined cost · outlined = matched"
        if self.matrix_choice.value != "Combined":
            result = next(r for r in self.comparisons if r.metric.value == self.matrix_choice.value)
            match result:
                case MetricFailure(reason=reason):
                    error = reason
                case MetricMatch(component=component, assignment=focused_assignment):
                    error = Absent("Focused metric succeeded")
                    assignment = focused_assignment
                    evaluation = CostEvaluation(component.cost, (component,))
                    config = replace(config, terms=(CostTerm(result.metric, 1),))
                    title = f"{result.metric.value} cost · own assignment"
        if isinstance(error, str):
            self.scene_axes.clear()
            self.matrix_axes.clear()
            self.details.value = (
                "<p>Assignment unavailable until the invalid input is corrected.</p>"
            )
            self._show_error(ValueError(error))
            self._request_frame()
            return
        draw_scene(
            self.scene_axes,
            self.scene,
            assignment,
            selected=self.selected,
            arrows=self.show_arrows.value,
            limits=self.limits,
        )
        self.scene_axes.set_title(
            f"{self.matrix_choice.value} · drag / resize / rotate", fontsize=11
        )
        draw_matrix(self.matrix_axes, self.scene, evaluation.total, assignment, title)
        self._update_status(evaluation, assignment)
        self._update_formula()
        self.details.value = pair_html(
            self.scene,
            config,
            evaluation,
            self.pair_prediction.value,
            self.pair_target.value,
        )
        self._request_frame()

    def _update_status(self, evaluation: CostEvaluation, assignment: Assignment) -> None:
        """Describe selected pairs, unmatched identities, and tied configurations."""
        pairs = (
            ", ".join(
                f"{self.scene.predictions[i].box.id} → {self.scene.targets[j].id}"
                for i, j in assignment.pairs
            )
            or "No pairs"
        )
        unmatched_p = (
            ", ".join(self.scene.predictions[i].box.id for i in assignment.unmatched_predictions)
            or "none"
        )
        unmatched_g = (
            ", ".join(self.scene.targets[j].id for j in assignment.unmatched_targets) or "none"
        )
        tie_note = (
            " · All costs disabled or zero: arbitrary tied assignment"
            if not evaluation.components
            else ""
        )
        self.status.value = (
            f"<b>{escape(self.matrix_choice.value)}: {escape(pairs)}</b> · Total: {assignment.total:.6g} · "
            f"Unmatched predictions: {escape(unmatched_p)} · Unmatched GT: {escape(unmatched_g)}"
            f"{tie_note}<br>Heatmap colors rescale for each view; compare printed values."
        )

    def _update_formula(self) -> None:
        """Explain the exact active normalization and cost transformations."""
        self.formula.value = (
            f"<p><b>Similarity cost:</b> {escape(self.config.transform.value)}; "
            f"ε={self.config.epsilon:g}. <b>Classification:</b> −log(max(p, ε)).<br>"
            f"<b>L1:</b> (|Δcx|+|Δw|)/{self.config.x_scale:g} + "
            f"(|Δcy|+|Δh|)/{self.config.y_scale:g} + "
            f"{self.config.angle_weight:g} × periodic |Δθ|/π. "
            "Normalization stays fixed when zooming.<br>"
            f"<b>Hausdorff:</b> symmetric nearest-point maximum over "
            f"{self.config.hausdorff_points} boundary points. "
            "Normalize cx/w by x scale and cy/h by y scale before rotation, preserving angle "
            "(RHINO convention). Lower is better; no similarity transform.<br>"
            f"<b>Gaussian covariance:</b> {escape(self.config.gaussian_convention.value)}. "
            "Scene coordinates are used directly; x/y normalization controls apply to L1/Hausdorff. "
            "KL and Bhattacharyya are dimensionless; W2 has scene units; "
            "Hellinger is in [0,1]. Symmetric KL is the mean of both directions. "
            "Distance costs ignore the similarity transform. All Gaussian methods lose square-box orientation.<br>"
            "Single class: confidence is constant across each row; it can select rows in an "
            "overcomplete scene but cannot change a square assignment by itself. "
            "Equal-cost optima may switch without a change in optimal total.</p>"
        )

    def close(self) -> None:
        """Disconnect Matplotlib callbacks and close all widgets owned by this instance."""
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            self.figure.canvas.mpl_disconnect(connection)
        descendants: list[widgets.Widget] = [self.view]
        while descendants:
            widget = descendants.pop()
            descendants.extend(getattr(widget, "children", ()))
            widget.close()
        plt.close(self.figure)
