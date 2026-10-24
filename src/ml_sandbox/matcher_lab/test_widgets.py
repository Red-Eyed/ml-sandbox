"""Exercise actual ipympl callbacks and widget events without a browser frontend."""

from collections.abc import Iterator
from math import pi
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

from .costs import Metric, Transform
from .geometry import corners
from .presets import baseline, scene_presets
from .visualization import Absent, find_box, handle_points, static_preview
from .widgets import MatcherLab


@pytest.fixture
def lab() -> Iterator[MatcherLab]:
    """Use the real widget backend and release all owned callbacks after each test."""
    previous = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    application = MatcherLab(scene_presets()["Rotated crossing"])
    application.figure.canvas.draw()
    yield application
    application.close()
    plt.switch_backend(previous)


def pointer(lab: MatcherLab, name: str, point: np.ndarray) -> None:
    """Dispatch a data-space gesture through Matplotlib's registered callback path."""
    x, y = lab.scene_axes.transData.transform(point)
    event = MouseEvent(name, lab.figure.canvas, x, y, button=MouseButton.LEFT)
    lab.figure.canvas.callbacks.process(name, event)


def test_mouse_move_and_release(lab: MatcherLab) -> None:
    """Mouse editing recomputes matching and release consumes the final unsampled position."""
    lab.selection.value = "P1"
    original = find_box(lab.scene, "P1")
    pointer(lab, "button_press_event", original.center)
    pointer(lab, "motion_notify_event", original.center + [1, 1])
    pointer(lab, "button_release_event", original.center + [5, 3])
    moved = find_box(lab.scene, "P1")
    np.testing.assert_allclose(moved.center, original.center + [5, 3], atol=0.4)
    assert isinstance(lab.drag, Absent)
    assert lab.assignment.pairs == ((0, 0),)


def test_mouse_resize_in_rotated_frame(lab: MatcherLab) -> None:
    """The actual resize callback fixes the opposite corner of a rotated OBB."""
    lab.selection.value = "P1"
    lab.figure.canvas.draw()
    original = find_box(lab.scene, "P1")
    anchor = corners(original)[0]
    target = anchor + np.array([70, 12]) @ original.rotation.T
    pointer(lab, "button_press_event", corners(original)[2])
    pointer(lab, "button_release_event", target)
    resized = find_box(lab.scene, "P1")
    np.testing.assert_allclose(corners(resized)[0], anchor, atol=1e-10)
    assert resized.width == pytest.approx(70, abs=0.4)
    assert resized.height == pytest.approx(12, abs=0.4)
    assert resized.angle == original.angle


def test_mouse_rotate_preserves_center_and_dimensions(lab: MatcherLab) -> None:
    """The rotation handle changes only orientation and produces updated numerical controls."""
    lab.selection.value = "P1"
    lab.figure.canvas.draw()
    original = find_box(lab.scene, "P1")
    handle = handle_points(lab.scene_axes, original)[4]
    offset = handle - original.center
    target = original.center + np.array([-offset[1], offset[0]])
    pointer(lab, "button_press_event", handle)
    pointer(lab, "button_release_event", target)
    rotated = find_box(lab.scene, "P1")
    np.testing.assert_allclose(rotated.center, original.center)
    assert rotated.width == original.width
    assert rotated.height == original.height
    assert rotated.angle == pytest.approx(original.angle + pi / 2, abs=0.03)


def test_numeric_confidence_and_config_events(lab: MatcherLab) -> None:
    """Widget edits update state without cell execution, and invalid edits remain recoverable."""
    lab.selection.value = "P1"
    lab.numeric["cx"].value = 40
    assert find_box(lab.scene, "P1").cx == 40
    lab.numeric["width"].value = -1
    assert find_box(lab.scene, "P1").width == 60
    assert "Invalid input" in lab.status.value
    lab.confidences.children[0].value = 0.2
    assert lab.scene.predictions[0].confidence == 0.2
    lab.term_controls[Metric.PROBIOU].enabled.value = True
    lab.transform.value = Transform.LOG.value
    assert lab.config.transform == Transform.LOG
    lab.matrix_choice.value = Metric.PROBIOU.value
    assert "ProbIoU cost" in lab.matrix_axes.get_title()
    lab.x_scale.value = 0
    assert lab.x_scale.value == lab.config.x_scale == 100


def test_scene_management_and_baseline_reset(lab: MatcherLab) -> None:
    """Add/remove buttons preserve identities, handle empty scenes, and reset only the matcher."""
    lab.scene_choice.value = "Empty scene"
    assert lab.evaluation.total.shape == (0, 0)
    buttons = {button.description: button for button in lab.view.children[4].children}
    buttons["Add GT"].click()
    assert len(lab.scene.targets) == 1
    assert lab.assignment.unmatched_targets == (0,)
    buttons["Add prediction"].click()
    assert len(lab.scene.predictions) == 1
    assert lab.assignment.pairs == ((0, 0),)
    buttons["Remove selected"].click()
    assert not lab.scene.predictions
    assert lab.assignment.unmatched_targets == (0,)
    lab.preset_choice.value = "Class + ProbIoU"
    buttons["Reset baseline"].click()
    assert lab.config == baseline()
    assert len(lab.scene.targets) == 1


def test_fit_does_not_change_costs(lab: MatcherLab) -> None:
    """Viewport framing changes display units per pixel but never cost normalization."""
    original = lab.evaluation.total.copy()
    buttons = {button.description: button for button in lab.view.children[4].children}
    buttons["Fit scene"].click()
    np.testing.assert_array_equal(lab.evaluation.total, original)


def test_static_preview_is_vector_only() -> None:
    """The GitHub artifact contains vector shapes and no embedded raster or executable script."""
    svg = static_preview(scene_presets()["Neighboring text lines"], baseline())
    root = ElementTree.fromstring(svg)
    tags = {element.tag.rsplit("}", 1)[-1] for element in root.iter()}
    assert "svg" in tags
    assert "image" not in tags
    assert "script" not in tags
    assert len(svg.encode()) < 100_000


def test_invalid_evaluation_clears_stale_assignment(
    lab: MatcherLab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected computation must not leave an old matrix attached to newly edited boxes."""

    def reject_costs(*args: object) -> None:
        """Represent a numerical failure at the evaluation boundary."""
        raise ValueError("Matching costs must be finite")

    with monkeypatch.context() as context:
        context.setattr("ml_sandbox.matcher_lab.widgets.evaluate_costs", reject_costs)
        lab.refresh()
        assert "Invalid input" in lab.status.value
        assert not lab.matrix_axes.collections
        lab.show_arrows.value = False
        assert "Invalid input" in lab.status.value
    lab.refresh()
    assert isinstance(lab.evaluation_error, Absent)
    assert lab.matrix_axes.collections


def test_saved_lab_has_only_static_evidence() -> None:
    """Check the actual saved artifact, including widget metadata and SVG internals."""
    from pathlib import Path

    import nbformat
    from nbformat.notebooknode import NotebookNode

    path = Path(__file__).resolve().parents[3] / "06_hungarian_matcher/matcher_lab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    assert isinstance(notebook, NotebookNode)
    assert "widgets" not in notebook.metadata
    assert path.stat().st_size < 512 * 1024
    svg_count = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error"
            for mime, value in output.get("data", {}).items():
                assert mime in {"text/plain", "text/html", "image/svg+xml"}
                if mime == "image/svg+xml":
                    svg_count += 1
                    root = ElementTree.fromstring(value)
                    assert all(
                        element.tag.rsplit("}", 1)[-1] not in {"image", "script"}
                        for element in root.iter()
                    )
    assert svg_count >= 2


def test_hausdorff_preset_sampling_and_inspection(lab: MatcherLab) -> None:
    """The preset and sampling selector update costs, matrix, and pair diagnostics immediately."""
    lab.preset_choice.value = "Geometry only: Hausdorff"
    assert [term.metric for term in lab.config.terms] == [Metric.HAUSDORFF]
    assert lab.hausdorff_points.value == lab.config.hausdorff_points == 4
    lab.hausdorff_points.value = 32
    assert lab.config.hausdorff_points == 32
    assert "32 boundary points" in lab.formula.value
    lab.matrix_choice.value = Metric.HAUSDORFF.value
    assert "Hausdorff cost" in lab.matrix_axes.get_title()
    assert "Hausdorff" in lab.details.value
    expected = lab.evaluation.total.copy()
    lab.transform.value = Transform.LOG.value
    np.testing.assert_array_equal(lab.evaluation.total, expected)
    buttons = {button.description: button for button in lab.view.children[4].children}
    buttons["Reset baseline"].click()
    assert lab.hausdorff_points.value == lab.config.hausdorff_points == 4
