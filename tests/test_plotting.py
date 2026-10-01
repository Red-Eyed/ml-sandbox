"""Tests for notebook presentation helpers."""

from ml_sandbox.plotting import bar_chart, chart_to_svg, grouped_bar_chart


def test_chart_to_svg_returns_standalone_static_markup() -> None:
    """Render a chart as SVG without adding an executable script."""
    chart = bar_chart(["gradient"], [1.5], title="Parameter state")

    svg = chart_to_svg(chart)

    assert svg.lstrip().startswith("<svg")
    assert "Parameter state" in svg
    assert "<script" not in svg


def test_grouped_bar_chart_preserves_category_order() -> None:
    """Keep the caller's conceptual order instead of sorting labels alphabetically."""
    chart = grouped_bar_chart(
        ["weight[0]", "weight[1]", "bias[0]"],
        {"gradient": [1.0, -1.0, 0.5]},
        title="Gradient",
        y_title="Value",
    )

    assert chart.to_dict()["encoding"]["x"]["sort"] == ["weight[0]", "weight[1]", "bias[0]"]
