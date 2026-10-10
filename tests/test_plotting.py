"""Tests for notebook presentation helpers."""

from ml_sandbox.plotting import bar_chart, chart_to_svg, grouped_bar_chart, line_chart


def test_chart_to_svg_returns_standalone_static_markup() -> None:
    """Render a chart as SVG without adding an executable script."""
    chart = bar_chart(["gradient"], [1.5], title="Parameter state")

    svg = chart_to_svg(chart)

    assert svg.lstrip().startswith("<svg")
    assert "Parameter state" in svg
    assert "<script" not in svg


def test_grouped_bar_chart_preserves_category_order() -> None:
    """Keep conceptual category and series order instead of sorting alphabetically."""
    chart = grouped_bar_chart(
        ["weight[0]", "weight[1]", "bias[0]"],
        {
            "current gradient": [1.0, -1.0, 0.5],
            "remembered state": [0.5, -0.5, 0.25],
        },
        title="Gradient",
        y_title="Value",
    )

    encoding = chart.to_dict()["encoding"]
    assert encoding["x"]["sort"] == ["weight[0]", "weight[1]", "bias[0]"]
    assert encoding["xOffset"]["sort"] == ["current gradient", "remembered state"]
    assert encoding["color"]["sort"] == ["current gradient", "remembered state"]


def test_line_chart_preserves_series_order() -> None:
    """Keep line colors and legend entries in the caller's teaching order."""
    chart = line_chart(
        [0, 1],
        {"parameter": [2.0, 1.9], "first moment": [0.0, 0.05]},
        title="State",
        x_title="Step",
        y_title="Value",
    )

    assert chart.to_dict()["encoding"]["color"]["sort"] == ["parameter", "first moment"]
