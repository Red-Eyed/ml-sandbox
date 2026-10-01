"""Build compact Altair visualizations for notebook experiments."""

from collections.abc import Mapping, Sequence

import altair as alt
import vl_convert as vlc


def chart_to_svg(chart: alt.Chart) -> str:
    """Return standalone SVG markup for static notebook output.

    The conversion runs locally and embeds no executable JavaScript. Conversion
    failures from Vega-Lite are propagated to the caller.
    """
    return vlc.vegalite_to_svg(chart.to_dict())


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    color: str = "#2563eb",
) -> alt.Chart:
    """Return a labeled horizontal bar chart for non-negative values."""
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    if any(value < 0 for value in values):
        raise ValueError("bar chart values must be non-negative")

    data = [
        {"label": label, "value": float(value), "order": order}
        for order, (label, value) in enumerate(zip(labels, values, strict=True))
    ]
    return (
        alt.Chart(alt.Data(values=data))
        .mark_bar(color=color)
        .encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("label:N", sort=alt.EncodingSortField(field="order"), title=None),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=".6g")],
        )
        .properties(title=title, width=520, height=alt.Step(28))
    )


def grouped_bar_chart(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    y_title: str,
) -> alt.Chart:
    """Return grouped bars that compare named series within each category."""
    data = [
        {"category": category, "value": float(value), "series": name}
        for name, values in series.items()
        for category, value in zip(categories, values, strict=True)
    ]
    return (
        alt.Chart(alt.Data(values=data))
        .mark_bar()
        .encode(
            x=alt.X("category:N", sort=list(categories), title=None),
            xOffset="series:N",
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("category:N"),
                alt.Tooltip("series:N"),
                alt.Tooltip("value:Q", format=".6g"),
            ],
        )
        .properties(title=title, width=600, height=300)
    )
