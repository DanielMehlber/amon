"""Bokeh figures shared by the report UI and the exporters."""
from __future__ import annotations

from bokeh.models import Span
from bokeh.plotting import figure

ACCENT = "#c62828"
ACCENT_MUTED = "#9e9e9e"


def intensity_figure(event: dict, width: int = 520, height: int = 240):
    """Line plot of an event's intensity timeline with its threshold."""
    fig = figure(
        width=width, height=height, title=event["anomaly_id"],
        x_axis_label="session time (s)", y_axis_label="intensity",
        toolbar_location=None, background_fill_color="white",
    )
    timeline = event["timeline"] or [[event["start"], event["max_intensity"]]]
    xs = [point[0] for point in timeline]
    ys = [point[1] for point in timeline]
    fig.line(xs, ys, line_width=2, color=ACCENT, legend_label="intensity")
    threshold = Span(
        location=event["threshold"], dimension="width",
        line_color=ACCENT_MUTED, line_dash="dashed", line_width=2,
    )
    fig.add_layout(threshold)
    fig.line([], [], color=ACCENT_MUTED, line_dash="dashed", legend_label="threshold")
    fig.legend.location = "top_right"
    fig.legend.background_fill_alpha = 0.6
    return fig
