"""Browser-based reporting interface built with Panel (white theme).

Layout: a session selector at the top, below it three tabs -

- **Events**: filterable, scrollable catalogue of collapsible anomaly cards
  (summary visible when collapsed, full detail when expanded).
- **Calibration**: annotated calibration media, HUD element cards, thresholds.
- **Export**: one-click export to the formats registered in
  :mod:`amon.exporters`.

All widgets are Panel components; no HTML is written by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import panel as pn

from amon.db import Database
from amon.panel_offline import configure_offline_panel, websocket_origins
from amon.plots import intensity_figure
from amon.timefmt import format_wall_time

# Report chrome — red accent throughout the UI and plots.
ACCENT = "#c62828"
ACCENT_DARK = "#b71c1c"
ACCENT_LIGHT = "#ffebee"

REPORT_TITLE = "Video Stream Anomaly Detection"
SORT_OPTIONS = {
    "Start time (earliest first)": lambda e: e["start"],
    "Start time (latest first)": lambda e: -e["start"],
    "Duration (longest first)": lambda e: -e["duration"],
    "Peak intensity (highest first)": lambda e: -e["max_intensity"],
}


def _session_label(session: dict) -> str:
    status = "" if session["status"] == "completed" else f" [{session['status']}]"
    started = format_wall_time(session["started_at"])
    name = session["name"]
    if name and name != session["id"]:
        return f"{session['id']} ({name}) · {started}{status}"
    return f"{session['id']} · {started}{status}"


def _filter_events(
    events: List[dict],
    search: str,
    selected_types: List[str],
    min_duration: float,
    sort_key: str,
) -> List[dict]:
    """Apply search, type and duration filters, then sort."""
    query = (search or "").strip().lower()
    chosen = []
    for event in events:
        if selected_types and event["anomaly_id"] not in selected_types:
            continue
        if event["duration"] < min_duration:
            continue
        if (
            query
            and query not in event["anomaly_id"].lower()
            and query not in event["detector"].lower()
        ):
            continue
        chosen.append(event)
    chosen.sort(
        key=SORT_OPTIONS.get(sort_key, SORT_OPTIONS["Start time (earliest first)"])
    )
    return chosen


def _json_block(data) -> pn.pane.Markdown:
    """Render JSON as a fenced code block (no jsoneditor CDN dependency)."""
    text = json.dumps(data, indent=2, sort_keys=True, default=str)
    return pn.pane.Markdown(f"```json\n{text}\n```")


def _event_summary(event: dict) -> str:
    """One-line label for a collapsed catalogue card."""
    return (
        f"{event['anomaly_id']}  ·  "
        f"{event['start']:.1f}s → {event['end']:.1f}s  ·  "
        f"{event['duration']:.1f}s  ·  "
        f"peak {event['max_intensity']:.2f}"
    )


def event_detail(event: dict) -> pn.Column:
    """Expanded card body: media, stats, intensity plot, metadata."""
    stats = pn.pane.Markdown(
        f"#### {event['anomaly_id']}\n\n"
        f"- **Detector:** `{event['detector']}`\n"
        f"- **Start:** {event['start']:.2f} s\n"
        f"- **End:** {event['end']:.2f} s\n"
        f"- **Duration:** {event['duration']:.2f} s\n"
        f"- **Peak intensity:** {event['max_intensity']:.3f}\n"
        f"- **Threshold:** {event['threshold']:.3f}\n",
        styles={"padding": "0.5rem 1rem"},
    )

    media_pane = None
    if event["media"] and Path(event["media"]).exists():
        media_pane = pn.pane.Image(
            event["media"],
            width=400,
            styles={"border": f"2px solid {ACCENT}", "border-radius": "6px"},
        )

    header = pn.Row(
        pn.Column(media_pane or pn.Spacer(width=0), width=420, margin=(0, 16, 0, 0)),
        pn.Column(stats, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    parts = [header, pn.pane.Bokeh(intensity_figure(event, width=680))]
    if event["metadata"]:
        parts.append(pn.pane.Markdown("**Detector metadata**"))
        parts.append(_json_block(event["metadata"]))
    if event["regions"]:
        parts.append(pn.pane.Markdown(f"**Highlighted regions:** `{event['regions']}`"))
    return pn.Column(*parts, sizing_mode="stretch_width", margin=(8, 0))


def _event_catalog(
    events: List[dict],
    search: str,
    selected_types: List[str],
    min_duration: float,
    sort_key: str,
) -> pn.Column:
    """Scrollable accordion catalogue, shop-style expandable cards."""
    chosen = _filter_events(events, search, selected_types, min_duration, sort_key)
    if not chosen:
        return pn.Column(
            pn.pane.Markdown("*No anomalies match the current filters.*"),
            sizing_mode="stretch_width",
        )

    cards = [(_event_summary(event), event_detail(event)) for event in chosen]
    catalogue = pn.Accordion(
        *cards,
        active=[],
        toggle=True,
        sizing_mode="stretch_width",
        stylesheets=[
            f"""
            :host .accordion-button {{
                font-weight: 600;
            }}
            :host .accordion-button:not(.collapsed) {{
                color: {ACCENT_DARK};
                background-color: {ACCENT_LIGHT};
            }}
        """
        ],
    )
    return pn.Column(
        pn.pane.Markdown(
            f"**{len(chosen)}** of **{len(events)}** anomalies  \n"
            "*Expand a card to see evidence, intensity plot and metadata.*"
        ),
        catalogue,
        scroll=True,
        height=680,
        sizing_mode="stretch_width",
        styles={
            "border": f"1px solid {ACCENT_LIGHT}",
            "border-radius": "8px",
            "padding": "12px",
            "background": "#fafafa",
        },
    )


def events_tab(events: list) -> pn.Column:
    """Filterable, scrollable catalogue of collapsible anomaly cards."""
    types = sorted({e["anomaly_id"] for e in events})
    max_duration = max([e["duration"] for e in events] + [1.0])

    search = pn.widgets.TextInput(
        name="Search",
        placeholder="Filter by anomaly or detector…",
        sizing_mode="stretch_width",
    )
    type_filter = pn.widgets.MultiChoice(
        name="Anomaly type",
        options=types,
        value=[],
        sizing_mode="stretch_width",
    )
    min_duration = pn.widgets.FloatSlider(
        name="Min duration (s)",
        start=0.0,
        end=max_duration,
        step=0.1,
        value=0.0,
    )
    sort_by = pn.widgets.Select(
        name="Sort by", options=list(SORT_OPTIONS), value="Start time (earliest first)"
    )

    filters = pn.Row(
        pn.Column(search, sizing_mode="stretch_width"),
        pn.Column(type_filter, sizing_mode="stretch_width"),
        pn.Column(min_duration, width=280),
        pn.Column(sort_by, width=280),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        filters,
        pn.bind(_event_catalog, events, search, type_filter, min_duration, sort_by),
        sizing_mode="stretch_width",
    )


def _hud_element_card(element: dict) -> pn.Column:
    """Single HUD element as a compact info card."""
    blink = element.get("blink_hz", 0)
    blink_line = f" · blinks at **{blink:g} Hz**" if blink else " · static"
    box = element.get("box", [])
    return pn.Column(
        pn.pane.Markdown(
            f"**{element.get('text') or element['id']}** (`{element['id']}`){blink_line}  \n"
            f"Bounding box: `{box}` · on-screen **{element.get('on_ratio', 0):.0%}** of calibration",
        ),
        styles={
            "border": f"1px solid {ACCENT_LIGHT}",
            "border-left": f"4px solid {ACCENT}",
            "border-radius": "6px",
            "padding": "10px 14px",
            "margin-bottom": "8px",
            "background": "white",
        },
        sizing_mode="stretch_width",
    )


def calibration_tab(calibration: dict) -> pn.Column:
    """Calibration review: media, HUD cards, feature-point count, thresholds."""
    if calibration is None:
        return pn.Column(pn.pane.Markdown("*No calibration recorded yet.*"))

    parts = [pn.pane.Markdown("### Calibration review")]
    if calibration["media"] and Path(calibration["media"]).exists():
        parts.append(
            pn.pane.Image(
                calibration["media"],
                width=520,
                styles={
                    "border": f"2px solid {ACCENT}",
                    "border-radius": "6px",
                    "display": "block",
                    "margin": "0 auto",
                },
            )
        )

    elements = calibration["annotations"].get("hud_elements", [])
    if elements:
        parts.append(pn.pane.Markdown("#### HUD elements"))
        parts.extend(_hud_element_card(element) for element in elements)

    keypoints = calibration["annotations"].get("keypoints", [])
    parts.append(
        pn.pane.Markdown(
            f"#### Feature points\n**{len(keypoints)}** stable corners tracked for spatial detection."
        )
    )

    parts.append(pn.pane.Markdown("#### Calibrated thresholds"))
    parts.append(_json_block(calibration["thresholds"]))
    return pn.Column(*parts, sizing_mode="stretch_width")


def export_tab(config: dict, session_id: str) -> pn.Column:
    from amon.exporters import EXPORTERS, export_session

    format_select = pn.widgets.Select(name="Format", options=list(EXPORTERS))
    button = pn.widgets.Button(name="Export report", button_type="primary")
    status = pn.pane.Markdown("")

    def run_export(_):
        try:
            path = export_session(config, session_id, format_select.value)
            status.object = f"Exported to `{path}`"
        except Exception as exc:
            status.object = f"**Export failed:** {exc}"

    button.on_click(run_export)
    return pn.Column(format_select, button, status)


def session_view(config: dict, session_id: str) -> pn.Column:
    """The full report for a single session."""
    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    try:
        session = db.get_session(session_id)
        events = db.list_events(session_id)
        calibration = db.get_calibration(session_id)
    finally:
        db.close()

    status_color = ACCENT if session["status"] == "running" else "#2e7d32"
    finished = ""
    if session.get("finished_at"):
        finished = f" · finished {format_wall_time(session['finished_at'])}"
    header = pn.pane.Markdown(
        f"## {session['id']}\n"
        f"**Started** {format_wall_time(session['started_at'])}{finished}  \n"
        f"<span style='color:{status_color};font-weight:600'>{session['status'].upper()}</span>"
        f" · **{len(events)}** anomalies · source `{session['source']}`"
        + (
            f" · config `{session['name']}`"
            if session.get("name") and session["name"] != session["id"]
            else ""
        ),
    )
    body = pn.Tabs(
        (
            "Anomalies",
            (
                events_tab(events)
                if events
                else pn.pane.Markdown("*No anomalies recorded.*")
            ),
        ),
        ("Calibration", calibration_tab(calibration)),
        ("Export", export_tab(config, session_id)),
        tabs_location="above",
        sizing_mode="stretch_width",
    )
    return pn.Column(header, body, sizing_mode="stretch_width")


def build_app(config: dict):
    """Assemble the report application (a Panel template)."""
    configure_offline_panel(config.get("report"))
    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    try:
        sessions = db.list_sessions()
    finally:
        db.close()

    template = pn.template.BootstrapTemplate(
        title=REPORT_TITLE,
        theme="default",
        header_background=ACCENT_DARK,
    )
    template.config.raw_css.append(
        f"""
        .bk-btn.bk-btn-primary {{
            background-color: {ACCENT};
            border-color: {ACCENT_DARK};
        }}
        .bk-btn.bk-btn-primary:hover {{
            background-color: {ACCENT_DARK};
        }}
    """
    )
    if not sessions:
        template.main.append(pn.pane.Markdown("## No monitoring sessions found."))
        return template

    options = {_session_label(s): s["id"] for s in sessions}
    session_select = pn.widgets.Select(
        name="Monitoring session", options=options, width=400
    )
    template.main.append(
        pn.Column(
            pn.Row(session_select, sizing_mode="stretch_width"),
            pn.bind(lambda sid: session_view(config, sid), session_select),
            sizing_mode="stretch_width",
        )
    )
    return template


def serve(config: dict) -> None:
    """Launch the report UI in the browser (blocks until stopped)."""
    report = config["report"]
    configure_offline_panel(report)
    port = int(report["port"])
    address = report.get("address", "127.0.0.1")
    pn.serve(
        lambda: build_app(config),
        port=port,
        address=address,
        show=True,
        websocket_origin=websocket_origins(
            address, port, report.get("websocket_origin")
        ),
    )
