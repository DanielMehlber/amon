"""Browser-based reporting interface built with Panel (white theme).

Layout: a session selector at the top, below it three tabs -

- **Events**: filterable chronological event table plus a detail view with
  the evidence GIF, the intensity/threshold plot and detector metadata.
- **Calibration**: the annotated calibration GIF (feature points, HUD
  markers, texts, blink frequencies) and the calibrated thresholds.
- **Export**: one-click export to the formats registered in
  :mod:`amon.exporters`.

All widgets are Panel components; no HTML is written by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import panel as pn

from amon.db import Database
from amon.plots import intensity_figure

pn.extension(design="material", theme="default")


def _session_label(session: dict) -> str:
    status = "" if session["status"] == "completed" else f" [{session['status']}]"
    return f"{session['name']} - {session['id']}{status}"


def _events_frame(events: list) -> pd.DataFrame:
    columns = ["id", "anomaly_id", "detector", "start", "end", "duration", "max_intensity"]
    frame = pd.DataFrame([{c: e[c] for c in columns} for e in events], columns=columns)
    return frame.round(2)


def event_detail(event: dict) -> pn.Column:
    """Detail view for one event: media, plot, metadata."""
    summary = pn.pane.Markdown(
        f"### {event['anomaly_id']}\n"
        f"| | |\n|---|---|\n"
        f"| Detector | {event['detector']} |\n"
        f"| Start | {event['start']:.2f} s |\n"
        f"| End | {event['end']:.2f} s |\n"
        f"| Duration | {event['duration']:.2f} s |\n"
        f"| Peak intensity | {event['max_intensity']:.3f} |\n"
        f"| Threshold | {event['threshold']:.3f} |\n"
    )
    parts = [summary]
    if event["media"] and Path(event["media"]).exists():
        parts.append(pn.pane.Image(event["media"], width=480))
    parts.append(pn.pane.Bokeh(intensity_figure(event)))
    if event["metadata"]:
        parts.append(pn.pane.Markdown("**Detector metadata**"))
        parts.append(pn.pane.JSON(event["metadata"], depth=2))
    if event["regions"]:
        parts.append(pn.pane.Markdown(f"**Highlighted regions:** {event['regions']}"))
    return pn.Column(*parts, sizing_mode="stretch_width")


def events_tab(events: list) -> pn.Column:
    """Chronological, filterable event list with a detail pane."""
    types = sorted({e["anomaly_id"] for e in events})
    type_filter = pn.widgets.MultiChoice(name="Anomaly types", options=types, value=[])
    min_duration = pn.widgets.FloatSlider(name="Min duration (s)", start=0.0,
                                          end=max([e["duration"] for e in events] + [1.0]), step=0.1)
    selector = pn.widgets.Select(name="Event", options={})

    def filtered(selected_types, duration):
        chosen = [
            e for e in events
            if (not selected_types or e["anomaly_id"] in selected_types)
            and e["duration"] >= duration
        ]
        selector.options = {
            f"#{e['id']} {e['anomaly_id']} @ {e['start']:.1f}s": e["id"] for e in chosen
        }
        if chosen:
            selector.value = chosen[0]["id"]
        return pn.pane.DataFrame(_events_frame(chosen), index=False, sizing_mode="stretch_width")

    def detail(event_id):
        event = next((e for e in events if e["id"] == event_id), None)
        return event_detail(event) if event else pn.pane.Markdown("*No event selected.*")

    return pn.Column(
        pn.Row(type_filter, min_duration),
        pn.bind(filtered, type_filter, min_duration),
        pn.layout.Divider(),
        selector,
        pn.bind(detail, selector),
        sizing_mode="stretch_width",
    )


def calibration_tab(calibration: dict) -> pn.Column:
    """Calibration review: annotated media, HUD table, thresholds."""
    if calibration is None:
        return pn.Column(pn.pane.Markdown("*No calibration recorded yet.*"))
    parts = [pn.pane.Markdown("### Calibration review")]
    if calibration["media"] and Path(calibration["media"]).exists():
        parts.append(pn.pane.Image(calibration["media"], width=480))
    elements = calibration["annotations"].get("hud_elements", [])
    if elements:
        parts.append(pn.pane.Markdown("**HUD elements**"))
        parts.append(pn.pane.DataFrame(pd.DataFrame(elements), index=False))
    keypoints = calibration["annotations"].get("keypoints", [])
    parts.append(pn.pane.Markdown(f"**Stable feature points:** {len(keypoints)}"))
    parts.append(pn.pane.Markdown("**Calibrated thresholds**"))
    parts.append(pn.pane.JSON(calibration["thresholds"], depth=3))
    return pn.Column(*parts, sizing_mode="stretch_width")


def export_tab(config: dict, session_id: str) -> pn.Column:
    from amon.exporters import EXPORTERS, export_session

    format_select = pn.widgets.Select(name="Format", options=list(EXPORTERS))
    button = pn.widgets.Button(name="Export report", button_type="primary")
    status = pn.pane.Markdown("")

    def run_export(_):
        path = export_session(config, session_id, format_select.value)
        status.object = f"Exported to `{path}`"

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

    header = pn.pane.Markdown(
        f"## Session {session['name']}\n"
        f"Status: **{session['status']}** - {len(events)} events - source `{session['source']}`"
    )
    body = pn.Tabs(
        ("Events", events_tab(events) if events else pn.pane.Markdown("*No events recorded.*")),
        ("Calibration", calibration_tab(calibration)),
        ("Export", export_tab(config, session_id)),
    )
    return pn.Column(header, body, sizing_mode="stretch_width")


def build_app(config: dict):
    """Assemble the report application (a Panel template)."""
    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    try:
        sessions = db.list_sessions()
    finally:
        db.close()

    template = pn.template.MaterialTemplate(title="amon monitoring reports", theme="default")
    if not sessions:
        template.main.append(pn.pane.Markdown("## No monitoring sessions found."))
        return template

    options = {_session_label(s): s["id"] for s in sessions}
    session_select = pn.widgets.Select(name="Monitoring session", options=options)
    template.main.append(pn.Column(
        session_select,
        pn.bind(lambda sid: session_view(config, sid), session_select),
        sizing_mode="stretch_width",
    ))
    return template


def serve(config: dict) -> None:
    """Launch the report UI in the browser (blocks until stopped)."""
    port = int(config["report"]["port"])
    pn.serve(lambda: build_app(config), port=port, show=True)
