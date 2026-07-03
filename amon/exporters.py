"""Report export.

Exporters are looked up by format name in :data:`EXPORTERS`; adding a new
format means implementing :class:`Exporter` and registering the class -
the CLI and report UI pick it up automatically.

The bundled :class:`HtmlArchiveExporter` produces a single standalone HTML
file: all media is base64-embedded and Bokeh resources are inlined, so the
archive can be emailed or opened offline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

import pandas as pd
import panel as pn
from bokeh.resources import INLINE

from amon.db import Database
from amon.plots import intensity_figure

pn.extension()


class Exporter(ABC):
    """Interface for report exporters."""

    #: Format name used in the config, CLI and UI.
    format: str = ""
    #: File suffix of the produced artifact.
    suffix: str = ""

    @abstractmethod
    def export(self, session: dict, events: list, calibration: Optional[dict], out_path: Path) -> Path:
        """Write the report for one session to ``out_path``."""


class HtmlArchiveExporter(Exporter):
    """Standalone offline HTML archive with embedded media and plots."""

    format = "html"
    suffix = ".html"

    def export(self, session: dict, events: list, calibration: Optional[dict], out_path: Path) -> Path:
        parts = [pn.pane.Markdown(
            f"# Monitoring report: {session['name']}\n"
            f"Session `{session['id']}` - status **{session['status']}** - "
            f"{len(events)} events - source `{session['source']}`"
        )]
        parts += self._calibration_section(calibration)
        parts += self._events_section(events)
        report = pn.Column(*parts, width=760)
        report.save(str(out_path), resources=INLINE, embed=True, title=f"amon report {session['id']}")
        return out_path

    def _calibration_section(self, calibration: Optional[dict]) -> list:
        if calibration is None:
            return [pn.pane.Markdown("## Calibration\n*Not recorded.*")]
        parts = [pn.pane.Markdown("## Calibration")]
        if calibration["media"] and Path(calibration["media"]).exists():
            parts.append(pn.pane.Image(calibration["media"], width=480, embed=True))
        elements = calibration["annotations"].get("hud_elements", [])
        if elements:
            parts.append(pn.pane.DataFrame(pd.DataFrame(elements), index=False))
        parts.append(pn.pane.JSON(calibration["thresholds"], depth=3, name="thresholds"))
        return parts

    def _events_section(self, events: list) -> list:
        parts = [pn.pane.Markdown(f"## Events ({len(events)})")]
        for event in events:
            parts.append(pn.pane.Markdown(
                f"### {event['anomaly_id']}\n"
                f"{event['start']:.2f}s - {event['end']:.2f}s "
                f"({event['duration']:.2f}s), peak {event['max_intensity']:.3f} "
                f"vs threshold {event['threshold']:.3f}"
            ))
            if event["media"] and Path(event["media"]).exists():
                parts.append(pn.pane.Image(event["media"], width=480, embed=True))
            parts.append(pn.pane.Bokeh(intensity_figure(event)))
            if event["metadata"]:
                parts.append(pn.pane.JSON(event["metadata"], depth=2))
        return parts


EXPORTERS: Dict[str, Type[Exporter]] = {
    HtmlArchiveExporter.format: HtmlArchiveExporter,
}


def export_session(config: dict, session_id: str, fmt: Optional[str] = None,
                   out_path: Optional[str] = None) -> Path:
    """Export one session's report; returns the written file path."""
    fmt = fmt or config["export"]["format"]
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown export format '{fmt}' (available: {', '.join(EXPORTERS)})")
    exporter = EXPORTERS[fmt]()

    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    try:
        session = db.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session '{session_id}'")
        events = db.list_events(session_id)
        calibration = db.get_calibration(session_id)
    finally:
        db.close()

    if out_path is None:
        out_dir = Path(config["data_dir"]) / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"report-{session_id}{exporter.suffix}"
    return exporter.export(session, events, calibration, Path(out_path))
