"""Report export.

Exporters are looked up by format name in :data:`EXPORTERS`; adding a new
format means implementing :class:`Exporter` and registering the class -
the CLI and report UI pick it up automatically.

The bundled :class:`HtmlArchiveExporter` writes a lightweight standalone
HTML file with base64-embedded media and static intensity plots — no
Panel/Bokeh runtime required to open it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

from amon.db import Database
from amon.html_export import write_html_report


class Exporter(ABC):
    """Interface for report exporters."""

    #: Format name used in the config, CLI and UI.
    format: str = ""
    #: File suffix of the produced artifact.
    suffix: str = ""

    @abstractmethod
    def export(
        self, session: dict, events: list, calibration: Optional[dict], out_path: Path
    ) -> Path:
        """Write the report for one session to ``out_path``."""


class HtmlArchiveExporter(Exporter):
    """Standalone offline HTML archive with embedded media and plots."""

    format = "html"
    suffix = ".html"

    def export(
        self, session: dict, events: list, calibration: Optional[dict], out_path: Path
    ) -> Path:
        return write_html_report(session, events, calibration, out_path)


EXPORTERS: Dict[str, Type[Exporter]] = {
    HtmlArchiveExporter.format: HtmlArchiveExporter,
}


def export_session(
    config: dict,
    session_id: str,
    fmt: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Path:
    """Export one session's report; returns the written file path."""
    fmt = fmt or config["export"]["format"]
    if fmt not in EXPORTERS:
        raise ValueError(
            f"unknown export format '{fmt}' (available: {', '.join(EXPORTERS)})"
        )
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
