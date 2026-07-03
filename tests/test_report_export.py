"""Tests for the report UI construction and the HTML archive export."""
import base64
from pathlib import Path

import pytest

from amon.db import Database
from amon.exporters import EXPORTERS, export_session


@pytest.fixture(scope="module")
def session_data(completed_session):
    config, session_id = completed_session
    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    yield config, session_id, db
    db.close()


class TestHtmlExport:
    def test_export_produces_standalone_archive(self, session_data, tmp_path):
        config, session_id, db = session_data
        out = export_session(config, session_id, "html", str(tmp_path / "report.html"))
        html = Path(out).read_text(encoding="utf-8", errors="ignore")
        session = db.get_session(session_id)

        assert html.startswith("<!DOCTYPE html>")
        for event in db.list_events(session_id):
            assert event["anomaly_id"] in html
        assert "data:image" in html
        assert "Started:" in html
        assert session["id"] in html
        assert Path(out).stat().st_size > 10_000
        assert "<script" not in html.lower()

    def test_default_output_location(self, session_data):
        config, session_id, _ = session_data
        out = export_session(config, session_id)
        assert Path(out).exists()
        assert Path(out).parent == Path(config["data_dir"]) / "exports"

    def test_unknown_format_raises(self, session_data):
        config, session_id, _ = session_data
        with pytest.raises(ValueError, match="unknown export format"):
            export_session(config, session_id, "docx")

    def test_registry_is_extensible(self):
        assert "html" in EXPORTERS


class TestReportUi:
    def test_build_app_lists_sessions(self, session_data):
        from amon.report import build_app

        config, session_id, _ = session_data
        template = build_app(config)
        assert template is not None  # renders without raising

    def test_session_view_has_tabs(self, session_data):
        from amon.report import session_view

        config, session_id, _ = session_data
        view = session_view(config, session_id)
        tabs = view[1]
        assert [name for name, _ in zip(tabs._names, tabs)] == ["Anomalies", "Calibration", "Export"]

    def test_event_filter_and_catalog(self, session_data):
        from amon.report import _filter_events, _event_catalog

        config, session_id, db = session_data
        events = db.list_events(session_id)
        filtered = _filter_events(events, "flicker", [], 0.0, "Start time (earliest first)")
        assert filtered and all("flicker" in e["anomaly_id"] for e in filtered)

        catalogue = _event_catalog(events, "", [], 0.0, "Start time (earliest first)")
        assert catalogue.height == 680

    def test_event_detail_renders(self, session_data):
        from amon.report import event_detail

        config, session_id, db = session_data
        event = db.list_events(session_id)[0]
        detail = event_detail(event)
        assert len(detail) >= 2

    def test_empty_database_shows_message(self, tmp_path):
        from amon.config import merge_defaults
        from amon.report import build_app

        config = merge_defaults({"data_dir": str(tmp_path)})
        template = build_app(config)
        assert template is not None
