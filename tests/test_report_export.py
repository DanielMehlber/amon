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
        assert [name for name, _ in zip(tabs._names, tabs)] == [
            "Anomalies",
            "Calibration",
            "Export",
        ]

    def test_event_filter_and_catalog(self, session_data):
        from amon.report import _filter_events, _event_catalog

        config, session_id, db = session_data
        events = db.list_events(session_id)
        filtered = _filter_events(
            events, "flicker", [], 0.0, "Start time (earliest first)"
        )
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


class TestOfflinePanel:
    def test_websocket_origins_for_loopback(self):
        from amon.panel_offline import websocket_origins

        assert set(websocket_origins("127.0.0.1", 5006)) == {
            "localhost:5006",
            "127.0.0.1:5006",
        }
        assert set(websocket_origins("localhost", 5006)) == {
            "localhost:5006",
            "127.0.0.1:5006",
        }
        assert websocket_origins("127.0.0.1", 5006, ["host:1234"]) == ["host:1234"]
        assert websocket_origins("0.0.0.0", 5006) == ["*"]

    def test_offline_config_forces_local_resources(self, monkeypatch):
        monkeypatch.setenv("BOKEH_RESOURCES", "cdn")

        import amon.panel_offline as panel_offline

        panel_offline._CONFIGURED = False
        panel_offline.configure_offline_panel({"offline": True})

        from bokeh.settings import settings
        from panel.theme.bootstrap import Bootstrap

        assert settings.resources() == "server"
        res = Bootstrap().resolve_resources(cdn=False)
        for url in list(res["css"].values()) + list(res["js"].values()):
            assert not url.startswith(("http://", "https://", "//"))
            assert "jsdelivr" not in url

    def test_served_html_has_no_jsdelivr(self, session_data, monkeypatch):
        import threading
        import time
        import urllib.request

        import panel as pn

        from amon.panel_offline import assert_offline_html, websocket_origins
        from amon.report import build_app

        monkeypatch.setenv("BOKEH_RESOURCES", "cdn")
        config, _, _ = session_data
        port = 5094
        address = config["report"].get("address", "127.0.0.1")
        threading.Thread(
            target=lambda: pn.serve(
                lambda: build_app(config),
                port=port,
                show=False,
                threaded=True,
                address=address,
                websocket_origin=websocket_origins(
                    address, port, config["report"].get("websocket_origin")
                ),
            ),
            daemon=True,
        ).start()
        time.sleep(4)
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert_offline_html(html)
        assert "jsdelivr" not in html
        assert "googleapis" not in html
        assert "cdn.bokeh.org" not in html
        assert "cdn.holoviz.org" not in html
