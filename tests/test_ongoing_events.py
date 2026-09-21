"""Tests for ongoing / completed event persistence."""

from amon.db import EVENT_STATUS_COMPLETED, EVENT_STATUS_ONGOING, Database
from amon.model import AnomalyEvent
from amon.report import _event_summary, _is_ongoing


def _event(aid: str = "hud/cam/text", start: float = 1.0, end: float = 1.5) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=aid,
        detector="hud",
        start=start,
        end=end,
        max_intensity=0.9,
        threshold=0.3,
        timeline=[(start, 0.9)],
        metadata={"text": "CAM01"},
        regions=[[1, 2, 3, 4]],
    )


class TestOngoingEvents:
    def test_upsert_then_complete(self, tmp_path):
        db = Database(tmp_path / "amon.sqlite")
        db.create_session("s1", "test", "file", 10.0)

        event = _event(end=1.2)
        row_id = db.upsert_ongoing_event("s1", event)
        rows = db.list_events("s1")
        assert len(rows) == 1
        assert rows[0]["status"] == EVENT_STATUS_ONGOING
        assert rows[0]["id"] == row_id
        assert rows[0]["media"] is None

        event.end = 4.0
        event.max_intensity = 1.2
        event.timeline.append((4.0, 1.2))
        completed_id = db.complete_event("s1", event, media="/tmp/x.gif")
        assert completed_id == row_id
        rows = db.list_events("s1")
        assert len(rows) == 1
        assert rows[0]["status"] == EVENT_STATUS_COMPLETED
        assert rows[0]["end"] == 4.0
        assert rows[0]["media"] == "/tmp/x.gif"
        db.close()

    def test_discard_removes_ongoing(self, tmp_path):
        db = Database(tmp_path / "amon.sqlite")
        db.create_session("s1", "test", "file", 10.0)
        db.upsert_ongoing_event("s1", _event())
        db.discard_ongoing_event("s1", "hud/cam/text")
        assert db.list_events("s1") == []
        db.close()

    def test_one_ongoing_per_anomaly(self, tmp_path):
        db = Database(tmp_path / "amon.sqlite")
        db.create_session("s1", "test", "file", 10.0)
        first = db.upsert_ongoing_event("s1", _event(end=1.0))
        second = db.upsert_ongoing_event("s1", _event(end=2.5))
        assert first == second
        rows = db.list_events("s1")
        assert len(rows) == 1
        assert rows[0]["end"] == 2.5
        db.close()

    def test_migrate_adds_status_to_legacy_db(self, tmp_path):
        path = tmp_path / "legacy.sqlite"
        import sqlite3

        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT, started_at REAL,
                finished_at REAL, source TEXT, fps REAL, status TEXT
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, anomaly_id TEXT, detector TEXT,
                start REAL, end REAL, duration REAL, max_intensity REAL,
                threshold REAL, timeline TEXT, metadata TEXT, regions TEXT, media TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES ('s1','n',1,NULL,'f',10,'running')"
        )
        conn.commit()
        conn.close()

        db = Database(path)
        assert "status" in {
            row[1] for row in db._conn.execute("PRAGMA table_info(events)")
        }
        db.close()


class TestOngoingReportLabels:
    def test_ongoing_summary_flag(self):
        event = {
            "anomaly_id": "hud/cam/text",
            "start": 10.0,
            "end": 12.0,
            "duration": 2.0,
            "max_intensity": 0.8,
            "status": "ongoing",
        }
        assert _is_ongoing(event)
        assert "ONGOING" in _event_summary(event)
