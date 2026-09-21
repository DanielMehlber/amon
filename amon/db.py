"""SQLite-backed persistence for sessions, calibrations and events.

A single file database (``<data_dir>/amon.sqlite``) holds all monitoring
sessions.  Media files (GIFs) live next to it and are referenced by
relative paths, so the whole data directory can be archived or moved.
Timestamps of events are stored in seconds relative to the session start;
the session row carries the absolute wall-clock start time.

Events may be ``ongoing`` (still above threshold) or ``completed``.  The
report UI surfaces ongoing rows while a session is running so a refresh
shows anomalies that have not closed yet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Union

from amon.model import AnomalyEvent

EVENT_STATUS_ONGOING = "ongoing"
EVENT_STATUS_COMPLETED = "completed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    source TEXT,
    fps REAL,
    status TEXT NOT NULL DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS calibrations (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    completed_at REAL NOT NULL,
    thresholds TEXT NOT NULL,
    annotations TEXT NOT NULL,
    media TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    anomaly_id TEXT NOT NULL,
    detector TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    duration REAL NOT NULL,
    max_intensity REAL NOT NULL,
    threshold REAL NOT NULL,
    timeline TEXT NOT NULL,
    metadata TEXT NOT NULL,
    regions TEXT NOT NULL,
    media TEXT,
    status TEXT NOT NULL DEFAULT 'completed'
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, start);
"""


class Database:
    """Thin convenience wrapper around the SQLite schema above."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns / indexes introduced after the initial schema."""
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "status" not in columns:
            self._conn.execute(
                "ALTER TABLE events ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
            )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_ongoing "
            "ON events(session_id, anomaly_id) WHERE status = 'ongoing'"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- sessions -----------------------------------------------------------
    def create_session(
        self, session_id: str, name: str, source: str, fps: float
    ) -> None:
        self._conn.execute(
            "INSERT INTO sessions (id, name, started_at, source, fps) VALUES (?, ?, ?, ?, ?)",
            (session_id, name, time.time(), source, fps),
        )
        self._conn.commit()

    def finish_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET status = 'completed', finished_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    def list_sessions(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- calibration -----------------------------------------------------------
    def save_calibration(
        self,
        session_id: str,
        thresholds: dict,
        annotations: dict,
        media: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO calibrations VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                time.time(),
                json.dumps(thresholds),
                json.dumps(annotations),
                media,
            ),
        )
        self._conn.commit()

    def get_calibration(self, session_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM calibrations WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["thresholds"] = json.loads(data["thresholds"])
        data["annotations"] = json.loads(data["annotations"])
        return data

    # --- events -------------------------------------------------------------
    def _event_values(
        self,
        session_id: str,
        event: AnomalyEvent,
        media: Optional[str],
        status: str,
    ) -> tuple:
        return (
            session_id,
            event.anomaly_id,
            event.detector,
            event.start,
            event.end,
            event.duration,
            event.max_intensity,
            event.threshold,
            json.dumps(event.timeline),
            json.dumps(event.metadata),
            json.dumps(event.regions),
            media,
            status,
        )

    def upsert_ongoing_event(
        self, session_id: str, event: AnomalyEvent
    ) -> int:
        """Insert or refresh the single ongoing row for ``event.anomaly_id``."""
        existing = self._conn.execute(
            "SELECT id FROM events WHERE session_id = ? AND anomaly_id = ?"
            " AND status = ?",
            (session_id, event.anomaly_id, EVENT_STATUS_ONGOING),
        ).fetchone()
        values = self._event_values(
            session_id, event, media=None, status=EVENT_STATUS_ONGOING
        )
        if existing is None:
            cursor = self._conn.execute(
                "INSERT INTO events (session_id, anomaly_id, detector, start, end,"
                " duration, max_intensity, threshold, timeline, metadata, regions,"
                " media, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            self._conn.commit()
            return int(cursor.lastrowid)

        self._conn.execute(
            "UPDATE events SET detector = ?, start = ?, end = ?, duration = ?,"
            " max_intensity = ?, threshold = ?, timeline = ?, metadata = ?,"
            " regions = ? WHERE id = ?",
            (
                event.detector,
                event.start,
                event.end,
                event.duration,
                event.max_intensity,
                event.threshold,
                json.dumps(event.timeline),
                json.dumps(event.metadata),
                json.dumps(event.regions),
                int(existing["id"]),
            ),
        )
        self._conn.commit()
        return int(existing["id"])

    def complete_event(
        self, session_id: str, event: AnomalyEvent, media: Optional[str] = None
    ) -> int:
        """Promote an ongoing row to completed, or insert if none exists."""
        existing = self._conn.execute(
            "SELECT id FROM events WHERE session_id = ? AND anomaly_id = ?"
            " AND status = ?",
            (session_id, event.anomaly_id, EVENT_STATUS_ONGOING),
        ).fetchone()
        values = self._event_values(
            session_id, event, media=media, status=EVENT_STATUS_COMPLETED
        )
        if existing is None:
            cursor = self._conn.execute(
                "INSERT INTO events (session_id, anomaly_id, detector, start, end,"
                " duration, max_intensity, threshold, timeline, metadata, regions,"
                " media, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            self._conn.commit()
            return int(cursor.lastrowid)

        self._conn.execute(
            "UPDATE events SET detector = ?, start = ?, end = ?, duration = ?,"
            " max_intensity = ?, threshold = ?, timeline = ?, metadata = ?,"
            " regions = ?, media = ?, status = ? WHERE id = ?",
            (
                event.detector,
                event.start,
                event.end,
                event.duration,
                event.max_intensity,
                event.threshold,
                json.dumps(event.timeline),
                json.dumps(event.metadata),
                json.dumps(event.regions),
                media,
                EVENT_STATUS_COMPLETED,
                int(existing["id"]),
            ),
        )
        self._conn.commit()
        return int(existing["id"])

    def discard_ongoing_event(self, session_id: str, anomaly_id: str) -> None:
        """Drop an ongoing row that failed ``min_duration_seconds``."""
        self._conn.execute(
            "DELETE FROM events WHERE session_id = ? AND anomaly_id = ?"
            " AND status = ?",
            (session_id, anomaly_id, EVENT_STATUS_ONGOING),
        )
        self._conn.commit()

    def insert_event(
        self, session_id: str, event: AnomalyEvent, media: Optional[str] = None
    ) -> int:
        """Persist a completed event (compatibility alias for :meth:`complete_event`)."""
        return self.complete_event(session_id, event, media=media)

    def list_events(self, session_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY start", (session_id,)
        ).fetchall()
        return [self._decode_event(r) for r in rows]

    def get_event(self, event_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return self._decode_event(row) if row else None

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("timeline", "metadata", "regions"):
            data[key] = json.loads(data[key])
        data.setdefault("status", EVENT_STATUS_COMPLETED)
        if not data.get("status"):
            data["status"] = EVENT_STATUS_COMPLETED
        return data
