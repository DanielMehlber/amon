"""SQLite-backed persistence for sessions, calibrations and events.

A single file database (``<data_dir>/amon.sqlite``) holds all monitoring
sessions.  Media files (GIFs) live next to it and are referenced by
relative paths, so the whole data directory can be archived or moved.
Timestamps of events are stored in seconds relative to the session start;
the session row carries the absolute wall-clock start time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Union

from amon.model import AnomalyEvent

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
    media TEXT
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
    def insert_event(
        self, session_id: str, event: AnomalyEvent, media: Optional[str] = None
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO events (session_id, anomaly_id, detector, start, end, duration,"
            " max_intensity, threshold, timeline, metadata, regions, media)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
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
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

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
        return data
