"""CSV export of session events for offline analysis."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

COLUMNS = (
    "event_id",
    "anomaly_id",
    "detector",
    "start_s",
    "end_s",
    "duration_s",
    "max_intensity",
    "threshold",
    "start_time",
    "end_time",
    "media",
)


def _wall_time(session_start: Optional[float], offset_s: float) -> str:
    if session_start is None:
        return ""
    return datetime.fromtimestamp(session_start + offset_s).isoformat(
        sep=" ", timespec="seconds"
    )


def write_events_csv(session: dict, events: List[dict], out_path: Path) -> Path:
    """Write one row per anomaly event; returns ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session_start = session.get("started_at")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event["id"],
                    "anomaly_id": event["anomaly_id"],
                    "detector": event["detector"],
                    "start_s": f"{event['start']:.3f}",
                    "end_s": f"{event['end']:.3f}",
                    "duration_s": f"{event['duration']:.3f}",
                    "max_intensity": f"{event['max_intensity']:.6f}",
                    "threshold": f"{event['threshold']:.6f}",
                    "start_time": _wall_time(session_start, event["start"]),
                    "end_time": _wall_time(session_start, event["end"]),
                    "media": event.get("media") or "",
                }
            )
    return out_path
