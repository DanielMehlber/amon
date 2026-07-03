"""Background process for all blocking I/O (database writes, GIF encoding).

The monitoring pipeline must never stall on disk I/O, so finalised events
and calibration results are handed over through a multiprocessing queue to
a worker process which generates media and writes to the database.  The
queue is unbounded: enqueueing is a memory copy and returns immediately.
"""
from __future__ import annotations

import multiprocessing as mp
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from amon import media
from amon.db import Database
from amon.model import AnomalyEvent


class BackgroundWorker:
    """Owns the worker process; the pipeline only ever calls ``submit_*``."""

    def __init__(self, db_path: str, media_dir: str, session_id: str, media_config: dict):
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=_worker_main,
            args=(self._queue, str(db_path), str(media_dir), session_id, dict(media_config)),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def submit_event(self, event: AnomalyEvent, frames: List[Tuple[float, np.ndarray]], fps: float) -> None:
        """Queue a finalised event for media generation and persistence."""
        self._queue.put(("event", event, frames, fps))

    def submit_calibration(
        self, thresholds: dict, annotations: dict,
        frames: List[Tuple[float, np.ndarray]], fps: float,
    ) -> None:
        """Queue calibration results for annotation media and persistence."""
        self._queue.put(("calibration", thresholds, annotations, frames, fps))

    def close(self) -> None:
        """Flush remaining jobs and wait for the worker to finish."""
        self._queue.put(None)
        self._process.join()


def _worker_main(queue: mp.Queue, db_path: str, media_dir: str, session_id: str, cfg: dict) -> None:
    db = Database(db_path)
    media_root = Path(media_dir)
    gif_fps = float(cfg.get("gif_max_fps", 10.0))
    try:
        while True:
            job = queue.get()
            if job is None:
                return
            if job[0] == "event":
                _, event, frames, fps = job
                path = _handle_media(lambda p: media.write_event_gif(frames, event, p, fps, gif_fps),
                                     media_root / session_id)
                db.insert_event(session_id, event, media=path)
            elif job[0] == "calibration":
                _, thresholds, annotations, frames, fps = job
                path = _handle_media(
                    lambda p: media.write_calibration_gif(frames, annotations, p, fps, gif_fps),
                    media_root / session_id,
                )
                db.save_calibration(session_id, thresholds, annotations, media=path)
    finally:
        db.close()


def _handle_media(writer, directory: Path) -> Optional[str]:
    """Run a media writer, returning the created path (or None on failure)."""
    path = directory / f"{uuid.uuid4().hex}.gif"
    try:
        writer(path)
        return str(path)
    except Exception:  # media problems must never lose the event record
        return None
