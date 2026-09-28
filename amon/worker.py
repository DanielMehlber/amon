"""Background process for all blocking I/O (database writes, GIF encoding).

The monitoring pipeline must never stall on disk I/O, so event lifecycle
jobs and calibration results are handed over through a multiprocessing
queue to a worker process which generates media and writes to the
database.

The queue is **bounded** (``media.queue_maxsize``, default 8) so a stalled
or dead worker cannot grow RAM without bound over multi-day runs.  Ongoing
refreshes are best-effort (``put_nowait``, dropped when full); final
events apply back-pressure, then degrade to DB-only (no GIF frames) rather
than aborting the session.

Job kinds:

- ``open`` / ``open_update`` — persist or refresh an *ongoing* event so the
  report can show anomalies that have not closed yet;
- ``discard`` — drop an ongoing row that failed the minimum-duration gate;
- ``event`` — finalise an event (media + ``completed`` status);
- ``calibration`` — store calibration thresholds / review media.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_mod
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from amon import media
from amon.db import Database
from amon.model import AnomalyEvent

log = logging.getLogger("amon.worker")

#: Default bound on pending background jobs.  Kept small because each event
#: job may carry multi-megabyte evidence frames; prefer back-pressure over RAM.
_DEFAULT_QUEUE_MAXSIZE = 8
#: How long finalisation may block waiting for queue space (seconds).
_PUT_TIMEOUT_SECONDS = 5.0


class BackgroundWorker:
    """Owns the worker process; the pipeline only ever calls ``submit_*``."""

    def __init__(
        self, db_path: str, media_dir: str, session_id: str, media_config: dict
    ):
        maxsize = int(media_config.get("queue_maxsize", _DEFAULT_QUEUE_MAXSIZE))
        self._queue: mp.Queue = mp.Queue(maxsize=max(maxsize, 1))
        self._process = mp.Process(
            target=_worker_main,
            args=(
                self._queue,
                str(db_path),
                str(media_dir),
                session_id,
                dict(media_config),
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        """True while the background process is still running."""
        return self._process.is_alive()

    def submit_open(self, event: AnomalyEvent) -> None:
        """Queue an ongoing-event insert/refresh (dropped if the queue is full)."""
        self._put_best_effort(("open", event))

    def submit_discard(self, anomaly_id: str) -> None:
        """Queue removal of an ongoing event that was too short to report."""
        self._put_best_effort(("discard", anomaly_id))

    def submit_event(
        self, event: AnomalyEvent, frames: List[Tuple[float, np.ndarray]], fps: float
    ) -> None:
        """Queue a finalised event for media generation and persistence.

        If the queue stays full, retries once **without** evidence frames so the
        DB row is not lost; only then logs and drops the job (session continues).
        """
        if self._put_blocking(("event", event, frames, fps), fatal=False):
            return
        if frames:
            log.warning(
                "background queue full — persisting %s without evidence GIF",
                event.anomaly_id,
            )
            if self._put_blocking(("event", event, [], fps), fatal=False):
                return
        log.error(
            "background worker queue full — dropping event %s persistence",
            event.anomaly_id,
        )

    def submit_calibration(
        self,
        thresholds: dict,
        annotations: dict,
        frames: List[Tuple[float, np.ndarray]],
        fps: float,
    ) -> None:
        """Queue calibration results for annotation media and persistence."""
        self._put_blocking(
            ("calibration", thresholds, annotations, frames, fps), fatal=True
        )

    def _put_best_effort(self, job: tuple) -> None:
        try:
            self._queue.put_nowait(job)
        except queue_mod.Full:
            log.warning(
                "background queue full — dropping %s job (worker may be behind)",
                job[0],
            )

    def _put_blocking(self, job: tuple, *, fatal: bool) -> bool:
        """Return True if enqueued.  When ``fatal``, raise on timeout."""
        try:
            self._queue.put(job, timeout=_PUT_TIMEOUT_SECONDS)
            return True
        except queue_mod.Full:
            if fatal:
                raise RuntimeError(
                    "background worker queue full for "
                    f"{_PUT_TIMEOUT_SECONDS:.0f}s — worker may have stalled"
                )
            return False

    def close(self) -> None:
        """Flush remaining jobs and wait for the worker to finish."""
        try:
            self._queue.put(None, timeout=_PUT_TIMEOUT_SECONDS)
        except queue_mod.Full:
            log.error("could not send shutdown to background worker (queue full)")
        self._process.join(timeout=60.0)


def _worker_main(
    queue: mp.Queue, db_path: str, media_dir: str, session_id: str, cfg: dict
) -> None:
    db = Database(db_path)
    media_root = Path(media_dir)
    gif_fps = float(cfg.get("gif_max_fps", 10.0))
    try:
        while True:
            job = queue.get()
            if job is None:
                return
            try:
                _handle_job(job, db, media_root, session_id, gif_fps)
            except Exception:
                # One bad job must not kill the worker for a multi-day session.
                log.exception("background worker failed on %s job", job[0])
    finally:
        db.close()


def _handle_job(
    job: tuple, db: Database, media_root: Path, session_id: str, gif_fps: float
) -> None:
    kind = job[0]
    if kind == "open":
        _, event = job
        db.upsert_ongoing_event(session_id, event)
    elif kind == "discard":
        _, anomaly_id = job
        db.discard_ongoing_event(session_id, anomaly_id)
    elif kind == "event":
        _, event, frames, fps = job
        path = None
        if frames:
            path = _handle_media(
                lambda p: media.write_event_gif(frames, event, p, fps, gif_fps),
                media_root / session_id,
            )
        db.complete_event(session_id, event, media=path)
    elif kind == "calibration":
        _, thresholds, annotations, frames, fps = job
        path = None
        if frames:
            path = _handle_media(
                lambda p: media.write_calibration_gif(
                    frames, annotations, p, fps, gif_fps
                ),
                media_root / session_id,
            )
        db.save_calibration(session_id, thresholds, annotations, media=path)
    else:
        log.warning("unknown background job kind %r", kind)


def _handle_media(writer, directory: Path) -> Optional[str]:
    """Run a media writer, returning the created path (or None on failure)."""
    path = directory / f"{uuid.uuid4().hex}.gif"
    try:
        writer(path)
        return str(path)
    except Exception:  # media problems must never lose the event record
        log.exception("media write failed for %s", path)
        return None
