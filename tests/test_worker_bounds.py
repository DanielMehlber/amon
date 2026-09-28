"""Resource-bound checks for multi-day monitoring stability."""

import queue as queue_mod

import pytest

from amon.model import AnomalyEvent
from amon.worker import BackgroundWorker


def test_worker_drops_open_jobs_when_queue_full(tmp_path):
    worker = BackgroundWorker(
        tmp_path / "amon.sqlite",
        tmp_path / "media",
        "Queue-Test",
        {"gif_max_fps": 10.0, "queue_maxsize": 1},
    )
    # Do not start the process — queue stays unread so it fills immediately.
    event = AnomalyEvent(
        anomaly_id="hud/alert/new",
        detector="hud",
        start=0.0,
        end=0.1,
        max_intensity=1.0,
        threshold=0.5,
    )
    worker.submit_open(event)  # fills the single slot
    worker.submit_open(event)  # must not raise — best-effort drop
    # Final events degrade (DB-only retry) instead of aborting the session.
    worker.submit_event(event, [], 20.0)


def test_worker_blocking_put_times_out(tmp_path, monkeypatch):
    worker = BackgroundWorker(
        tmp_path / "amon.sqlite",
        tmp_path / "media",
        "Queue-Test",
        {"gif_max_fps": 10.0, "queue_maxsize": 1},
    )
    event = AnomalyEvent(
        anomaly_id="a",
        detector="temporal",
        start=0.0,
        end=1.0,
        max_intensity=1.0,
        threshold=0.5,
    )
    worker.submit_open(event)
    monkeypatch.setattr("amon.worker._PUT_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(RuntimeError, match="stalled"):
        worker.submit_calibration({}, {}, [], 20.0)


def test_worker_event_degrades_to_no_frames_when_queue_full(tmp_path, monkeypatch):
    worker = BackgroundWorker(
        tmp_path / "amon.sqlite",
        tmp_path / "media",
        "Queue-Test",
        {"gif_max_fps": 10.0, "queue_maxsize": 1},
    )
    event = AnomalyEvent(
        anomaly_id="hud/x/new",
        detector="hud",
        start=0.0,
        end=1.0,
        max_intensity=1.0,
        threshold=0.5,
    )
    # Fill the queue with a heavyweight stand-in; event must not raise.
    worker.submit_open(event)
    monkeypatch.setattr("amon.worker._PUT_TIMEOUT_SECONDS", 0.05)
    worker.submit_event(event, [(0.0, object())], 20.0)  # type: ignore[list-item]
