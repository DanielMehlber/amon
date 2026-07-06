"""The real-time monitoring pipeline.

Lifecycle:

1. **Calibration** - frames are fed to all detectors in calibration mode
   for the configured duration.  Afterwards :meth:`Detector.finish_calibration`
   derives thresholds, detectors switch to detection mode and an annotated
   calibration review job is handed to the background worker.
2. **Monitoring** - each frame's detector intensities are compared against
   the calibrated thresholds by the :class:`~amon.aggregate.EventAggregator`.
   While an event is open the pipeline captures an evidence clip (a little
   lead-in from the ring buffer plus up to ``media.max_clip_seconds``).
   Finalised events are enqueued to the :class:`~amon.worker.BackgroundWorker`
   so GIF encoding and database writes never block frame processing.

The pipeline runs until the source is exhausted or ``stop`` is set.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from amon.aggregate import EventAggregator, Reading
from amon.db import Database
from amon.detectors import Detector
from amon.model import AnomalyEvent, Frame
from amon.names import generate_session_id
from amon.plugins import instantiate
from amon.sources import VideoSource
from amon.worker import BackgroundWorker

log = logging.getLogger("amon.pipeline")

#: Length of the annotated calibration review clip.
CALIBRATION_CLIP_SECONDS = 2.0


class Pipeline:
    """Runs one monitoring session as described in the module docstring.

    ``source``, ``detectors`` and ``worker`` are normally instantiated from
    the configuration but can be injected for testing.
    """

    def __init__(
        self,
        config: dict,
        source: Optional[VideoSource] = None,
        detectors: Optional[List[Detector]] = None,
        worker: Optional[BackgroundWorker] = None,
    ):
        self.config = config
        self.source = source or instantiate(config["video_source"])
        self.detectors = (
            detectors
            if detectors is not None
            else [instantiate(spec) for spec in config["detectors"]]
        )
        self.aggregator = EventAggregator(config["aggregation"])
        self.session_id: Optional[str] = None

        data_dir = Path(config["data_dir"])
        self.db_path = data_dir / "amon.sqlite"
        self.media_dir = data_dir / "media"
        self._worker = worker

        media_cfg = config["media"]
        self._lead = float(media_cfg["lead_seconds"])
        self._max_clip = float(media_cfg["max_clip_seconds"])
        self._detector_of: Dict[str, Detector] = {}
        self._clips: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        self._enrichment: Dict[str, Tuple[dict, list]] = {}

    def run(self, max_frames: Optional[int] = None, stop=None) -> str:
        """Process the stream until it ends; returns the session ID."""
        fps = self.source.fps
        calibration_end = float(self.config["calibration"]["duration_seconds"])
        ring: deque = deque(
            maxlen=int(max(self._lead, CALIBRATION_CLIP_SECONDS) * fps) + 4
        )

        db = Database(self.db_path)
        existing = {row["id"] for row in db.list_sessions()}
        self.session_id = generate_session_id(existing=existing)
        db.create_session(
            self.session_id,
            self.config["session_name"],
            str(self.config["video_source"].get("config", {}).get("path", "")),
            fps,
        )
        db.close()

        worker = self._worker or BackgroundWorker(
            self.db_path, self.media_dir, self.session_id, self.config["media"]
        )
        worker.start()
        log.info(
            "session %s started (calibrating for %.1fs)",
            self.session_id,
            calibration_end,
        )

        calibrated = False
        last_t = 0.0
        try:
            for frame in self.source.frames():
                if stop is not None and stop.is_set():
                    break
                if max_frames is not None and frame.index >= max_frames:
                    break
                last_t = frame.timestamp
                ring.append((frame.timestamp, frame.image))

                if not calibrated and frame.timestamp >= calibration_end:
                    self._finish_calibration(worker, ring, fps)
                    calibrated = True
                if not calibrated:
                    for detector in self.detectors:
                        detector.process(frame)
                    continue

                self._monitor_frame(frame, ring, worker, fps)
            for event in self.aggregator.flush():
                self._finalize_event(event, worker, fps)
        finally:
            worker.close()
            self.source.close()
            db = Database(self.db_path)
            db.finish_session(self.session_id)
            db.close()
            log.info("session %s finished", self.session_id)
        return self.session_id

    # --- calibration hand-off ------------------------------------------------
    def _finish_calibration(
        self, worker: BackgroundWorker, ring: deque, fps: float
    ) -> None:
        thresholds: Dict[str, dict] = {}
        annotations: dict = {}
        for detector in self.detectors:
            result = detector.finish_calibration()
            thresholds[detector.name] = result.thresholds
            annotations.update(result.annotations)
            for anomaly_id in result.thresholds:
                self._detector_of[anomaly_id] = detector
        clip = [(t, img) for t, img in ring][-int(CALIBRATION_CLIP_SECONDS * fps) :]
        worker.submit_calibration(thresholds, annotations, clip, fps)
        log.info("calibration complete: %d anomalies armed", len(self._detector_of))

    # --- monitoring ------------------------------------------------------------
    def _monitor_frame(
        self, frame: Frame, ring: deque, worker: BackgroundWorker, fps: float
    ) -> None:
        readings: Dict[str, Reading] = {}
        for detector in self.detectors:
            calibrated = detector.thresholds()
            for anomaly_id, intensity in detector.process(frame).items():
                readings[anomaly_id] = Reading(
                    intensity, calibrated[anomaly_id], detector.name
                )

        opened, closed, discarded = self.aggregator.update(frame.timestamp, readings)
        for anomaly_id in discarded:  # too short to report - free capture state
            self._clips.pop(anomaly_id, None)
            self._enrichment.pop(anomaly_id, None)

        for anomaly_id in opened:
            detector = self._detector_of[anomaly_id]
            self._enrichment[anomaly_id] = (
                detector.metadata(anomaly_id),
                [list(box) for box in detector.regions(anomaly_id)],
            )
            # Start the evidence clip with lead-in frames from the ring buffer.
            lead_start = frame.timestamp - self._lead
            self._clips[anomaly_id] = [(t, img) for t, img in ring if t >= lead_start]

        for anomaly_id, clip in self._clips.items():
            if (
                clip
                and frame.timestamp <= clip[0][0] + self._lead + self._max_clip
                and clip[-1][0] < frame.timestamp
            ):
                clip.append((frame.timestamp, frame.image))

        for event in closed:
            self._finalize_event(event, worker, fps)

    def _finalize_event(
        self, event: AnomalyEvent, worker: BackgroundWorker, fps: float
    ) -> None:
        metadata, regions = self._enrichment.pop(event.anomaly_id, ({}, []))
        event.metadata = metadata
        event.regions = regions
        clip = self._clips.pop(event.anomaly_id, [])
        worker.submit_event(event, clip, fps)
        log.info(
            "event %s: %.1fs-%.1fs (%.1fs, peak %.2f)",
            event.anomaly_id,
            event.start,
            event.end,
            event.duration,
            event.max_intensity,
        )
