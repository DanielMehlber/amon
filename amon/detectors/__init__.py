"""Anomaly detector plugin interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from amon.model import Box, CalibrationResult, Frame


class Detector(ABC):
    """Interface every anomaly detector plugin implements.

    A detector starts in *calibration* mode: frames are fed to
    :meth:`process` and the detector gathers statistics.  The pipeline then
    calls :meth:`finish_calibration`, which derives thresholds and switches
    the detector to *detection* mode.  From then on :meth:`process` returns
    a mapping of anomaly IDs to intensity values; the pipeline compares
    those against the calibrated thresholds (queried via
    :meth:`thresholds`) and aggregates events.

    Subclasses only implement ``_calibrate``, ``_finish_calibration`` and
    ``_detect`` - mode handling lives here so implementations stay concise.
    """

    #: Short identifier used in event records and the report UI.
    name: str = "detector"

    def __init__(self, config: dict = None):
        self.config = {**self.default_config(), **(config or {})}
        self.mode = "calibration"
        self._thresholds: Dict[str, float] = {}

    @classmethod
    def default_config(cls) -> dict:
        """Detector-specific configuration defaults (override as needed)."""
        return {}

    def process(self, frame: Frame) -> Dict[str, float]:
        """Feed a frame; returns anomaly intensities while in detection mode."""
        if self.mode == "calibration":
            self._calibrate(frame)
            return {}
        return self._detect(frame)

    def finish_calibration(self) -> CalibrationResult:
        """Derive thresholds from gathered statistics and enter detection mode."""
        result = self._finish_calibration()
        self._thresholds = dict(result.thresholds)
        self.mode = "detection"
        return result

    def thresholds(self) -> Dict[str, float]:
        """Calibrated per-anomaly thresholds (valid after calibration)."""
        return dict(self._thresholds)

    def metadata(self, anomaly_id: str) -> dict:
        """Static, JSON-serialisable metadata describing an anomaly."""
        return {}

    def regions(self, anomaly_id: str) -> List[Box]:
        """Image regions affected by the anomaly in the last processed frame."""
        return []

    @abstractmethod
    def _calibrate(self, frame: Frame) -> None:
        """Gather statistics from a calibration frame."""

    @abstractmethod
    def _finish_calibration(self) -> CalibrationResult:
        """Compute thresholds and baseline data from gathered statistics."""

    @abstractmethod
    def _detect(self, frame: Frame) -> Dict[str, float]:
        """Return anomaly intensities for a monitoring frame."""
