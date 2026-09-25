"""Anomaly detector plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Union

from amon.model import Box, CalibrationResult, Frame

#: Scalar (all anomalies) or mapping keyed by full anomaly ID / trailing
#: segment (``noise``, ``text``, …) / ``default``.
ToleranceConfig = Union[float, Mapping[str, float]]


def resolve_tolerance(anomaly_id: str, tolerance: ToleranceConfig) -> float:
    """Return the threshold multiplier for ``anomaly_id``.

    ``tolerance`` may be a single float (applied to every anomaly) or a
    mapping.  Mapping lookup order:

    1. exact anomaly ID (``temporal/noise``);
    2. trailing segment (``noise``, ``text``, ``distortion``, …);
    3. ``default`` key;
    4. ``1.0``.
    """
    if isinstance(tolerance, Mapping):
        if anomaly_id in tolerance:
            return float(tolerance[anomaly_id])
        aspect = anomaly_id.rsplit("/", 1)[-1]
        if aspect in tolerance:
            return float(tolerance[aspect])
        if "default" in tolerance:
            return float(tolerance["default"])
        return 1.0
    return float(tolerance)


class Detector(ABC):
    """Interface every anomaly detector plugin implements.

    A detector starts in *calibration* mode: frames are fed to
    :meth:`process` and the detector gathers statistics.  The pipeline then
    calls :meth:`finish_calibration`, which derives thresholds and switches
    the detector to *detection* mode.  From then on :meth:`process` returns
    a mapping of anomaly IDs to intensity values; the pipeline compares
    those against the calibrated thresholds (queried via
    :meth:`thresholds`) and aggregates events.  Config key ``tolerance``
    multiplies calibrated thresholds — a float for all anomalies of this
    detector, or a mapping per anomaly type (``1.2`` means intensity must
    exceed the learned cutoff by 20%).

    Subclasses only implement ``_calibrate``, ``_finish_calibration`` and
    ``_detect`` - mode handling lives here so implementations stay concise.
    """

    #: Short identifier used in event records and the report UI.
    name: str = "detector"

    def __init__(self, config: dict = None):
        # ``tolerance`` multiplies calibrated thresholds: a float for every
        # anomaly, or a mapping keyed by anomaly ID / trailing segment
        # (``noise``, ``text``, …) / ``default``.
        self.config = {
            "tolerance": 1.0,
            **self.default_config(),
            **(config or {}),
        }
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
        tolerance = self.config.get("tolerance", 1.0)
        thresholds = {
            anomaly_id: float(value) * resolve_tolerance(anomaly_id, tolerance)
            for anomaly_id, value in result.thresholds.items()
        }
        self._thresholds = thresholds
        self.mode = "detection"
        return CalibrationResult(thresholds=thresholds, annotations=result.annotations)

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
