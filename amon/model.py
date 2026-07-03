"""Core data model shared across the framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

#: Axis-aligned bounding box as (x, y, width, height) in pixels.
Box = Tuple[int, int, int, int]


@dataclass
class Frame:
    """A single video frame and its position in the stream."""

    index: int
    timestamp: float  #: seconds since the start of the session
    image: np.ndarray  #: BGR uint8 image


@dataclass
class CalibrationResult:
    """Everything a detector learned during its calibration phase.

    ``thresholds`` maps anomaly IDs to the intensity level above which a
    detection is reported.  ``annotations`` holds JSON-serialisable,
    detector-specific baseline information (keypoints, HUD elements, ...)
    used for calibration review and media annotation.
    """

    thresholds: Dict[str, float]
    annotations: dict = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """A contiguous anomaly occurrence aggregated over many frames."""

    anomaly_id: str
    detector: str
    start: float  #: seconds since session start
    end: float
    max_intensity: float
    threshold: float
    timeline: List[Tuple[float, float]] = field(default_factory=list)  #: (t, intensity)
    metadata: dict = field(default_factory=dict)
    regions: List[Box] = field(default_factory=list)  #: highlighted image regions

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        return self.end - self.start
