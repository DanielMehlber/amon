"""Spatial anomaly detector: geometric distortion of the background scene.

During calibration the detector builds a *baseline image* as the temporal
median of sampled calibration frames (removing transient content) and
detects stable Shi-Tomasi corner features on it.  Pixels that were ever
bright during calibration - HUD overlays - are dilated and excluded, so
HUD changes never influence this detector.

In detection mode every feature point is tracked from the baseline image
into the current frame with pyramidal Lucas-Kanade optical flow.  A
forward-backward consistency check discards unreliable tracks.  The
anomaly intensity ``spatial/distortion`` is the third-largest point
displacement in pixels: a genuine distortion moves a cluster of points,
while the ranking makes single-point outliers harmless.

The detection threshold is calibrated by measuring displacement jitter of
the calibration frames against the baseline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from amon.detectors import Detector
from amon.model import Box, CalibrationResult, Frame
from amon.stats import robust_threshold

DISTORTION = "spatial/distortion"


class SpatialDetector(Detector):
    """Detects background distortions via feature tracking, ignoring HUDs."""

    name = "spatial"

    @classmethod
    def default_config(cls) -> dict:
        return {
            "bright_threshold": 220,   # HUD brightness (matches HudDetector)
            "exclusion_dilate": 21,    # px dilation around HUD pixels
            "max_corners": 150,
            "corner_quality": 0.03,
            "corner_min_distance": 7,
            "max_baseline_frames": 40, # calibration frames kept for the median
            "fb_max_error": 1.5,       # forward-backward tolerance (px)
            "outlier_rank": 3,         # use the k-th largest displacement
            "sigma_k": 8.0,
            "floor": 2.5,              # minimum displacement threshold (px)
            "region_size": 28,         # highlight box size around moved points
        }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._grays: List[np.ndarray] = []
        self._bright: Optional[np.ndarray] = None
        self._baseline: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._last_moved: List[Box] = []

    # --- calibration ------------------------------------------------------
    def _calibrate(self, frame: Frame) -> None:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        bright = gray > int(self.config["bright_threshold"])
        self._bright = bright if self._bright is None else (self._bright | bright)
        self._grays.append(gray)

    def _finish_calibration(self) -> CalibrationResult:
        # Subsample stored frames to bound the median computation.
        keep = int(self.config["max_baseline_frames"])
        stride = max(1, len(self._grays) // keep)
        samples = self._grays[::stride]
        self._baseline = np.median(np.stack(samples), axis=0).astype(np.uint8)

        kernel = np.ones((self.config["exclusion_dilate"],) * 2, np.uint8)
        excluded = cv2.dilate(self._bright.astype(np.uint8), kernel)
        self._points = cv2.goodFeaturesToTrack(
            self._baseline,
            maxCorners=int(self.config["max_corners"]),
            qualityLevel=float(self.config["corner_quality"]),
            minDistance=int(self.config["corner_min_distance"]),
            mask=(1 - excluded) * 255,
        )

        jitter = [self._displacement(g) for g in samples]
        thresholds = {
            DISTORTION: robust_threshold(jitter, self.config["sigma_k"], self.config["floor"])
        }
        keypoints = [] if self._points is None else self._points.reshape(-1, 2).tolist()
        self._grays = []  # free calibration memory
        return CalibrationResult(thresholds=thresholds, annotations={"keypoints": keypoints})

    # --- detection ----------------------------------------------------------
    def _detect(self, frame: Frame) -> Dict[str, float]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        return {DISTORTION: self._displacement(gray, record_regions=True)}

    def _displacement(self, gray: np.ndarray, record_regions: bool = False) -> float:
        """Rank-filtered maximum displacement of baseline features in ``gray``."""
        if self._points is None or len(self._points) == 0:
            return 0.0
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(self._baseline, gray, self._points, None)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, self._baseline, fwd, None)
        fb_error = np.linalg.norm((back - self._points).reshape(-1, 2), axis=1)
        valid = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_error < self.config["fb_max_error"])
        if not valid.any():
            return 0.0
        displacement = np.linalg.norm((fwd - self._points).reshape(-1, 2), axis=1)
        displacement[~valid] = 0.0

        if record_regions:
            threshold = self._thresholds.get(DISTORTION, np.inf)
            half = int(self.config["region_size"]) // 2
            self._last_moved = [
                (int(x) - half, int(y) - half, 2 * half, 2 * half)
                for (x, y), d in zip(self._points.reshape(-1, 2), displacement)
                if d >= threshold
            ]
        rank = min(int(self.config["outlier_rank"]), int(valid.sum())) - 1
        return float(np.sort(displacement)[::-1][max(rank, 0)])

    # --- event enrichment ------------------------------------------------------
    def metadata(self, anomaly_id: str) -> dict:
        count = 0 if self._points is None else len(self._points)
        return {"tracked_points": int(count), "moved_points": len(self._last_moved)}

    def regions(self, anomaly_id: str) -> List[Box]:
        return list(self._last_moved)
