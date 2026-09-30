"""Spatial anomaly detector: geometric distortion of the background scene.

During calibration the detector builds a *baseline image* as the temporal
median of sampled calibration frames (removing transient content) and
detects stable Shi-Tomasi corner features on it. Pixels that were ever
bright during calibration - HUD overlays - are dilated and excluded, so
HUD changes never influence this detector.

In detection mode every feature point is tracked from the baseline image
into the current frame with pyramidal Lucas-Kanade optical flow.  A
forward-backward consistency check discards unreliable tracks. The
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
from amon.scale import REFERENCE_WIDTH_PX, rel_width, rel_width_float
from amon.stats import robust_threshold

DISTORTION = "spatial/distortion"

# Absolute defaults were authored at REFERENCE_WIDTH_PX; config stores
# fractions of the processed frame width.
_W = float(REFERENCE_WIDTH_PX)


class SpatialDetector(Detector):
    """Detects background distortions via feature tracking, ignoring HUDs."""

    name = "spatial"

    @classmethod
    def default_config(cls) -> dict:
        return {
            "segmentation": {
                "bright_threshold": 220,  # HUD brightness (matches HudDetector)
                "exclusion_dilate_rel": 21 / _W,  # was 21 px
            },
            "features": {
                "max_corners": 150,
                "corner_quality": 0.03,
                "corner_min_distance_rel": 7 / _W,  # was 7 px
                "max_baseline_frames": 40,  # calibration frames kept for the median
            },
            "tracking": {
                "fb_max_error_rel": 1.5 / _W,  # was 1.5 px
                "outlier_rank": 3,  # use the k-th largest displacement
                "region_size_rel": 28 / _W,  # was 28 px
            },
            "thresholds": {
                "sigma_k": 8.0,
                "floor_rel": 2.5 / _W,  # was 2.5 px
            },
            # Per-anomaly threshold multipliers (float also accepted).
            "tolerance": {"distortion": 1.0},
        }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._grays: List[np.ndarray] = []
        self._bright: Optional[np.ndarray] = None
        self._baseline: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._last_moved: List[Box] = []
        self._frame_width = REFERENCE_WIDTH_PX

    def _set_frame_width(self, image: np.ndarray) -> None:
        self._frame_width = int(image.shape[1])

    def _length_px(self, key: str, *, minimum: int = 1) -> int:
        return rel_width(self.config[key], self._frame_width, minimum=minimum)

    def _length_pxf(self, key: str, *, minimum: float = 0.0) -> float:
        return rel_width_float(
            self.config[key], self._frame_width, minimum=minimum
        )

    # --- calibration ------------------------------------------------------
    def _calibrate(self, frame: Frame) -> None:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        self._set_frame_width(gray)
        bright = gray > int(self.config["bright_threshold"])
        self._bright = bright if self._bright is None else (self._bright | bright)
        self._grays.append(gray)

    def _finish_calibration(self) -> CalibrationResult:
        if self._grays:
            self._set_frame_width(self._grays[0])
        # Subsample stored frames to bound the median computation.
        keep = int(self.config["max_baseline_frames"])
        stride = max(1, len(self._grays) // keep)
        samples = self._grays[::stride]
        self._baseline = np.median(np.stack(samples), axis=0).astype(np.uint8)

        dilate = max(3, self._length_px("exclusion_dilate_rel") * 3)
        kernel = np.ones((dilate, dilate), np.uint8)
        excluded = cv2.dilate(self._bright.astype(np.uint8), kernel)

        # Find the corner features in the baseline image
        # using Shi-Tomasi corner detection (which is the default)
        self._points = cv2.goodFeaturesToTrack(
            self._baseline,
            maxCorners=int(self.config["max_corners"]),
            qualityLevel=float(self.config["corner_quality"]),
            minDistance=self._length_px("corner_min_distance_rel"),
            mask=(1 - excluded) * 255,
        )

        jitter = [self._calculate_keypoint_displacement(g) for g in samples]
        thresholds = {
            DISTORTION: robust_threshold(
                jitter,
                self.config["sigma_k"],
                self._length_pxf("floor_rel"),
            )
        }
        keypoints = [] if self._points is None else self._points.reshape(-1, 2).tolist()
        self._grays = []  # free calibration memory
        return CalibrationResult(
            thresholds=thresholds, annotations={"keypoints": keypoints}
        )

    # --- detection ----------------------------------------------------------
    def _detect(self, frame: Frame) -> Dict[str, float]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        self._set_frame_width(gray)
        return {
            DISTORTION: self._calculate_keypoint_displacement(gray, record_regions=True)
        }

    def _calculate_keypoint_displacement(
        self, gray: np.ndarray, record_regions: bool = False
    ) -> float:
        """Rank-filtered maximum displacement of baseline features in ``gray``."""
        if self._points is None or len(self._points) == 0:
            return 0.0

        # Calculate the forward and backward optical flow of the tracked keypoints
        # between the baseline image and the current frame.
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(
            self._baseline, gray, self._points, None
        )
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, self._baseline, fwd, None)

        # Calculate the forward-backward error.
        fb_error = np.linalg.norm((back - self._points).reshape(-1, 2), axis=1)
        valid = (
            (st_f.ravel() == 1)
            & (st_b.ravel() == 1)
            & (fb_error < self._length_pxf("fb_max_error_rel"))
        )
        if not valid.any():
            return 0.0

        # Calculate the displacement of the tracked keypoints.
        displacement = np.linalg.norm((fwd - self._points).reshape(-1, 2), axis=1)
        displacement[~valid] = 0.0

        # Ignore points under (or next to) bright HUD pixels so overlay
        # motion/blink cannot register as background distortion.
        if displacement.any():
            thr = int(self.config["bright_threshold"])
            dilate = max(3, self._length_px("exclusion_dilate_rel") * 3)
            kernel = np.ones((dilate, dilate), np.uint8)
            hud = cv2.dilate((gray > thr).astype(np.uint8), kernel)
            h, w = hud.shape[:2]

            def _on_hud(pts: np.ndarray) -> np.ndarray:
                rounded = np.round(pts.reshape(-1, 2)).astype(int)
                flags = np.zeros(len(rounded), dtype=bool)
                for i, (x, y) in enumerate(rounded):
                    if 0 <= x < w and 0 <= y < h and hud[y, x]:
                        flags[i] = True
                return flags

            displacement[_on_hud(self._points) | _on_hud(fwd)] = 0.0

        # Record the regions of the moved keypoints.
        if record_regions:
            threshold = self._thresholds.get(DISTORTION, np.inf)
            half = self._length_px("region_size_rel") // 2
            self._last_moved = [
                (int(x) - half, int(y) - half, 2 * half, 2 * half)
                for (x, y), d in zip(self._points.reshape(-1, 2), displacement)
                if d >= threshold
            ]

        # Calculate the rank-filtered maximum displacement to avoid large outliers
        # by using the k-th largest displacement. Alternatives like the mean or median
        # would be more sensitive to small changes in the background.
        rank = min(int(self.config["outlier_rank"]), int(valid.sum())) - 1
        return float(np.sort(displacement)[::-1][max(rank, 0)])

    # --- event enrichment ------------------------------------------------------
    def metadata(self, anomaly_id: str) -> dict:
        count = 0 if self._points is None else len(self._points)
        return {"tracked_points": int(count), "moved_points": len(self._last_moved)}

    def regions(self, anomaly_id: str) -> List[Box]:
        return list(self._last_moved)
