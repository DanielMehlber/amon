"""Temporal anomaly detector: noise, flicker and contrast changes.

Algorithms (all on the grayscale frame):

- **Noise** (``temporal/noise``): the difference between consecutive frames
  is taken and its spatial mean removed (which cancels global brightness
  shifts such as flicker).  The intensity is a robust standard deviation
  (``1.4826 * MAD``) of the residual, so small legitimately changing areas
  like blinking HUD elements barely register while sensor noise, which
  affects every pixel, produces a large value.
- **Flicker** (``temporal/flicker``): the mean absolute change of the
  global brightness over a short sliding window.  Uniform brightness
  oscillation moves the global mean strongly; noise and local changes do
  not.
- **Contrast** (``temporal/contrast``): the relative deviation of the
  frame's intensity standard deviation from the baseline learned during
  calibration, ``|std / baseline_std - 1|``.

Thresholds are derived from calibration samples via
:func:`amon.stats.robust_threshold`; the configurable floors only encode
the physical scale of each metric.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np

from amon.detectors import Detector
from amon.model import CalibrationResult, Frame
from amon.stats import robust_threshold

NOISE = "temporal/noise"
FLICKER = "temporal/flicker"
CONTRAST = "temporal/contrast"


class TemporalDetector(Detector):
    """Detects noise, flicker and contrast changes on the whole frame."""

    name = "temporal"

    @classmethod
    def default_config(cls) -> dict:
        return {
            "window_seconds": 0.5,   # sliding window for the flicker metric
            "sigma_k": 8.0,          # threshold distance in robust sigmas
            "noise_floor": 3.0,      # minimum noise threshold (gray levels)
            "flicker_floor": 5.0,    # minimum flicker threshold (gray levels)
            "contrast_floor": 0.15,  # minimum relative contrast deviation
        }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._prev_gray: Optional[np.ndarray] = None
        self._means: deque = deque()  # (timestamp, global mean)
        self._samples: Dict[str, list] = {NOISE: [], FLICKER: []}
        self._std_samples: list = []
        self._baseline_std = 1.0

    # --- shared metric computation ----------------------------------------
    def _metrics(self, frame: Frame) -> Dict[str, float]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        noise = 0.0
        if self._prev_gray is not None:
            diff = gray - self._prev_gray
            residual = diff - float(diff.mean())
            noise = 1.4826 * float(np.median(np.abs(residual)))
        self._prev_gray = gray

        window = float(self.config["window_seconds"])
        self._means.append((frame.timestamp, float(gray.mean())))
        while self._means and frame.timestamp - self._means[0][0] > window:
            self._means.popleft()
        means = [m for _, m in self._means]
        flicker = float(np.mean(np.abs(np.diff(means)))) if len(means) >= 2 else 0.0

        return {NOISE: noise, FLICKER: flicker, "std": float(gray.std())}

    # --- Detector interface ------------------------------------------------
    def _calibrate(self, frame: Frame) -> None:
        metrics = self._metrics(frame)
        self._samples[NOISE].append(metrics[NOISE])
        self._samples[FLICKER].append(metrics[FLICKER])
        self._std_samples.append(metrics["std"])

    def _finish_calibration(self) -> CalibrationResult:
        sigma_k = float(self.config["sigma_k"])
        self._baseline_std = float(np.median(self._std_samples)) or 1.0
        contrast_samples = [abs(s / self._baseline_std - 1.0) for s in self._std_samples]
        thresholds = {
            NOISE: robust_threshold(self._samples[NOISE], sigma_k, self.config["noise_floor"]),
            FLICKER: robust_threshold(self._samples[FLICKER], sigma_k, self.config["flicker_floor"]),
            CONTRAST: robust_threshold(contrast_samples, sigma_k, self.config["contrast_floor"]),
        }
        annotations = {"temporal": {"baseline_std": self._baseline_std}}
        return CalibrationResult(thresholds=thresholds, annotations=annotations)

    def _detect(self, frame: Frame) -> Dict[str, float]:
        metrics = self._metrics(frame)
        return {
            NOISE: metrics[NOISE],
            FLICKER: metrics[FLICKER],
            CONTRAST: abs(metrics["std"] / self._baseline_std - 1.0),
        }

    def metadata(self, anomaly_id: str) -> dict:
        return {"baseline_std": round(self._baseline_std, 2)} if anomaly_id == CONTRAST else {}
