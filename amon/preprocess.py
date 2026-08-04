"""Configurable frame preprocessing applied after the video source.

Ops run in order: rotate → scale → brightness/contrast.  All keys are
optional; omitted or identity values are no-ops.

Config (top-level ``preprocessing`` in the YAML)::

    preprocessing:
      scale: 50          # percent of input size (aspect preserved); 100 = none
      rotate: 90         # degrees clockwise (any float; 90/180/270 use fast path)
      brightness: 0      # additive offset in [-255, 255]
      contrast: 1.0      # multiplicative gain (1.0 = unchanged)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from amon.model import Frame
from amon.sources import SourceError

log = logging.getLogger("amon.preprocess")

Size = Tuple[int, int]


class FramePreprocessor:
    """Applies geometric and photometric transforms to each frame."""

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        self._scale = _parse_scale(cfg.get("scale"))
        self._rotate = _parse_rotate(cfg.get("rotate", 0))
        self._brightness = _parse_brightness(cfg.get("brightness", 0))
        self._contrast = _parse_contrast(cfg.get("contrast", 1.0))
        self.enabled = any(
            (
                self._scale is not None,
                abs(self._rotate) > 1e-9,
                abs(self._brightness) > 1e-9,
                abs(self._contrast - 1.0) > 1e-9,
            )
        )
        if self.enabled:
            log.info(
                "preprocessing: scale=%s rotate=%.1f° brightness=%+.1f contrast=%.2f",
                f"{self._scale:g}%" if self._scale is not None else "100%",
                self._rotate,
                self._brightness,
                self._contrast,
            )

    def __call__(self, frame: Frame) -> Frame:
        if not self.enabled:
            return frame
        image = frame.image
        if abs(self._rotate) > 1e-9:
            image = _rotate_image(image, self._rotate)
        if self._scale is not None:
            image = _scale_image(image, self._scale)
        if abs(self._brightness) > 1e-9 or abs(self._contrast - 1.0) > 1e-9:
            image = _adjust_brightness_contrast(
                image, self._contrast, self._brightness
            )
        return Frame(index=frame.index, timestamp=frame.timestamp, image=image)


def _parse_scale(value) -> Optional[float]:
    if value is None:
        return None
    try:
        percent = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(
            f"preprocessing.scale must be a positive number (percent), got {value!r}"
        ) from exc
    if percent <= 0:
        raise SourceError(f"preprocessing.scale must be positive, got {percent}")
    if abs(percent - 100.0) < 1e-9:
        return None
    return percent


def _parse_rotate(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(
            f"preprocessing.rotate must be a number (degrees), got {value!r}"
        ) from exc


def _parse_brightness(value) -> float:
    try:
        brightness = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(
            f"preprocessing.brightness must be a number in [-255, 255], got {value!r}"
        ) from exc
    if brightness < -255 or brightness > 255:
        raise SourceError(
            f"preprocessing.brightness must be in [-255, 255], got {brightness}"
        )
    return brightness


def _parse_contrast(value) -> float:
    try:
        contrast = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(
            f"preprocessing.contrast must be a non-negative number, got {value!r}"
        ) from exc
    if contrast < 0:
        raise SourceError(f"preprocessing.contrast must be >= 0, got {contrast}")
    return contrast


def _scale_image(image: np.ndarray, percent: float) -> np.ndarray:
    factor = percent / 100.0
    height, width = image.shape[:2]
    size = (
        max(1, int(round(width * factor))),
        max(1, int(round(height * factor))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _rotate_image(image: np.ndarray, degrees_clockwise: float) -> np.ndarray:
    """Rotate clockwise.  Exact 90° steps use a fast path; others warp."""
    # Normalise to (-180, 180] for the fast path, keep full angle for warp.
    normalized = degrees_clockwise % 360.0
    if abs(normalized - 0.0) < 1e-6 or abs(normalized - 360.0) < 1e-6:
        return image
    if abs(normalized - 90.0) < 1e-6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if abs(normalized - 180.0) < 1e-6:
        return cv2.rotate(image, cv2.ROTATE_180)
    if abs(normalized - 270.0) < 1e-6:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # OpenCV's getRotationMatrix2D uses counter-clockwise positive angles.
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -degrees_clockwise, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _adjust_brightness_contrast(
    image: np.ndarray, contrast: float, brightness: float
) -> np.ndarray:
    """``out = contrast * image + brightness``, clipped to uint8."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
