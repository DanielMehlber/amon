"""Synthetic test video with a precisely known anomaly schedule.

The generated clip uses :data:`BACKGROUND_IMAGE` (an infrared landscape
photograph) as the static scene, with a constant low level of sensor noise
and brightness flicker.  Up to four white text HUD overlays sit at different
positions.  Anomalies are injected at the timestamps in :data:`SCHEDULE` so
integration tests can verify detections against ground truth.  All randomness
is seeded, making the video fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
FPS = 20.0
DURATION = 86.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Infrared landscape photograph in the repository root.
BACKGROUND_IMAGE = Path(__file__).resolve().parent.parent / "infrared-landscape.png"

# Baseline sensor character (always present; calibration learns these levels).
BASELINE_NOISE_SIGMA = 4.0
BASELINE_FLICKER_AMP = 7.0
BASELINE_FLICKER_HZ = 0.8

#: (anomaly key, start seconds, end seconds).  The first 12 s are clean so
#: a calibration duration of up to 12 s observes normal footage only.
SCHEDULE: List[Tuple[str, float, float]] = [
    ("noise", 16.0, 19.0),
    ("flicker", 23.0, 26.0),
    ("contrast", 30.0, 33.0),
    ("hud_text", 37.0, 40.0),
    ("hud_blink_freq", 44.0, 47.0),
    ("hud_blink_stop", 51.0, 54.0),
    ("hud_position", 58.0, 61.0),
    ("hud_size", 65.0, 68.0),
    ("spatial", 72.0, 75.0),
    ("overlap_flicker_noise", 79.0, 82.0),
]

#: Anomaly ID patterns (``*`` = any element) the default detector set is
#: expected to report per schedule entry.
EXPECTED_EVENTS = {
    "noise": "temporal/noise",
    "flicker": "temporal/flicker",
    "contrast": "temporal/contrast",
    "hud_text": "hud/*/text",
    "hud_blink_freq": "hud/*/blink",
    "hud_blink_stop": "hud/*/blink",
    "hud_position": "hud/*/position",
    "hud_size": "hud/*/size",
    "spatial": "spatial/distortion",
    "overlap_flicker_noise": "temporal/flicker",
}

#: Rotation centre for the spatial anomaly — over the bright tree canopy.
WARP_REGION = (110, 40, 290, 180)


@dataclass(frozen=True)
class HudSpec:
    """One text-only HUD overlay."""

    key: str
    text: str
    org: Tuple[int, int]
    scale: float
    blink_hz: float  # 0 = always visible


# Four HUD elements spread across the frame for variance.
HUD_SPECS: Tuple[HudSpec, ...] = (
    HudSpec("cam", "CAM01", (14, 28), 1.0, 0.0),
    HudSpec("rec", "REC", (250, 28), 1.0, 2.0),
    HudSpec("temp", "TEMP22", (118, 28), 0.9, 1.0),
    HudSpec("stat", "STAT01", (14, 220), 0.85, 0.0),
)


class SyntheticVideo:
    """Renders frames of the synthetic scene for any timestamp."""

    def __init__(self, seed: int = 7, background_path: Optional[Union[str, Path]] = None):
        self.seed = seed
        path = Path(background_path) if background_path else BACKGROUND_IMAGE
        self.background = self._load_background(path)

    @staticmethod
    def _load_background(path: Path) -> np.ndarray:
        """Load, centre-crop and resize the infrared landscape to the frame size."""
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"cannot load background image: {path}")

        height, width = gray.shape
        scale = max(WIDTH / width, HEIGHT / height)
        resized = cv2.resize(
            gray,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = (resized.shape[0] - HEIGHT) // 2
        x0 = (resized.shape[1] - WIDTH) // 2
        crop = resized[y0:y0 + HEIGHT, x0:x0 + WIDTH]

        # Leave headroom below pure white so HUD text stands out clearly.
        crop = np.clip(crop.astype(np.float32) * 0.95, 0, 248).astype(np.uint8)
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    def active(self, t: float) -> Set[str]:
        """Anomaly keys scheduled to be active at time ``t``."""
        return {key for key, start, end in SCHEDULE if start <= t < end}

    @staticmethod
    def _square(t: float, hz: float) -> bool:
        """Square wave: True during the 'on' half-period."""
        return int(t * hz * 2) % 2 == 0

    @staticmethod
    def _draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float) -> None:
        """White text overlay — no backing plate or icons."""
        cv2.putText(
            img, text, org, FONT, 0.6 * scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _hud_text(self, spec: HudSpec, active: Set[str]) -> str:
        if spec.key == "cam" and "hud_text" in active:
            return "ERR42"
        return spec.text

    def _hud_org(self, spec: HudSpec, active: Set[str]) -> Tuple[int, int]:
        if spec.key == "cam" and "hud_position" in active:
            return spec.org[0] + 14, spec.org[1] + 10
        return spec.org

    def _hud_scale(self, spec: HudSpec, active: Set[str]) -> float:
        if spec.key == "cam" and "hud_size" in active:
            return spec.scale * 1.35
        return spec.scale

    def _hud_visible(self, spec: HudSpec, t: float, active: Set[str]) -> bool:
        if spec.blink_hz <= 0:
            return True
        if spec.key == "rec" and "hud_blink_stop" in active:
            return True
        hz = 6.0 if (spec.key == "rec" and "hud_blink_freq" in active) else spec.blink_hz
        return self._square(t, hz)

    def _apply_noise(self, img: np.ndarray, sigma: float, frame_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 100_003 + frame_index)
        noise = rng.normal(0.0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def frame(self, t: float, index: int = None) -> np.ndarray:
        """Render the BGR frame for timestamp ``t`` (seconds)."""
        active = self.active(t)
        frame_index = index if index is not None else int(round(t * FPS))
        img = self.background.copy()

        if "spatial" in active:
            cx = (WARP_REGION[0] + WARP_REGION[2]) / 2.0
            cy = (WARP_REGION[1] + WARP_REGION[3]) / 2.0
            matrix = cv2.getRotationMatrix2D((cx, cy), 5.0, 1.12)
            img = cv2.warpAffine(
                img, matrix, (WIDTH, HEIGHT),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

        for spec in HUD_SPECS:
            if self._hud_visible(spec, t, active):
                self._draw_text(
                    img,
                    self._hud_text(spec, active),
                    self._hud_org(spec, active),
                    self._hud_scale(spec, active),
                )

        if "contrast" in active:
            mean = img.mean()
            img = np.clip(mean + 0.45 * (img.astype(np.float32) - mean), 0, 255).astype(np.uint8)

        # Constant baseline flicker, plus stronger oscillation during anomalies.
        flicker = BASELINE_FLICKER_AMP * np.sin(2 * np.pi * BASELINE_FLICKER_HZ * t)
        if "flicker" in active or "overlap_flicker_noise" in active:
            flicker += 38.0 if self._square(t, 5.0) else -38.0
        img = np.clip(img.astype(np.float32) + flicker, 0, 255).astype(np.uint8)

        # Constant baseline noise, plus burst noise during anomalies.
        img = self._apply_noise(img, BASELINE_NOISE_SIGMA, frame_index)
        if "noise" in active or "overlap_flicker_noise" in active:
            img = self._apply_noise(img, 22.0, frame_index + 1_000_000)
        return img


def write_video(path: str, seed: int = 7, duration: float = DURATION, fps: float = FPS) -> str:
    """Write the synthetic video to ``path`` (.avi uses MJPG, .mp4 uses mp4v)."""
    fourcc_name = "mp4v" if str(path).lower().endswith(".mp4") else "MJPG"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    video = SyntheticVideo(seed)
    try:
        for i in range(int(round(duration * fps))):
            writer.write(video.frame(i / fps, index=i))
    finally:
        writer.release()
    return str(path)
