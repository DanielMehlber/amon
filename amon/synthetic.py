"""Synthetic test video with a precisely known anomaly schedule.

The generated clip shows a static textured scene with two HUD overlays:

- a static label ``CAM 01`` (top-left), and
- a ``REC`` indicator (top-right) blinking at :data:`BLINK_HZ`.

Anomalies are injected at the timestamps in :data:`SCHEDULE` so integration
tests can verify detections against ground truth.  All randomness is
seeded, making the video fully deterministic.
"""
from __future__ import annotations

from typing import List, Set, Tuple

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
FPS = 20.0
DURATION = 86.0
BLINK_HZ = 2.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

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
#: expected to report per schedule entry.  HUD element names come from OCR,
#: so tests match on the anomaly aspect rather than the exact element ID.
#: In the overlap window flicker must suppress the noise detection.
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

#: Region distorted by the "spatial" anomaly (x0, y0, x1, y1).
WARP_REGION = (195, 135, 275, 205)


class SyntheticVideo:
    """Renders frames of the synthetic scene for any timestamp."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.background = self._make_background(seed)

    @staticmethod
    def _make_background(seed: int) -> np.ndarray:
        """Textured static scene, brightness capped below HUD whites."""
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, 255, (HEIGHT, WIDTH)).astype(np.uint8)
        base = cv2.GaussianBlur(noise, (0, 0), 8)
        base = cv2.normalize(base, None, 55, 165, cv2.NORM_MINMAX)

        # Deterministic shapes provide corners for feature tracking.
        cv2.rectangle(base, (40, 60), (100, 110), 60, -1)
        cv2.rectangle(base, (46, 66), (94, 104), 150, 2)
        cv2.circle(base, (160, 170), 25, 70, -1)
        cv2.circle(base, (160, 170), 15, 160, 2)
        cv2.rectangle(base, (250, 40), (300, 90), 145, -1)
        cv2.line(base, (250, 40), (300, 90), 60, 2)

        # Grid texture inside the warp region so distortions are visible.
        x0, y0, x1, y1 = WARP_REGION
        for gy in range(y0 + 6, y1, 12):
            cv2.line(base, (x0, gy), (x1, gy), 170, 1)
        for gx in range(x0 + 6, x1, 12):
            cv2.line(base, (gx, y0), (gx, y1), 60, 1)

        # Slight per-channel tint so the scene is not pure grayscale.
        b = base
        g = np.clip(base.astype(np.int16) - 8, 0, 255).astype(np.uint8)
        r = np.clip(base.astype(np.int16) - 15, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    def active(self, t: float) -> Set[str]:
        """Anomaly keys scheduled to be active at time ``t``."""
        return {key for key, start, end in SCHEDULE if start <= t < end}

    @staticmethod
    def _square(t: float, hz: float) -> bool:
        """Square wave: True during the 'on' half-period."""
        return int(t * hz * 2) % 2 == 0

    @staticmethod
    def _draw_label(img: np.ndarray, text: str, org: Tuple[int, int], scale: float) -> None:
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55 * scale, 1)
        x, y = org
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), (25, 25, 25), -1)
        cv2.putText(img, text, (x, y), FONT, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA)

    def frame(self, t: float, index: int = None) -> np.ndarray:
        """Render the BGR frame for timestamp ``t`` (seconds)."""
        active = self.active(t)
        img = self.background.copy()

        if "spatial" in active:
            x0, y0, x1, y1 = WARP_REGION
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            map_x = xs + 6.0 * np.sin((ys - y0) / 6.0)
            map_y = ys + 6.0 * np.cos((xs - x0) / 6.0)
            img[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        # --- HUD overlays -------------------------------------------------
        label_text = "ERR 42" if "hud_text" in active else "CAM 01"
        label_org = (26, 30) if "hud_position" in active else (14, 22)
        label_scale = 1.35 if "hud_size" in active else 1.0
        self._draw_label(img, label_text, label_org, label_scale)

        blink_hz = 6.0 if "hud_blink_freq" in active else BLINK_HZ
        rec_on = True if "hud_blink_stop" in active else self._square(t, blink_hz)
        if rec_on:
            cv2.circle(img, (WIDTH - 78, 16), 4, (255, 255, 255), -1)
            cv2.putText(img, "REC", (WIDTH - 68, 21), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- global disturbances ------------------------------------------
        if "contrast" in active:
            mean = img.mean()
            img = np.clip(mean + 0.45 * (img.astype(np.float32) - mean), 0, 255).astype(np.uint8)
        if "flicker" in active or "overlap_flicker_noise" in active:
            offset = 40 if self._square(t, 5.0) else -40
            img = np.clip(img.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        if "noise" in active or "overlap_flicker_noise" in active:
            frame_index = index if index is not None else int(round(t * FPS))
            rng = np.random.default_rng(self.seed * 100003 + frame_index)
            gauss = rng.normal(0.0, 25.0, img.shape)
            img = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
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
