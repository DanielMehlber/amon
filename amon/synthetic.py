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
from PIL import Image, ImageDraw, ImageFont

from amon.textocr import resolve_glyph_font

WIDTH, HEIGHT = 320, 240
FPS = 20.0
DURATION = 114.0

#: Infrared landscape photograph shipped next to this module (package data).
BACKGROUND_IMAGE = Path(__file__).resolve().with_name("infrared-landscape.png")

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
    ("hud_new", 69.0, 71.0),
    ("spatial", 72.0, 75.0),
    ("overlap_flicker_noise", 79.0, 82.0),
    # After flicker linger (~2.5s): parallel changes on different HUD elements.
    ("hud_parallel_text", 87.0, 90.0),
    ("hud_parallel_position", 87.0, 90.0),
    # Appear as ALERT, rewrite in place to 1000, then vanish — one /new event.
    ("hud_new_mutate", 91.0, 96.0),
    # Gap so mutate tracks TTL out, then: 1000→2000 steps, blink, vanish.
    ("hud_new_cycle", 98.0, 107.0),
    # Non-text icons: centre crosshair + side dot → symbol-N, not letter slugs.
    ("hud_new_symbols", 109.0, 113.0),
    # Still above threshold when the file ends — must finalize as completed.
    ("hud_text_until_end", 110.0, DURATION),
]

#: Anomaly ID patterns the default detector set is expected to report per
#: schedule entry.  ``*`` matches any path span (see :mod:`amon.aggregate`).
EXPECTED_EVENTS = {
    "noise": "temporal/noise",
    "flicker": "temporal/flicker",
    "contrast": "temporal/contrast",
    "hud_text": "hud/*/text",
    "hud_blink_freq": "hud/*/blink",
    "hud_blink_stop": "hud/*/blink",
    "hud_position": "hud/*/position",
    "hud_size": "hud/*/size",
    "hud_new": "hud/*/new",
    "spatial": "spatial/distortion",
    "overlap_flicker_noise": "temporal/flicker",
    "hud_parallel_text": "hud/*/text",
    "hud_parallel_position": "hud/*/position",
    "hud_new_mutate": "hud/alert/new",
    "hud_new_cycle": "hud/1000/new",
    "hud_new_symbols": "hud/symbol-*/new",
    "hud_text_until_end": "hud/*/text",
}

#: How many distinct events are expected for a schedule key (default 1).
EXPECTED_EVENT_COUNTS = {
    "hud_new": 2,  # ALERT + WARN appear together → two hud/<slug>/new events
    "hud_new_symbols": 2,  # crosshair + filled dot
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


# Baseline HUD overlays: multi-glyph labels plus lone letters (real HUDs often
# show a single status glyph; ``S`` in particular used to fall through to
# symbol-N when connected components fragmented the ink).
HUD_SPECS: Tuple[HudSpec, ...] = (
    HudSpec("cam", "CAM01", (14, 28), 1.0, 0.0),
    HudSpec("rec", "REC", (250, 28), 1.0, 2.0),
    HudSpec("temp", "TEMP22", (118, 28), 0.9, 1.0),
    HudSpec("stat", "STAT01", (14, 220), 1.0, 0.0),
    HudSpec("letter_s", "S", (60, 100), 1.0, 0.0),
    HudSpec("letter_a", "A", (280, 100), 1.0, 0.0),
    HudSpec("letter_x", "X", (160, 220), 1.0, 0.0),
)

#: Overlays that appear only during the ``hud_new`` schedule window.
NEW_HUDS: Tuple[HudSpec, ...] = (
    HudSpec("alert", "ALERT", (200, 220), 1.0, 0.0),
    HudSpec("warn", "WARN", (200, 120), 1.0, 0.0),
)

#: Single overlay that rewrites its glyphs mid-lifetime (``hud_new_mutate``).
MUTATE_HUD = HudSpec("mutate", "ALERT", (200, 160), 1.0, 0.0)
MUTATE_FLIP_AT = 93.5  # seconds — switch ALERT → 1000 at the same org
MUTATE_TEXT_AFTER = "1000"

#: Runtime overlay: 1000 → 2000 in 1s steps, then blink, then vanish.
#: Placed well away from MUTATE_HUD so centroid tracks cannot collide.
CYCLE_HUD = HudSpec("cycle", "1000", (60, 140), 1.0, 0.0)
CYCLE_START = 98.0
CYCLE_STEP_SECONDS = 5.0  # time to climb 1000 → 2000
CYCLE_BLINK_START = CYCLE_START + CYCLE_STEP_SECONDS  # 103.0
CYCLE_BLINK_HZ = 2.0
CYCLE_TEXT_START = 1000
CYCLE_TEXT_END = 2000
CYCLE_TEXT_STEP = 200  # 1000, 1200, …, 2000

#: Non-text icons for ``hud_new_symbols`` (must score below glyph match gate).
SYMBOL_CROSSHAIR_CENTER = (WIDTH // 2, HEIGHT // 2)  # frame centre
SYMBOL_DOT_CENTER = (280, 170)  # clear of calibrated HUD corners
SYMBOL_START = 109.0
SYMBOL_END = 113.0


def cycle_hud_text(t: float) -> str:
    """Numeric label for the cycle overlay at time ``t`` (frozen after steps)."""
    elapsed = max(0.0, t - CYCLE_START)
    step_index = min(
        int((CYCLE_TEXT_END - CYCLE_TEXT_START) / CYCLE_TEXT_STEP),
        int(elapsed),  # one step per second
    )
    return str(CYCLE_TEXT_START + step_index * CYCLE_TEXT_STEP)


def cycle_hud_visible(t: float) -> bool:
    """Visible during step phase; square-wave blink afterward until schedule end."""
    if t < CYCLE_BLINK_START:
        return True
    return SyntheticVideo._square(t, CYCLE_BLINK_HZ)


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
        """White HUD text using the same TrueType font as glyph matching.

        Characters are blitted with a fixed 1 px ink gap so that:
        - OCR can still segment individual glyphs, and
        - the width-relative merge kernel (0.03 × width → ~10 px @ 320)
          joins them into one HUD element on the synthetic frame.
        """
        font_size = max(10, int(round(18 * scale)))
        font = ImageFont.truetype(str(resolve_glyph_font()), size=font_size)
        # OpenCV ``org`` is the baseline; PIL places text at the top-left.
        top = max(0, org[1] - font_size)
        canvas = Image.new("L", (img.shape[1], img.shape[0]), 0)
        # Render far apart first, then pack by ink bounding boxes.
        glyph_ink: List[Tuple[Image.Image, int, int]] = []
        for ch in text:
            # Generous cell so the glyph is never clipped.
            cell = Image.new("L", (font_size * 2, font_size * 2), 0)
            ImageDraw.Draw(cell).text((0, 0), ch, fill=255, font=font)
            bbox = cell.getbbox()
            if bbox is None:
                continue
            cropped = cell.crop(bbox)
            glyph_ink.append((cropped, bbox[2] - bbox[0], bbox[3] - bbox[1]))
        x = org[0]
        for cropped, gw, gh in glyph_ink:
            canvas.paste(cropped, (x, top))
            x += gw + 1  # 1 px gap between ink boxes
        mask = np.asarray(canvas)
        img[mask > 0] = (255, 255, 255)

    @staticmethod
    def _draw_crosshair(
        img: np.ndarray,
        center: Tuple[int, int],
        arm: Optional[int] = None,
        thickness: int = 2,
    ) -> None:
        """White crosshair (non-text icon) centred on ``center``."""
        if arm is None:
            # Keep the icon inside width-relative glyph size gates (~64 px @ 1200).
            arm = max(4, WIDTH // 50)
        cx, cy = center
        color = (255, 255, 255)
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), color, thickness)
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), color, thickness)

    @staticmethod
    def _draw_dot(
        img: np.ndarray, center: Tuple[int, int], radius: Optional[int] = None
    ) -> None:
        """Filled white disk (non-text icon)."""
        if radius is None:
            radius = max(3, WIDTH // 60)
        cv2.circle(img, center, radius, (255, 255, 255), thickness=-1)

    def _hud_text(self, spec: HudSpec, active: Set[str]) -> str:
        if spec.key == "cam" and (
            "hud_text" in active
            or "hud_parallel_text" in active
            or "hud_text_until_end" in active
        ):
            return "ERR42"
        return spec.text

    def _hud_org(self, spec: HudSpec, active: Set[str]) -> Tuple[int, int]:
        # Shift stays inside the width-relative search margin (~20 px @ 1200).
        shift = max(2, round(WIDTH * 20 / 1200 * 0.6))
        if spec.key == "cam" and "hud_position" in active:
            return spec.org[0] + shift, spec.org[1] + shift
        if spec.key == "temp" and "hud_parallel_position" in active:
            return spec.org[0] + shift, spec.org[1] + shift
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

        if "hud_new" in active:
            for spec in NEW_HUDS:
                self._draw_text(img, spec.text, spec.org, spec.scale)

        if "hud_new_mutate" in active:
            text = (
                MUTATE_TEXT_AFTER if t >= MUTATE_FLIP_AT else MUTATE_HUD.text
            )
            self._draw_text(img, text, MUTATE_HUD.org, MUTATE_HUD.scale)

        if "hud_new_cycle" in active and cycle_hud_visible(t):
            self._draw_text(
                img, cycle_hud_text(t), CYCLE_HUD.org, CYCLE_HUD.scale
            )

        if "hud_new_symbols" in active:
            self._draw_crosshair(img, SYMBOL_CROSSHAIR_CENTER)
            self._draw_dot(img, SYMBOL_DOT_CENTER)

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
