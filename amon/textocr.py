"""
Tiny offline text recognizer for HUD overlays.

HUD text is typically rendered in a fixed OSD-style font on a plain
background, so full OCR is unnecessary. Characters are segmented via
connected components, normalized to a fixed size and matched against
glyph templates rendered from a TrueType font (default: bundled
VCR OSD Mono). The recognizer covers ``A-Z`` and ``0-9``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GLYPH_SIZE = (24, 32)  # (width, height) of normalised glyphs
CONFUSABLE = str.maketrans({"O": "0", "I": "1"})

FONTS_DIR = Path(__file__).with_name("fonts")
DEFAULT_GLYPH_FONT = "VCR_OSD_MONO_1.001.ttf"


def resolve_glyph_font(font: Optional[str] = None) -> Path:
    """Resolve a glyph-font path from config or the bundled default.

    Relative names are looked up under ``amon/fonts/`` first, then the
    current working directory. Absolute paths are used as-is.
    """
    name = (font or DEFAULT_GLYPH_FONT).strip() or DEFAULT_GLYPH_FONT
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    bundled = FONTS_DIR / name
    if bundled.is_file():
        return bundled.resolve()
    cwd = Path.cwd() / name
    if cwd.is_file():
        return cwd.resolve()
    raise FileNotFoundError(
        f"glyph font not found: {name!r} (looked in {FONTS_DIR} and {Path.cwd()})"
    )


def _normalise(binary: np.ndarray) -> Tuple[np.ndarray, float]:
    """Tight-crop a binary glyph; return the canonical image and aspect ratio."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        blank = np.zeros(GLYPH_SIZE[::-1], np.float32)
        return blank, 1.0
    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    aspect = crop.shape[1] / crop.shape[0]
    return (
        cv2.resize(crop.astype(np.float32), GLYPH_SIZE, interpolation=cv2.INTER_AREA),
        aspect,
    )


def _create_glyph_templates(font_path: Path) -> Dict[str, Tuple[np.ndarray, float]]:
    """Render charset templates from a TrueType / OpenType font."""
    font = ImageFont.truetype(str(font_path), size=48)
    templates: Dict[str, Tuple[np.ndarray, float]] = {}
    for char in CHARSET:
        img = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Anchor near the top-left; tight-crop in ``_normalise`` removes padding.
        draw.text((8, 8), char, fill=255, font=font)
        templates[char] = _normalise(np.asarray(img) > 128)
    return templates


@lru_cache(maxsize=8)
def _templates_for(font_path: str) -> Dict[str, Tuple[np.ndarray, float]]:
    return _create_glyph_templates(Path(font_path))


def _match_char(
    glyph: np.ndarray,
    aspect: float,
    templates: Dict[str, Tuple[np.ndarray, float]],
) -> str:
    """Best charset character by correlation, weighted by aspect similarity.

    The aspect-ratio weight disambiguates glyph pairs that look alike once
    stretched to the canonical size (e.g. ``0`` vs ``O``, ``1`` vs ``I``).
    """
    best_char, best_score = "?", -np.inf
    for char, (tmpl, tmpl_aspect) in templates.items():
        correlation = float(cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        ratio = min(aspect, tmpl_aspect) / max(aspect, tmpl_aspect)
        score = correlation * ratio

        if score > best_score:
            best_char, best_score = char, score

    return best_char


def read_text(
    gray: np.ndarray,
    threshold: Optional[int] = None,
    *,
    min_glyph_height: int = 8,
    min_glyph_area: int = 20,
    max_glyph_height: int = 64,
    max_glyph_width: int = 64,
    glyph_font: Optional[str] = None,
) -> str:
    """Recognise bright text in a grayscale crop, including word spaces.

    Without an explicit ``threshold`` the text/background split is found
    with Otsu's method, which keeps anti-aliased stroke edges intact.

    Connected components outside ``[min_glyph_height, max_glyph_height]``
    or wider than ``max_glyph_width``, or with fewer than ``min_glyph_area``
    bright pixels, are ignored — this rejects tiny IR speckles and large
    non-text bright regions that would otherwise match letter templates.

    ``glyph_font`` selects the TrueType file used for templates (see
    :func:`resolve_glyph_font`).
    """
    templates = _templates_for(str(resolve_glyph_font(glyph_font)))

    # Find the threshold for the text/background split using Otsu's method.
    if threshold is None:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create a binary image of the text/background split.
    binary = (gray > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    min_h = max(1, int(min_glyph_height))
    max_h = max(min_h, int(max_glyph_height))
    max_w = max(1, int(max_glyph_width))
    min_area = max(1, int(min_glyph_area))

    # Find the bounding boxes of glyph-sized components only.
    boxes: List[tuple] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if min_h <= h <= max_h and w <= max_w and area >= min_area:
            boxes.append((x, y, w, h))

    if not boxes:
        return ""

    # Sort the bounding boxes by the x-coordinate.
    boxes.sort(key=lambda b: b[0])

    # Calculate the median width of the bounding boxes.
    median_width = float(np.median([b[2] for b in boxes]))

    # Iterate over the bounding boxes and extract the text.
    text, prev_right = "", None
    for x, y, w, h in boxes:
        if prev_right is not None and (x - prev_right) > 0.6 * median_width:
            text += " "
        glyph, aspect = _normalise(binary[y : y + h, x : x + w] > 0)
        text += _match_char(glyph, aspect, templates)
        prev_right = x + w

    # Account for confusable characters by canonicalizing them.
    return text.translate(CONFUSABLE)


def slugify(text: str) -> str:
    """Lower-case alphanumeric identifier derived from recognised text."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def levenshtein_norm(a: str, b: str) -> float:
    """Levenshtein distance normalised to [0, 1] by the longer string."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))
