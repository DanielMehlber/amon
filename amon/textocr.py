"""Tiny offline text recognizer for HUD overlays.

HUD text is typically rendered in a clean sans-serif font on a plain
background, so full OCR is unnecessary.  Characters are segmented via
connected components, normalised to a fixed size and matched against
glyph templates rendered with OpenCV's Hershey font.  The recognizer
covers ``A-Z`` and ``0-9`` which suffices for status overlays; extend
:data:`CHARSET` for richer HUDs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GLYPH_SIZE = (24, 32)  # (width, height) of normalised glyphs

#: Small HUD glyphs cannot reliably be told apart for these pairs, so they
#: are canonicalised.  Recognised text stays stable across frames, which is
#: all the change detection needs.
CONFUSABLE = str.maketrans({"O": "0", "I": "1"})


def _normalise(binary: np.ndarray) -> Tuple[np.ndarray, float]:
    """Tight-crop a binary glyph; returns (canonical image, aspect ratio)."""
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    aspect = crop.shape[1] / crop.shape[0]
    return cv2.resize(crop.astype(np.float32), GLYPH_SIZE, interpolation=cv2.INTER_AREA), aspect


def _glyph_templates() -> Dict[str, Tuple[np.ndarray, float]]:
    templates = {}
    for char in CHARSET:
        canvas = np.zeros((64, 48), np.uint8)
        cv2.putText(canvas, char, (4, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 2, cv2.LINE_AA)
        templates[char] = _normalise(canvas > 128)
    return templates


_TEMPLATES = _glyph_templates()


def _match(glyph: np.ndarray, aspect: float) -> str:
    """Best charset character by correlation, weighted by aspect similarity.

    The aspect-ratio weight disambiguates glyph pairs that look alike once
    stretched to the canonical size (e.g. ``0`` vs ``O``, ``1`` vs ``I``).
    """
    best_char, best_score = "?", -np.inf
    for char, (tmpl, tmpl_aspect) in _TEMPLATES.items():
        correlation = float(cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        ratio = min(aspect, tmpl_aspect) / max(aspect, tmpl_aspect)
        score = correlation * ratio
        if score > best_score:
            best_char, best_score = char, score
    return best_char


def read_text(gray: np.ndarray, threshold: Optional[int] = None) -> str:
    """Recognise bright text in a grayscale crop, including word spaces.

    Without an explicit ``threshold`` the text/background split is found
    with Otsu's method, which keeps anti-aliased stroke edges intact.
    """
    if threshold is None:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (gray > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes: List[tuple] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= 6 and h >= 5:
            boxes.append((x, y, w, h))
    if not boxes:
        return ""
    boxes.sort(key=lambda b: b[0])

    median_width = float(np.median([b[2] for b in boxes]))
    text, prev_right = "", None
    for x, y, w, h in boxes:
        if prev_right is not None and (x - prev_right) > 0.6 * median_width:
            text += " "
        glyph, aspect = _normalise(binary[y:y + h, x:x + w] > 0)
        text += _match(glyph, aspect)
        prev_right = x + w
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
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / max(len(a), len(b))
