"""Convert width-relative detector settings into pixel sizes.

Absolute pixel knobs were historically tuned at about 1200 px frame width.
Config now stores fractions of the *processed* frame width (or of width² for
areas) so the same YAML stays valid after ``preprocessing.scale``.
"""

from __future__ import annotations

import math

#: Reference width used when the original absolute defaults were authored.
REFERENCE_WIDTH_PX = 1200


def of_width(
    fraction: float,
    width: int,
    *,
    minimum: int = 1,
    round_up: bool = False,
) -> int:
    """Return ``fraction * width`` as an integer pixel length."""
    if width <= 0:
        return minimum
    raw = float(fraction) * float(width)
    value = int(math.ceil(raw - 1e-12)) if round_up else int(round(raw))
    return max(minimum, value)


def of_width_sq(fraction: float, width: int, *, minimum: int = 1) -> int:
    """Return ``fraction * width²`` as an integer pixel area."""
    if width <= 0:
        return minimum
    return max(minimum, int(round(float(fraction) * float(width) * float(width))))


def of_width_float(
    fraction: float, width: int, *, minimum: float = 0.0
) -> float:
    """Return ``fraction * width`` as a float (floors, match distances)."""
    if width <= 0:
        return float(minimum)
    return max(float(minimum), float(fraction) * float(width))
