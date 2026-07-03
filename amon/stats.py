"""Robust statistics used for automatic threshold calibration."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def robust_threshold(
    samples: Sequence[float],
    sigma_k: float = 8.0,
    floor: float = 1e-3,
    headroom: float = 1.5,
) -> float:
    """Derive a detection threshold from calibration intensity samples.

    The threshold is the maximum of three candidates so that clean footage
    never triggers detections:

    - ``median + sigma_k * robust_std`` where the robust standard deviation
      is estimated via the median absolute deviation (MAD), making the
      estimate insensitive to occasional calibration glitches;
    - ``headroom * max(samples)``, guaranteeing clearance above everything
      observed during calibration;
    - ``floor``, a detector-supplied minimum that encodes the metric's
      physical scale (used when calibration samples are all ~zero).
    """
    if len(samples) == 0:
        return float(floor)
    arr = np.asarray(samples, dtype=float)
    median = float(np.median(arr))
    robust_std = float(np.median(np.abs(arr - median))) * 1.4826
    return float(max(floor, median + sigma_k * robust_std, headroom * float(arr.max())))
