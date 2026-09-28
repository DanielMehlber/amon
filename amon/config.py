"""Configuration loading with sensible defaults.

The whole framework is driven by a single YAML file (see ``config.yaml`` in
the repository root).  Every value has a default so a minimal config only
needs to name a video file.  Detectors and video sources are specified as
``{"class": "<dotted path>", "config": {...}}`` plugin specs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import yaml

DEFAULTS: dict = {
    "session_name": "session",
    "data_dir": "data",
    "video_source": {
        "class": "amon.sources.file.VideoFileSource",
        "config": {},
    },
    "calibration": {
        "duration_seconds": 10.0,
    },
    "detectors": [
        {"class": "amon.detectors.temporal.TemporalDetector", "config": {}},
        {"class": "amon.detectors.hud.HudDetector", "config": {}},
        {"class": "amon.detectors.spatial.SpatialDetector", "config": {}},
    ],
    "aggregation": {
        # An event ends once intensity stayed below threshold this long.
        "cooldown_seconds": 1.0,
        # Events shorter than this are considered glitches and dropped.
        "min_duration_seconds": 0.5,
        # Suppressors keep suppressing this long after they subside, so
        # windowed metrics can drain a primary anomaly's side-effects.
        "suppression_linger_seconds": 2.5,
        "max_timeline_points": 2000,
        # No default exclusion rules — every suppression must be listed in the
        # session YAML (deep-merge would otherwise keep stale default keys).
        "suppresses": {},
    },
    "media": {
        "max_clip_seconds": 6.0,  # long events are clipped to this length
        "lead_seconds": 1.0,  # context recorded before the event start
        "gif_max_fps": 10.0,
        # Bound background jobs so a stalled worker cannot OOM a long run.
        # Each event may carry evidence frames — keep this small (default 8).
        "queue_maxsize": 8,
    },
    # Applied to every frame after the video source (any source type).
    "preprocessing": {
        "scale": 100,       # percent of input size; 100 = no resize
        "rotate": 0,        # degrees clockwise
        "brightness": 0,    # additive [-255, 255]
        "contrast": 1.0,    # multiplicative gain
    },
    "report": {
        "port": 5006,
        # Serve Panel/Bokeh assets locally — required for air-gapped use.
        "offline": True,
        # Bind address for the report server (use "0.0.0.0" on isolated LANs).
        "address": "127.0.0.1",
    },
    "export": {
        "format": "html",
    },
    # Diagnostic logging for field machines (no debugger).
    # Per-run file: <logging.dir>/<session_id>.log
    # Console and file levels are independent (logger uses the quieter of the two).
    "logging": {
        "console": True,
        "console_level": "INFO",   # anomalies / session lifecycle on stderr
        "file_level": "DEBUG",     # full detection trace in the log file
        "dir": "logs",
        "max_bytes": 50 * 1024 * 1024,  # rotate before multi-day DEBUG fills disk
        "backup_count": 5,
    },
}


def merge_defaults(config: dict) -> dict:
    """Return ``config`` deep-merged over the framework defaults."""
    return _deep_merge(copy.deepcopy(DEFAULTS), config or {})


def load_config(path: Union[str, Path]) -> dict:
    """Load a YAML config file and merge it with the defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        return merge_defaults(yaml.safe_load(fh) or {})


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
