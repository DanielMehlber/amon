"""Configuration-driven plugin loading.

Detectors and video sources are referenced in the configuration file by
their dotted class path, e.g. ``amon.detectors.temporal.TemporalDetector``.
New plugins therefore require no changes to the core application: implement
the interface anywhere on the Python path and reference it in the config.
"""

from __future__ import annotations

import importlib
from typing import Any


def load_class(dotted_path: str) -> type:
    """Import and return the class referenced by ``dotted_path``."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"'{dotted_path}' is not a valid dotted class path")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"module '{module_path}' has no class '{class_name}'"
        ) from exc


def instantiate(spec: dict) -> Any:
    """Instantiate a plugin from its config spec ``{"class": ..., "config": {...}}``."""
    cls = load_class(spec["class"])
    return cls(spec.get("config") or {})
