"""Video source plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from amon.model import Frame


class SourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource(ABC):
    """Interface every video source plugin implements.

    Sources are instantiated from the configuration file with a single
    ``config`` dict and only need to yield :class:`~amon.model.Frame`
    objects; all generic logic (calibration, detection, persistence) is
    handled by the framework.
    """

    def __init__(self, config: dict = None):
        self.config = dict(config or {})

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second of the stream."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames until the stream ends or the source is closed."""

    def close(self) -> None:
        """Release any underlying resources."""

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def source_label(spec: dict) -> str:
    """Human-readable source identifier stored on the session row."""
    cfg = spec.get("config") or {}
    if "path" in cfg:
        return str(cfg["path"])
    if "device" in cfg:
        return str(cfg["device"])
    return str(spec.get("class", ""))
