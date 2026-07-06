"""Video file source based on OpenCV's ``VideoCapture``."""

from __future__ import annotations

import time
from typing import Iterator

import cv2

from amon.model import Frame
from amon.sources import SourceError, VideoSource


class VideoFileSource(VideoSource):
    """Reads frames sequentially from a video file.

    Config keys:

    - ``path`` (required): path to the video file.
    - ``realtime`` (default ``false``): pace playback to the file's FPS so
      the pipeline behaves like a live stream.  When false, frames are
      processed as fast as possible (useful for tests and re-analysis).
    - ``fps``: override for files with missing/broken FPS metadata.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        path = self.config.get("path")
        if not path:
            raise SourceError("video_source config requires a 'path'")
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise SourceError(f"cannot open video file: {path}")
        self._fps = float(
            self.config.get("fps") or self._capture.get(cv2.CAP_PROP_FPS) or 0
        )
        if self._fps <= 0:
            raise SourceError(
                f"cannot determine FPS of {path}; set 'fps' in the config"
            )
        self._realtime = bool(self.config.get("realtime", False))

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Frame]:
        index = 0
        wall_start = time.monotonic()
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            timestamp = index / self._fps
            if self._realtime:
                lag = timestamp - (time.monotonic() - wall_start)
                if lag > 0:
                    time.sleep(lag)
            yield Frame(index=index, timestamp=timestamp, image=image)
            index += 1

    def close(self) -> None:
        self._capture.release()
