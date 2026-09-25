"""Video file source based on OpenCV's ``VideoCapture``."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2

from amon.model import Frame
from amon.sources import SourceError, VideoSource


class VideoFileSource(VideoSource):
    """Reads frames sequentially from a video file.

    Config keys:

    - ``path`` (required): path to the video file.
    - ``realtime`` (default ``false``): pace playback to the delivered FPS so
      the pipeline behaves like a live stream.  When false, frames are
      processed as fast as possible (useful for tests and re-analysis).
    - ``fps``: override for files with missing/broken FPS metadata.
    - ``processing_fps``: optional maximum frame rate delivered to the
      pipeline (e.g. ``21``).  Extra frames from the file are skipped so
      the session never sees more than this many frames per second of
      video time.  Defaults to the file FPS when omitted.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        path = self.config.get("path")
        if not path:
            raise SourceError("video_source config requires a 'path'")
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise SourceError(f"cannot open video file: {path}")
        self._file_fps = float(
            self.config.get("fps") or self._capture.get(cv2.CAP_PROP_FPS) or 0
        )
        if self._file_fps <= 0:
            raise SourceError(
                f"cannot determine FPS of {path}; set 'fps' in the config"
            )

        processing_fps = self.config.get("processing_fps")
        if processing_fps is None:
            self._output_fps = self._file_fps
        else:
            self._output_fps = float(processing_fps)
            if self._output_fps <= 0:
                raise SourceError("processing_fps must be positive")
            # Cap cannot increase rate above what the file contains.
            self._output_fps = min(self._output_fps, self._file_fps)

        self._realtime = bool(self.config.get("realtime", False))
        self._min_interval = 1.0 / self._output_fps

    @property
    def fps(self) -> float:
        """Frame rate seen by the monitoring pipeline (after any cap)."""
        return self._output_fps

    @property
    def file_fps(self) -> float:
        """Native frame rate of the video file."""
        return self._file_fps

    def frames(self) -> Iterator[Frame]:
        emit_index = 0
        file_index = 0
        last_emit_t: Optional[float] = None
        wall_start = time.monotonic()
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            file_t = file_index / self._file_fps
            file_index += 1

            # Drop frames that would push the delivery rate above processing_fps.
            if (
                last_emit_t is not None
                and (file_t - last_emit_t) < self._min_interval
            ):
                continue

            if self._realtime:
                lag = file_t - (time.monotonic() - wall_start)
                if lag > 0:
                    time.sleep(lag)

            yield Frame(index=emit_index, timestamp=file_t, image=image)
            last_emit_t = file_t
            emit_index += 1

    def close(self) -> None:
        self._capture.release()
