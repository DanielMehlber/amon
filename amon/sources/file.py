"""Video file source based on OpenCV's ``VideoCapture``."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2

from amon.model import Frame
from amon.sources import SourceError, VideoSource
from amon.sources.frame_buffer import OverloadMonitor


class VideoFileSource(VideoSource):
    """Reads frames sequentially from a video file.

    Config keys:

    - ``path`` (required): path to the video file.
    - ``realtime`` (default ``false``): pace playback to the delivered FPS so
      the pipeline behaves like a live stream.  When false, frames are
      processed as fast as possible (useful for tests and re-analysis).
      When true and the pipeline falls behind, frames are skipped to catch
      up (same idea as the live-stream dropping buffer).
    - ``fps``: override for files with missing/broken FPS metadata.
    - ``processing_fps``: optional maximum frame rate delivered to the
      pipeline (e.g. ``21``).  Extra frames from the file are skipped so
      the session never sees more than this many frames per second of
      video time.  Defaults to the file FPS when omitted.
    - ``warn_on_dropped_frames`` (default ``true``): when ``realtime`` is
      on, log a warning if catch-up skips are sustained because the
      pipeline cannot keep up.  Set ``false`` to silence that warning
      (frames are still skipped).  Has no effect when ``realtime`` is
      false — ``processing_fps`` skips are intentional, not overload.
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
        self._warn_on_dropped_frames = bool(
            self.config.get("warn_on_dropped_frames", True)
        )
        self._min_interval = 1.0 / self._output_fps
        self._frame_size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )

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
        dropped = 0
        width, height = self._frame_size
        overload = OverloadMonitor(
            width=width,
            height=height,
            source_fps=self._file_fps,
            enabled=self._warn_on_dropped_frames and self._realtime,
        )
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            if (width <= 0 or height <= 0) and image is not None:
                height, width = image.shape[:2]
                overload = OverloadMonitor(
                    width=width,
                    height=height,
                    source_fps=self._file_fps,
                    enabled=self._warn_on_dropped_frames and self._realtime,
                )
            file_t = file_index / self._file_fps
            file_index += 1

            # Drop frames that would push the delivery rate above processing_fps.
            if (
                last_emit_t is not None
                and (file_t - last_emit_t) < self._min_interval
            ):
                continue

            if self._realtime:
                # When the pipeline lags wall-clock playback, skip ahead
                # (overload drops) until file time catches elapsed wall time.
                while file_t < (time.monotonic() - wall_start):
                    dropped += 1
                    ok, image = self._capture.read()
                    if not ok:
                        return
                    file_t = file_index / self._file_fps
                    file_index += 1
                # Catch-up may land on a frame that processing_fps would skip.
                if (
                    last_emit_t is not None
                    and (file_t - last_emit_t) < self._min_interval
                ):
                    continue
                lag = file_t - (time.monotonic() - wall_start)
                if lag > 0:
                    time.sleep(lag)

            overload.note_emit(dropped)
            yield Frame(index=emit_index, timestamp=file_t, image=image)
            last_emit_t = file_t
            emit_index += 1

    def close(self) -> None:
        self._capture.release()
