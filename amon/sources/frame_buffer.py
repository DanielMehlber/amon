"""Bounded capture buffer that prefers recent frames over growing backlog.

A live camera keeps producing frames while the monitoring pipeline is busy.
Without a dedicated reader, OpenCV's driver buffer fills and the session
falls further and further behind wall-clock time — or, on some backends,
RAM grows without bound.

:class:`DroppingFrameBuffer` runs ``read()`` on a daemon thread and keeps at
most ``maxsize`` frames (default 1 = always the latest).  When a new frame
arrives and the buffer is full the oldest entry is discarded.  Callers that
cannot keep up therefore see recent imagery instead of an ever-growing queue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

import numpy as np

log = logging.getLogger("amon.sources.frame_buffer")

ReadFn = Callable[[], Tuple[bool, Optional[np.ndarray]]]
BufferedFrame = Tuple[np.ndarray, float]  # image, capture monotonic time


class DroppingFrameBuffer:
    """Continuously read frames into a fixed-size queue; drop oldest on overflow."""

    def __init__(self, read_fn: ReadFn, *, maxsize: int = 1, name: str = "capture"):
        if maxsize < 1:
            raise ValueError("frame buffer maxsize must be >= 1")
        self._read_fn = read_fn
        self._maxsize = int(maxsize)
        self._name = name
        self._queue: Deque[BufferedFrame] = deque()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._pushed = 0
        self._ended = False
        self._error: Optional[BaseException] = None

    @property
    def dropped(self) -> int:
        """Total frames discarded because the consumer was behind."""
        with self._cond:
            return self._dropped

    @property
    def pushed(self) -> int:
        """Total frames accepted from the capture device (including later dropped)."""
        with self._cond:
            return self._pushed

    @property
    def ended(self) -> bool:
        with self._cond:
            return self._ended

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("frame buffer already started")
        self._thread = threading.Thread(
            target=self._run, name=f"amon-{self._name}-reader", daemon=True
        )
        self._thread.start()

    def get(self, timeout: Optional[float] = 1.0) -> Optional[BufferedFrame]:
        """Pop the oldest buffered frame, or ``None`` on timeout / end-of-stream."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._queue and not self._ended and self._error is None:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            if self._error is not None:
                raise RuntimeError(
                    f"capture reader failed: {self._error}"
                ) from self._error
            if self._queue:
                return self._queue.popleft()
            return None

    def close(self) -> None:
        """Stop the reader thread and discard any leftover frames."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._cond:
            self._queue.clear()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, image = self._read_fn()
                if self._stop.is_set():
                    break
                if not ok or image is None:
                    with self._cond:
                        self._ended = True
                        self._cond.notify_all()
                    return
                # OpenCV reuses the Mat backing store — copy before the next read.
                frame = np.ascontiguousarray(image)
                captured_at = time.monotonic()
                with self._cond:
                    if len(self._queue) >= self._maxsize:
                        self._queue.popleft()
                        self._dropped += 1
                    self._queue.append((frame, captured_at))
                    self._pushed += 1
                    self._cond.notify()
        except BaseException as exc:  # surface to consumer; never kill silently
            with self._cond:
                self._error = exc
                self._ended = True
                self._cond.notify_all()


class OverloadMonitor:
    """Emit a rate-limited warning when the pipeline routinely drops frames."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        source_fps: float,
        window_seconds: float = 5.0,
        warn_drop_ratio: float = 0.25,
        warn_interval_seconds: float = 30.0,
        enabled: bool = True,
    ):
        self._width = int(width)
        self._height = int(height)
        self._source_fps = float(source_fps)
        self._window = float(window_seconds)
        self._warn_ratio = float(warn_drop_ratio)
        self._warn_interval = float(warn_interval_seconds)
        self._enabled = bool(enabled)
        self._samples: Deque[Tuple[float, int, int]] = deque()  # t, dropped, emitted
        self._last_warn: Optional[float] = None
        self._last_dropped = 0
        self._emitted = 0

    def note_emit(self, dropped_total: int) -> None:
        """Record that one frame was delivered; ``dropped_total`` is cumulative."""
        if not self._enabled:
            return
        now = time.monotonic()
        self._emitted += 1
        self._samples.append((now, dropped_total, self._emitted))
        while self._samples and now - self._samples[0][0] > self._window:
            self._samples.popleft()
        if len(self._samples) < 2:
            return
        if (
            self._last_warn is not None
            and now - self._last_warn < self._warn_interval
        ):
            return

        t0, d0, e0 = self._samples[0]
        t1, d1, e1 = self._samples[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return
        window_drops = d1 - d0
        window_emits = e1 - e0
        considered = window_drops + window_emits
        if considered <= 0:
            return
        ratio = window_drops / considered
        # Also require a meaningful absolute rate so a single blip is quiet.
        drop_fps = window_drops / elapsed
        if ratio < self._warn_ratio or drop_fps < max(1.0, 0.1 * self._source_fps):
            return

        self._last_warn = now
        log.warning(
            "pipeline cannot keep up with the input stream: dropped %d of %d "
            "frames in the last %.1fs (%.0f%%, ~%.1f drop/s). Hardware may be "
            "too slow for %dx%d @ %.1f fps — lower video_source.config."
            "processing_fps or preprocessing.scale, or use a faster machine",
            window_drops,
            considered,
            elapsed,
            100.0 * ratio,
            drop_fps,
            self._width,
            self._height,
            self._source_fps,
        )
