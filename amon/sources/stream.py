"""Live video capture source via OpenCV ``VideoCapture``.

Works with any camera-style input OpenCV can open: USB capture adapters,
capture cards, webcams.  Prefer a numeric ``device`` index (``0``, ``1``, …)
— that form is portable across Linux, macOS and Windows.  Device paths such
as ``/dev/video0`` are accepted where the OS exposes them (typically Linux).

Native resolution and frame rate are detected automatically.  Optional
``processing_fps`` throttles how many frames reach the pipeline.  A dedicated
reader thread keeps a small :class:`~amon.sources.frame_buffer.DroppingFrameBuffer`
so a slow pipeline drops *old* frames instead of growing a backlog.  Geometric
and photometric transforms (scale, rotate, brightness, contrast) live in the
top-level ``preprocessing`` config — see :mod:`amon.preprocess`.
"""

from __future__ import annotations

import glob
import logging
import time
from typing import Iterator, List, Optional, Tuple, Union

import cv2

from amon.model import Frame
from amon.sources import SourceError, VideoSource
from amon.sources.frame_buffer import DroppingFrameBuffer, OverloadMonitor

log = logging.getLogger("amon.sources.stream")

Device = Union[int, str]
Size = Tuple[int, int]

# How many device indices to probe when enumerating capture hardware.
_MAX_DEVICE_PROBE = 10


def _open_capture(device: Device) -> cv2.VideoCapture:
    """Open a capture handle in a backend-agnostic way.

    Integer indices use ``CAP_ANY`` so OpenCV picks the host's native
    backend (V4L2 / DirectShow / AVFoundation / …).  String identifiers
    (device paths or backend-specific names) are passed through as-is.
    """
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_ANY)
    return cv2.VideoCapture(device, cv2.CAP_ANY)


def list_capture_devices(max_probe: int = _MAX_DEVICE_PROBE) -> List[str]:
    """Return identifiers for capture devices that OpenCV can open.

    Always probes numeric indices (portable).  Also includes ``/dev/video*``
    paths when they exist on the host — empty on Windows/macOS, no OS check
    required.
    """
    found: List[str] = []
    seen = set()

    for index in range(max_probe):
        key = str(index)
        if _probe_capture_device(index):
            found.append(key)
            seen.add(key)

    for path in sorted(glob.glob("/dev/video*")):
        if path in seen:
            continue
        if _probe_capture_device(path):
            found.append(path)
            seen.add(path)

    return found


def _probe_capture_device(device: Device) -> bool:
    """True if the device opens and yields at least one frame."""
    capture = _open_capture(device)
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None)
    finally:
        capture.release()


def _parse_device(device) -> Device:
    if isinstance(device, bool):
        raise SourceError(f"invalid capture device: {device!r}")
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        text = device.strip()
        if text.isdigit():
            return int(text)
        return text
    raise SourceError(
        f"'device' must be an integer index or device path, got {type(device).__name__}"
    )


def _detect_stream_size(capture: cv2.VideoCapture) -> Size:
    """Prefer a real frame's shape; CAP_PROP_* alone is unreliable on some backends."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, image = capture.read()
    if ok and image is not None:
        height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise SourceError("cannot determine capture resolution from the device")
    return width, height


def _detect_stream_fps(capture: cv2.VideoCapture) -> float:
    reported = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if reported > 0:
        return reported
    return _estimate_fps(capture)


def _estimate_fps(capture: cv2.VideoCapture, max_samples: int = 20) -> float:
    """Measure throughput by reading a short burst of frames."""
    start = time.monotonic()
    count = 0
    while count < max_samples:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    elapsed = time.monotonic() - start
    if count < 2 or elapsed <= 0:
        return 0.0
    return count / elapsed


class VideoInputStream(VideoSource):
    """Reads frames from a live capture device (USB adapter, capture card, …).

    Config keys:

    Input
    -----
    - ``device`` (required): portable capture index (``0``, ``1``, …) or a
      host-specific path/name (e.g. ``/dev/video0`` on Linux).
    - ``capture_fourcc``: optional pixel format negotiated with the hardware
      (e.g. ``"MJPG"``).  Ignored by backends that do not support it.
    - ``capture_buffer_size`` (default ``1``): driver buffer depth when
      supported; ``1`` minimises latency.
    - ``frame_buffer_size`` (default ``1``): software queue between the
      capture thread and the pipeline.  When full, the oldest frame is
      dropped so the session stays near real time instead of growing RAM.
    - ``warn_on_dropped_frames`` (default ``true``): log a warning when the
      pipeline routinely drops frames because it cannot keep up.  Set
      ``false`` to silence that warning (frames are still dropped).
    - ``reconnect_attempts`` (default ``10``): how many reopen tries after
      consecutive failed reads before the stream gives up.
    - ``reconnect_backoff_seconds`` (default ``1.0``): delay between reopen
      attempts (grows mildly with each try, capped at 30s).

    Rate limiting (optional)
    ------------------------
    Native resolution and frame rate are detected automatically.

    - ``processing_fps``: maximum frame rate delivered to the pipeline;
      extra frames from the device are dropped.

    Image transforms (scale, rotate, brightness, contrast) are configured
    under the top-level ``preprocessing`` section, not here.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        if "device" not in self.config:
            raise SourceError("video_source config requires a 'device'")
        self._device = _parse_device(self.config["device"])
        self._capture = _open_capture(self._device)
        if not self._capture.isOpened():
            self._raise_device_unavailable()
        self._frame_buffer: Optional[DroppingFrameBuffer] = None
        self._closed = False

        # Best-effort: some backends ignore BUFFERSIZE / FOURCC; that is fine.
        self._driver_buffer_size = self.config.get("capture_buffer_size", 1)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self._driver_buffer_size))

        fourcc = self.config.get("capture_fourcc")
        if fourcc:
            if len(fourcc) != 4:
                raise SourceError(
                    f"capture_fourcc must be four characters, got {fourcc!r}"
                )
            self._capture_fourcc = fourcc
            self._capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)
            )
        else:
            self._capture_fourcc = None

        self._native_size = _detect_stream_size(self._capture)
        self._native_fps = _detect_stream_fps(self._capture)
        if self._native_fps <= 0:
            raise SourceError(
                "cannot determine capture frame rate; check the input signal "
                "or set processing_fps to the expected rate"
            )

        processing_fps = self.config.get("processing_fps")
        self._output_fps = (
            float(processing_fps) if processing_fps is not None else self._native_fps
        )
        if self._output_fps <= 0:
            raise SourceError("processing_fps must be positive")

        frame_buffer_size = int(self.config.get("frame_buffer_size", 1))
        if frame_buffer_size < 1:
            raise SourceError("frame_buffer_size must be >= 1")
        self._frame_buffer_size = frame_buffer_size
        self._warn_on_dropped_frames = bool(
            self.config.get("warn_on_dropped_frames", True)
        )
        self._reconnect_attempts = max(
            0, int(self.config.get("reconnect_attempts", 10))
        )
        self._reconnect_backoff = float(
            self.config.get("reconnect_backoff_seconds", 1.0)
        )
        if self._reconnect_backoff < 0:
            raise SourceError("reconnect_backoff_seconds must be >= 0")
        self._reconnect_attempts_used = 0

        backend = ""
        try:
            backend = self._capture.getBackendName()
        except Exception:
            pass
        log.info(
            "Video input on %r (%s): native %dx%d @ %.2f fps → pipeline @ %.2f fps "
            "(frame_buffer_size=%d, reconnect_attempts=%d)",
            self._device,
            backend or "unknown-backend",
            self._native_size[0],
            self._native_size[1],
            self._native_fps,
            self._output_fps,
            self._frame_buffer_size,
            self._reconnect_attempts,
        )

    def _raise_device_unavailable(self) -> None:
        alternatives = list_capture_devices()
        message = f"cannot open capture device {self._device!r}"
        if alternatives:
            message += "; available devices: " + ", ".join(alternatives)
        else:
            message += "; no capture devices found"
        raise SourceError(message)

    @property
    def fps(self) -> float:
        """Frame rate seen by the monitoring pipeline."""
        return self._output_fps

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def native_size(self) -> Size:
        return self._native_size

    @property
    def device(self) -> Device:
        return self._device

    def _apply_capture_options(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self._driver_buffer_size))
        if self._capture_fourcc:
            capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._capture_fourcc)
            )

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in short slices so ``close()`` can abort a reconnect wait."""
        deadline = time.monotonic() + max(0.0, seconds)
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _reopen_capture(self) -> bool:
        """Release and reopen the device.  Returns True on success."""
        try:
            self._capture.release()
        except Exception:
            pass
        capture = _open_capture(self._device)
        if not capture.isOpened():
            log.warning("reconnect: cannot reopen capture device %r", self._device)
            self._capture = capture
            return False
        self._apply_capture_options(capture)
        self._capture = capture
        log.info("reconnect: capture device %r reopened", self._device)
        return True

    def _read_with_reconnect(self):
        """Read one frame; reopen the device after transient failures.

        Returns ``(False, None)`` only when the source is closed or reconnect
        attempts are exhausted — so a USB glitch does not end a multi-day run.
        A successful frame resets the per-outage reconnect budget.
        """
        fail_streak = 0
        while not self._closed:
            ok, image = self._capture.read()
            if ok and image is not None:
                self._reconnect_attempts_used = 0
                return True, image

            fail_streak += 1
            if self._closed:
                break
            if self._reconnect_attempts <= 0:
                return False, None

            if self._reconnect_attempts_used >= self._reconnect_attempts:
                log.error(
                    "capture device %r failed after %d reconnect attempts — ending stream",
                    self._device,
                    self._reconnect_attempts_used,
                )
                return False, None

            self._reconnect_attempts_used += 1
            backoff = min(
                30.0, self._reconnect_backoff * self._reconnect_attempts_used
            )
            log.warning(
                "capture read failed on %r (streak=%d) — reconnecting in %.1fs "
                "(attempt %d/%d)",
                self._device,
                fail_streak,
                backoff,
                self._reconnect_attempts_used,
                self._reconnect_attempts,
            )
            self._interruptible_sleep(backoff)
            if self._closed:
                break
            self._reopen_capture()
        return False, None

    def frames(self) -> Iterator[Frame]:
        """Yield frames from a dropping prefetch buffer.

        A reader thread keeps draining the device so a slow pipeline never
        accumulates an unbounded backlog — oldest buffered frames are discarded
        when newer ones arrive.  Frequent drops trigger an overload warning
        unless ``warn_on_dropped_frames`` is ``false``.
        Transient capture failures trigger reopen attempts before ending.
        """
        buffer = DroppingFrameBuffer(
            self._read_with_reconnect,
            maxsize=self._frame_buffer_size,
            name=f"device-{self._device}",
        )
        self._frame_buffer = buffer
        buffer.start()
        overload = OverloadMonitor(
            width=self._native_size[0],
            height=self._native_size[1],
            source_fps=self._native_fps,
            enabled=self._warn_on_dropped_frames,
        )
        index = 0
        min_interval = 1.0 / self._output_fps if self._output_fps > 0 else 0.0
        last_emit: Optional[float] = None
        try:
            while True:
                item = buffer.get(timeout=1.0)
                if item is None:
                    if buffer.ended:
                        return
                    continue
                image, _captured_at = item

                now = time.monotonic()
                if (
                    min_interval > 0
                    and last_emit is not None
                    and (now - last_emit) < min_interval
                ):
                    # Intentional throttle: frame already drained from the
                    # device so we stay near real time without counting this
                    # as pipeline overload.  Sleep the remaining gap so a full
                    # buffer cannot busy-spin the consumer thread.
                    time.sleep(min_interval - (now - last_emit))
                    continue
                last_emit = now
                overload.note_emit(buffer.dropped)
                yield Frame(
                    index=index, timestamp=index / self._output_fps, image=image
                )
                index += 1
        finally:
            buffer.close()
            if self._frame_buffer is buffer:
                self._frame_buffer = None

    def close(self) -> None:
        self._closed = True
        if self._frame_buffer is not None:
            self._frame_buffer.close()
            self._frame_buffer = None
        self._capture.release()
