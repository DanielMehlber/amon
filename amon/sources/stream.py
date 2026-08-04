"""Live video capture source via OpenCV ``VideoCapture``.

Works with any camera-style input OpenCV can open: USB capture adapters,
capture cards, webcams.  Prefer a numeric ``device`` index (``0``, ``1``, …)
— that form is portable across Linux, macOS and Windows.  Device paths such
as ``/dev/video0`` are accepted where the OS exposes them (typically Linux).

Native resolution and frame rate are detected automatically.  Optional
``processing_fps`` throttles how many frames reach the pipeline.  Geometric
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

        # Best-effort: some backends ignore BUFFERSIZE / FOURCC; that is fine.
        buffer_size = self.config.get("capture_buffer_size", 1)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, float(buffer_size))

        fourcc = self.config.get("capture_fourcc")
        if fourcc:
            if len(fourcc) != 4:
                raise SourceError(
                    f"capture_fourcc must be four characters, got {fourcc!r}"
                )
            self._capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)
            )

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

        backend = ""
        try:
            backend = self._capture.getBackendName()
        except Exception:
            pass
        log.info(
            "Video input on %r (%s): native %dx%d @ %.2f fps → pipeline @ %.2f fps",
            self._device,
            backend or "unknown-backend",
            self._native_size[0],
            self._native_size[1],
            self._native_fps,
            self._output_fps,
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

    def frames(self) -> Iterator[Frame]:
        index = 0
        min_interval = 1.0 / self._output_fps if self._output_fps > 0 else 0.0
        last_emit: Optional[float] = None

        while True:
            ok, image = self._capture.read()
            if not ok:
                return

            now = time.monotonic()
            if (
                min_interval > 0
                and last_emit is not None
                and (now - last_emit) < min_interval
            ):
                continue
            last_emit = now

            yield Frame(index=index, timestamp=index / self._output_fps, image=image)
            index += 1

    def close(self) -> None:
        self._capture.release()
