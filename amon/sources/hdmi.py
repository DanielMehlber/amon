"""Live HDMI capture source via OpenCV ``VideoCapture``.

Typical setup: an HDMI capture card (USB or PCIe) appears as a V4L2 device
on Linux (``/dev/video0``), a DirectShow device on Windows, or an AVFoundation
device on macOS.  The native stream resolution and frame rate are detected
automatically; optional ``processing_*`` settings downscale and throttle
frames before they reach the pipeline.
"""

from __future__ import annotations

import glob
import logging
import time
from typing import Iterator, List, Optional, Tuple, Union

import cv2

from amon.model import Frame
from amon.sources import SourceError, VideoSource

log = logging.getLogger("amon.sources.hdmi")

Device = Union[int, str]
Size = Tuple[int, int]

# How many device indices to probe when enumerating capture hardware.
_MAX_DEVICE_PROBE = 10


def list_capture_devices(max_probe: int = _MAX_DEVICE_PROBE) -> List[str]:
    """Return identifiers for capture devices that OpenCV can open."""
    found: List[str] = []
    seen = set()

    for path in sorted(glob.glob("/dev/video*")):
        if _probe_capture_device(path):
            found.append(path)
            seen.add(path)

    for index in range(max_probe):
        key = str(index)
        if key in seen:
            continue
        if _probe_capture_device(index):
            found.append(key)
            seen.add(key)

    return found


def _probe_capture_device(device: Device) -> bool:
    capture = cv2.VideoCapture(device)
    try:
        return capture.isOpened()
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


def _processing_size(native: Size, config: dict) -> Optional[Size]:
    """Return the (width, height) passed to the pipeline, if downscaling."""
    width = config.get("processing_width")
    height = config.get("processing_height")
    if width is None and height is None:
        return None
    native_w, native_h = native
    if width is not None and height is not None:
        return int(width), int(height)
    if width is not None:
        scale = int(width) / native_w
        return int(width), max(1, int(round(native_h * scale)))
    scale = int(height) / native_h
    return max(1, int(round(native_w * scale))), int(height)


class HdmiCaptureSource(VideoSource):
    """Reads frames from a live HDMI (or other) capture device.

    Config keys:

    Input
    -----
    - ``device`` (required): capture device index (``0``, ``1``, …) or path
      (``/dev/video0`` on Linux).
    - ``capture_fourcc``: optional pixel format negotiated with the hardware
      (e.g. ``"MJPG"``).
    - ``capture_buffer_size`` (default ``1``): driver buffer depth; ``1``
      minimises latency.

    Processing (optional)
    ---------------------
    Native resolution and frame rate are detected automatically.  Use these
    keys to limit what the monitoring pipeline receives:

    - ``processing_width``, ``processing_height``: downscale each frame
      (aspect ratio preserved when only one dimension is set).
    - ``processing_fps``: maximum frame rate delivered to the pipeline;
      extra frames from the device are dropped.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        if "device" not in self.config:
            raise SourceError("video_source config requires a 'device'")
        self._device = _parse_device(self.config["device"])
        self._capture = cv2.VideoCapture(self._device)
        if not self._capture.isOpened():
            self._raise_device_unavailable()

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
                "cannot determine capture frame rate; check the HDMI signal "
                "or set processing_fps to the expected rate"
            )

        self._output_size = _processing_size(self._native_size, self.config)
        processing_fps = self.config.get("processing_fps")
        self._output_fps = (
            float(processing_fps) if processing_fps is not None else self._native_fps
        )
        if self._output_fps <= 0:
            raise SourceError("processing_fps must be positive")

        log.info(
            "HDMI capture on %r: native %dx%d @ %.2f fps → pipeline %s @ %.2f fps",
            self._device,
            self._native_size[0],
            self._native_size[1],
            self._native_fps,
            f"{self._output_size[0]}x{self._output_size[1]}"
            if self._output_size
            else f"{self._native_size[0]}x{self._native_size[1]}",
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

            if self._output_size is not None:
                image = cv2.resize(
                    image,
                    self._output_size,
                    interpolation=cv2.INTER_AREA,
                )

            yield Frame(index=index, timestamp=index / self._output_fps, image=image)
            index += 1

    def close(self) -> None:
        self._capture.release()
