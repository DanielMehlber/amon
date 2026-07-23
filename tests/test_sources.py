"""Unit tests for the video source abstraction."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from amon.sources import SourceError, source_label
from amon.sources.file import VideoFileSource
from amon.sources.hdmi import HdmiCaptureSource, list_capture_devices
from amon.synthetic import FPS


class TestSourceLabel:
    def test_file_path(self):
        assert source_label({"class": "x", "config": {"path": "a.avi"}}) == "a.avi"

    def test_hdmi_device(self):
        assert source_label({"class": "x", "config": {"device": 0}}) == "0"

    def test_falls_back_to_class(self):
        assert source_label({"class": "pkg.Source"}) == "pkg.Source"


class TestVideoFileSource:
    def test_missing_path_raises(self):
        with pytest.raises(SourceError, match="path"):
            VideoFileSource({})

    def test_unopenable_file_raises(self, tmp_path):
        with pytest.raises(SourceError, match="cannot open"):
            VideoFileSource({"path": str(tmp_path / "nope.avi")})

    def test_reads_frames_sequentially(self, synthetic_video):
        with VideoFileSource({"path": synthetic_video}) as source:
            assert source.fps == pytest.approx(FPS)
            frames = []
            for frame in source.frames():
                frames.append(frame)
                if frame.index >= 9:
                    break
            assert [f.index for f in frames] == list(range(10))
            assert frames[1].timestamp == pytest.approx(1 / FPS)
            assert frames[0].image.ndim == 3


def _mock_open_capture(
    capture_cls,
    *,
    opened=True,
    width=1920,
    height=1080,
    fps=30.0,
    read_frames=None,
):
    cap = capture_cls.return_value
    cap.isOpened.return_value = opened

    def get_prop(prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return height
        if prop == cv2.CAP_PROP_FPS:
            return fps
        return 0

    cap.get.side_effect = get_prop
    if read_frames is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        read_frames = [(True, image)]
    cap.read.side_effect = read_frames
    return cap


class TestListCaptureDevices:
    def test_lists_openable_indices(self):
        with patch("amon.sources.hdmi._probe_capture_device") as probe:
            probe.side_effect = lambda device: device in (0, 2, "/dev/video0")
            with patch("amon.sources.hdmi.glob.glob", return_value=["/dev/video0"]):
                assert list_capture_devices(max_probe=4) == ["/dev/video0", "0", "2"]


class TestHdmiCaptureSource:
    def test_missing_device_raises(self):
        with pytest.raises(SourceError, match="device"):
            HdmiCaptureSource({})

    def test_unopenable_device_lists_alternatives(self):
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            capture_cls.return_value.isOpened.return_value = False
            with patch(
                "amon.sources.hdmi.list_capture_devices", return_value=["0", "2"]
            ):
                with pytest.raises(SourceError, match="available devices: 0, 2"):
                    HdmiCaptureSource({"device": 99})

    def test_unopenable_device_without_alternatives(self):
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            capture_cls.return_value.isOpened.return_value = False
            with patch("amon.sources.hdmi.list_capture_devices", return_value=[]):
                with pytest.raises(SourceError, match="no capture devices found"):
                    HdmiCaptureSource({"device": 99})

    def test_auto_detects_native_properties(self):
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(capture_cls, width=1280, height=720, fps=25.0)
            source = HdmiCaptureSource({"device": 0})
            assert source.native_size == (1280, 720)
            assert source.native_fps == 25.0
            assert source.fps == 25.0

    def test_processing_fps_caps_output_rate(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(
                capture_cls,
                width=320,
                height=240,
                fps=30.0,
                read_frames=[(True, image)] * 5 + [(False, None)],
            )
            source = HdmiCaptureSource({"device": 0, "processing_fps": 10})
            assert source.fps == 10.0

            times = iter([0.0, 0.05, 0.10, 0.20, 0.30])
            with patch("amon.sources.hdmi.time.monotonic", side_effect=lambda: next(times)):
                frames = list(source.frames())
            assert len(frames) == 3
            assert frames[1].timestamp == pytest.approx(0.1)

    def test_processing_size_downscales(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(
                capture_cls,
                width=1920,
                height=1080,
                fps=30.0,
                read_frames=[(True, image), (True, image), (False, None)],
            )
            with patch("amon.sources.hdmi.cv2.resize") as resize:
                resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                source = HdmiCaptureSource(
                    {
                        "device": 0,
                        "processing_width": 640,
                        "processing_height": 480,
                    }
                )
                times = iter([0.0, 0.05])
                with patch(
                    "amon.sources.hdmi.time.monotonic", side_effect=lambda: next(times)
                ):
                    frame = next(source.frames())
            resize.assert_called_once()
            assert frame.image.shape[:2] == (480, 640)

    def test_reads_frames_from_device(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(
                capture_cls,
                width=320,
                height=240,
                fps=25.0,
                read_frames=[(True, image), (True, image), (True, image), (False, None)],
            )
            with HdmiCaptureSource({"device": "/dev/video0"}) as source:
                assert source.device == "/dev/video0"
                times = iter([0.0, 0.05, 0.10, 0.15])
                with patch(
                    "amon.sources.hdmi.time.monotonic", side_effect=lambda: next(times)
                ):
                    frames = list(source.frames())
            assert len(frames) == 2
            assert frames[1].timestamp == pytest.approx(1 / 25)
            capture_cls.return_value.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1.0)

    def test_numeric_device_string(self):
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(capture_cls)
            HdmiCaptureSource({"device": "2"})
            capture_cls.assert_called_once_with(2)

    def test_invalid_capture_fourcc_raises(self):
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            _mock_open_capture(capture_cls)
            with pytest.raises(SourceError, match="capture_fourcc"):
                HdmiCaptureSource({"device": 0, "capture_fourcc": "MJ"})

    def test_estimates_fps_when_device_reports_zero(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch("amon.sources.hdmi.cv2.VideoCapture") as capture_cls:
            cap = _mock_open_capture(
                capture_cls,
                fps=0.0,
                read_frames=[(True, image)] * 11 + [(False, None)],
            )
            times = iter([0.0, 0.5])
            with patch("amon.sources.hdmi.time.monotonic", side_effect=lambda: next(times)):
                source = HdmiCaptureSource({"device": 0})
            assert source.native_fps == pytest.approx(20.0, rel=0.01)
            assert cap.read.call_count >= 11
