"""Unit tests for the video source abstraction."""

import pytest

from amon.sources import SourceError
from amon.sources.file import VideoFileSource
from amon.synthetic import FPS


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
