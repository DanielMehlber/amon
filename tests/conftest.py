"""Shared fixtures: synthetic test video and a completed monitoring session."""

from __future__ import annotations

import pytest

from amon.config import merge_defaults
from amon.model import Frame
from amon.synthetic import FPS, SyntheticVideo, write_video


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    """Path to the deterministic synthetic test video."""
    path = tmp_path_factory.mktemp("video") / "synthetic.avi"
    return write_video(str(path))


@pytest.fixture(scope="session")
def scene() -> SyntheticVideo:
    """Direct (codec-free) access to synthetic frames for unit tests."""
    return SyntheticVideo()


def make_frames(scene: SyntheticVideo, t0: float, t1: float):
    """Yield synthetic frames for the time range [t0, t1)."""
    for i in range(int(round(t0 * FPS)), int(round(t1 * FPS))):
        t = i / FPS
        yield Frame(index=i, timestamp=t, image=scene.frame(t, index=i))


def calibrate(detector, scene: SyntheticVideo, duration: float = 10.0):
    """Run a detector through the calibration phase on clean footage."""
    for frame in make_frames(scene, 0.0, duration):
        detector.process(frame)
    return detector.finish_calibration()


@pytest.fixture(scope="session")
def completed_session(synthetic_video, tmp_path_factory):
    """Run the full pipeline once on the synthetic video.

    Returns ``(config, session_id)``; tests inspect the resulting database
    and media files.
    """
    from amon.pipeline import Pipeline

    data_dir = tmp_path_factory.mktemp("data")
    config = merge_defaults(
        {
            "session_name": "e2e",
            "data_dir": str(data_dir),
            "video_source": {
                "class": "amon.sources.file.VideoFileSource",
                "config": {"path": synthetic_video},
            },
            "calibration": {"duration_seconds": 10.0},
        }
    )
    session_id = Pipeline(config).run()
    return config, session_id
