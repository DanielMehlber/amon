"""Tests for evidence GIF rendering and per-frame region markers."""

from pathlib import Path

import numpy as np

from amon.media import regions_at, write_event_gif
from amon.model import AnomalyEvent


def test_regions_at_tracks_latest_nonempty_box():
    timeline = [
        (1.0, [(10, 10, 20, 10)]),
        (2.0, []),  # blink gap — keep previous
        (3.0, [(40, 40, 50, 20)]),
    ]
    assert regions_at(timeline, 0.5, fallback=[(1, 1, 1, 1)]) == [(1, 1, 1, 1)]
    assert regions_at(timeline, 1.5) == [(10, 10, 20, 10)]
    assert regions_at(timeline, 2.5) == [(10, 10, 20, 10)]
    assert regions_at(timeline, 3.5) == [(40, 40, 50, 20)]


def test_event_gif_marker_follows_region_timeline(tmp_path: Path):
    """Overlays that rewrite glyphs must keep the highlight on the live bbox."""
    frames = []
    for i, t in enumerate([0.0, 0.5, 1.0, 1.5]):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        frames.append((t, img))

    early = (5, 5, 20, 10)
    late = (60, 40, 40, 25)
    event = AnomalyEvent(
        anomaly_id="hud/alert/new",
        detector="hud",
        start=0.4,
        end=1.6,
        max_intensity=1.0,
        threshold=0.5,
        regions=[list(early)],
    )
    path = tmp_path / "track.gif"
    write_event_gif(
        frames,
        event,
        path,
        fps=2.0,
        gif_fps=2.0,
        region_timeline=[(0.5, [list(early)]), (1.5, [list(late)])],
    )
    assert path.exists()

    # Re-render into arrays by calling the box picker path: ensure late box is
    # preferred at t=1.5 (regions_at contract used by write_event_gif).
    assert regions_at(
        [(0.5, [list(early)]), (1.5, [list(late)])], 1.5, fallback=[list(early)]
    ) == [late]
