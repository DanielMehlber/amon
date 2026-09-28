"""Tests for evidence GIF rendering and per-frame region markers."""

from pathlib import Path

import numpy as np

from amon.media import HIGHLIGHT, regions_at, write_event_gif
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


def test_event_gif_keeps_marker_after_event_end_and_content_change(tmp_path: Path):
    """Marker must survive past event.end and follow the rewritten bbox.

    Mimics a /new overlay that briefly drops out of aggregation (early end)
    while evidence frames continue and the glyphs change size/position.
    """
    early = (5, 5, 20, 10)
    late = (60, 40, 40, 25)
    frames = []
    for i, t in enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 2.5)):
        img = np.full((80, 120, 3), i * 10, dtype=np.uint8)  # unique frames
        frames.append((t, img))

    event = AnomalyEvent(
        anomaly_id="hud/alert/new",
        detector="hud",
        start=0.4,
        end=0.8,  # closed early (detection gap) while clip continues
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
        region_timeline=[
            (0.5, [list(early)]),
            (1.5, [list(late)]),
        ],
    )

    assert regions_at([(0.5, [list(early)]), (1.5, [list(late)])], 2.5) == [
        late
    ]

    from PIL import Image

    gif = Image.open(path)
    assert gif.n_frames == 6
    # Last frame is past event.end; late box must still be stroked in red.
    gif.seek(5)
    arr = np.asarray(gif.convert("RGB"))
    rgb_highlight = (HIGHLIGHT[2], HIGHLIGHT[1], HIGHLIGHT[0])
    x, y, _, _ = late
    assert tuple(arr[y, x].tolist()) == rgb_highlight
    # Lead-in before first region must not be marked.
    gif.seek(0)
    arr0 = np.asarray(gif.convert("RGB"))
    assert not (arr0 == rgb_highlight).all(axis=2).any()
