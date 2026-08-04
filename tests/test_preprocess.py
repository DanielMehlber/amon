"""Unit tests for the frame preprocessing stage."""

import numpy as np
import pytest

from amon.model import Frame
from amon.preprocess import FramePreprocessor
from amon.sources import SourceError


def _frame(width=200, height=100, value=128):
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(index=0, timestamp=0.0, image=image)


class TestFramePreprocessor:
    def test_identity_is_disabled(self):
        prep = FramePreprocessor({"scale": 100, "rotate": 0, "brightness": 0, "contrast": 1.0})
        assert not prep.enabled
        frame = _frame()
        assert prep(frame) is frame

    def test_scale_preserves_aspect_ratio(self):
        prep = FramePreprocessor({"scale": 50})
        out = prep(_frame(200, 100))
        assert out.image.shape[:2] == (50, 100)

    def test_rotate_90_swaps_dimensions(self):
        prep = FramePreprocessor({"rotate": 90})
        out = prep(_frame(200, 100))
        assert out.image.shape[:2] == (200, 100)

    def test_rotate_180_keeps_dimensions(self):
        prep = FramePreprocessor({"rotate": 180})
        out = prep(_frame(200, 100))
        assert out.image.shape[:2] == (100, 200)

    def test_brightness_raises_pixel_values(self):
        prep = FramePreprocessor({"brightness": 40})
        out = prep(_frame(value=100))
        assert int(out.image.mean()) == pytest.approx(140, abs=1)

    def test_contrast_scales_around_zero(self):
        prep = FramePreprocessor({"contrast": 2.0})
        out = prep(_frame(value=50))
        assert int(out.image.mean()) == pytest.approx(100, abs=1)

    def test_invalid_scale_raises(self):
        with pytest.raises(SourceError, match="scale"):
            FramePreprocessor({"scale": 0})

    def test_invalid_brightness_raises(self):
        with pytest.raises(SourceError, match="brightness"):
            FramePreprocessor({"brightness": 300})

    def test_order_rotate_then_scale(self):
        prep = FramePreprocessor({"rotate": 90, "scale": 50})
        # 200x100 → rotate 90 → 200x100 → scale 50% → 100x50
        out = prep(_frame(200, 100))
        assert out.image.shape[:2] == (100, 50)
