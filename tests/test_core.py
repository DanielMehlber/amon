"""Unit tests for stats, config, plugin loading and the OCR helper."""

import numpy as np
import pytest

from amon.config import DEFAULTS, load_config, merge_defaults
from amon.plugins import instantiate, load_class
from amon.stats import robust_threshold
from amon.textocr import levenshtein_norm, read_text, slugify


class TestRobustThreshold:
    def test_empty_samples_fall_back_to_floor(self):
        assert robust_threshold([], floor=2.5) == 2.5

    def test_threshold_clears_all_calibration_samples(self):
        samples = list(np.random.default_rng(1).normal(1.0, 0.1, 200))
        threshold = robust_threshold(samples)
        assert threshold > max(samples)

    def test_outliers_do_not_explode_threshold(self):
        samples = [0.1] * 100 + [50.0]  # one glitch during calibration
        threshold = robust_threshold(samples)
        assert threshold == pytest.approx(75.0)  # headroom * max, not sigma-based

    def test_floor_applies_to_zero_samples(self):
        assert robust_threshold([0.0] * 50, floor=3.0) == 3.0


class TestConfig:
    def test_defaults_are_merged(self):
        config = merge_defaults({"calibration": {"duration_seconds": 3.0}})
        assert config["calibration"]["duration_seconds"] == 3.0
        assert config["media"]["lead_seconds"] == DEFAULTS["media"]["lead_seconds"]

    def test_load_config(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("session_name: abc\nmedia:\n  gif_max_fps: 5\n")
        config = load_config(path)
        assert config["session_name"] == "abc"
        assert config["media"]["gif_max_fps"] == 5
        assert config["aggregation"]["cooldown_seconds"] > 0


class TestPlugins:
    def test_load_class(self):
        cls = load_class("amon.detectors.temporal.TemporalDetector")
        assert cls.__name__ == "TemporalDetector"

    def test_instantiate_passes_config(self):
        detector = instantiate(
            {
                "class": "amon.detectors.temporal.TemporalDetector",
                "config": {"noise_floor": 9.0},
            }
        )
        assert detector.config["noise_floor"] == 9.0
        assert detector.mode == "calibration"

    def test_invalid_paths_raise(self):
        with pytest.raises(ValueError):
            load_class("NoDots")
        with pytest.raises(ImportError):
            load_class("amon.detectors.temporal.Missing")


class TestTextOcr:
    def _render_text(self, text: str, size: int = 28) -> np.ndarray:
        from amon.textocr import resolve_glyph_font
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(resolve_glyph_font()), size=size)
        img = Image.new("L", (220, 48), 0)
        ImageDraw.Draw(img).text((8, 8), text, fill=255, font=font)
        return np.asarray(img)

    def test_reads_rendered_text(self):
        assert read_text(self._render_text("CAM 01")) == "CAM 01"

    def test_rejects_tiny_speckles(self):
        """Sub-min-size bright blobs must not be matched as letters."""
        canvas = np.zeros((40, 40), np.uint8)
        canvas[10:13, 10:13] = 255  # 3x3 speck — below default min height/area
        assert read_text(canvas) == ""
        # Same speck is accepted only if thresholds are lowered
        assert read_text(canvas, min_glyph_height=2, min_glyph_area=4) != ""

    def test_rejects_oversized_blobs(self):
        """Large bright regions must not be treated as glyphs."""
        canvas = np.zeros((120, 120), np.uint8)
        canvas[10:100, 10:100] = 255  # 90x90 — above default max height/width
        assert read_text(canvas) == ""
        assert read_text(canvas, max_glyph_height=100, max_glyph_width=100) != ""

    def test_empty_image_reads_empty(self):
        assert read_text(np.zeros((20, 20), np.uint8)) == ""

    def test_resolve_default_font(self):
        from amon.textocr import DEFAULT_GLYPH_FONT, resolve_glyph_font

        path = resolve_glyph_font()
        assert path.name == DEFAULT_GLYPH_FONT
        assert path.is_file()

    def test_slugify(self):
        assert slugify("CAM 01") == "cam01"
        assert slugify("!!") == ""

    def test_levenshtein_norm(self):
        assert levenshtein_norm("abc", "abc") == 0.0
        assert levenshtein_norm("abc", "") == 1.0
        assert 0.0 < levenshtein_norm("CAM 01", "ERR 42") <= 1.0
