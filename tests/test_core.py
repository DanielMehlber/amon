"""Unit tests for stats, config, plugin loading and the OCR helper."""

import numpy as np
import pytest

from amon.config import DEFAULTS, load_config, merge_defaults
from amon.plugins import instantiate, load_class
from amon.stats import robust_threshold
from amon.textocr import levenshtein_norm, read_text, slugify

import cv2


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
    def test_reads_rendered_text(self):
        canvas = np.zeros((40, 200), np.uint8)
        cv2.putText(
            canvas,
            "CAM 01",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            255,
            1,
            cv2.LINE_AA,
        )
        assert read_text(canvas) == "CAM 01"

    def test_empty_image_reads_empty(self):
        assert read_text(np.zeros((20, 20), np.uint8)) == ""

    def test_slugify(self):
        assert slugify("CAM 01") == "cam01"
        assert slugify("!!") == ""

    def test_levenshtein_norm(self):
        assert levenshtein_norm("abc", "abc") == 0.0
        assert levenshtein_norm("abc", "") == 1.0
        assert 0.0 < levenshtein_norm("CAM 01", "ERR 42") <= 1.0
