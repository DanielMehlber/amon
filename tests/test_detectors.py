"""Unit tests for the three bundled detectors.

Each detector is calibrated on the clean first seconds of the synthetic
scene, then fed frames from a scheduled anomaly window.  Ground truth comes
from :mod:`amon.synthetic`.
"""
import numpy as np
import pytest

from amon.detectors.hud import HudDetector
from amon.detectors.spatial import SpatialDetector
from amon.detectors.temporal import CONTRAST, FLICKER, NOISE, TemporalDetector
from amon.synthetic import SyntheticVideo

from .conftest import calibrate, make_frames


def peak_intensities(detector, scene, t0, t1, warmup=0.3):
    """Max intensity per anomaly over a time range (detection mode).

    A short warmup precedes the measured range so that stateful metrics
    (previous frame, sliding windows) are not polluted by the time jump
    between test windows.
    """
    for frame in make_frames(scene, t0 - warmup, t0):
        detector.process(frame)
    peaks = {}
    for frame in make_frames(scene, t0, t1):
        for aid, value in detector.process(frame).items():
            peaks[aid] = max(peaks.get(aid, 0.0), value)
    return peaks


class TestTemporalDetector:
    @pytest.fixture(scope="class")
    def detector(self, scene):
        detector = TemporalDetector()
        calibrate(detector, scene)
        return detector

    def test_calibration_produces_thresholds(self, detector):
        thresholds = detector.thresholds()
        assert set(thresholds) == {NOISE, FLICKER, CONTRAST}
        assert all(v > 0 for v in thresholds.values())
        assert detector.mode == "detection"

    def test_clean_footage_stays_below_thresholds(self, detector, scene):
        peaks = peak_intensities(detector, scene, 12.0, 15.0)
        thresholds = detector.thresholds()
        for aid in thresholds:
            assert peaks[aid] < thresholds[aid], aid

    def test_noise_window_fires_noise(self, detector, scene):
        peaks = peak_intensities(detector, scene, 16.5, 18.5)
        assert peaks[NOISE] > detector.thresholds()[NOISE]
        assert peaks[FLICKER] < detector.thresholds()[FLICKER]

    def test_flicker_window_fires_flicker(self, detector, scene):
        peaks = peak_intensities(detector, scene, 23.5, 25.5)
        assert peaks[FLICKER] > detector.thresholds()[FLICKER]

    def test_contrast_window_fires_contrast(self, detector, scene):
        peaks = peak_intensities(detector, scene, 30.5, 32.5)
        assert peaks[CONTRAST] > detector.thresholds()[CONTRAST]
        assert peaks[NOISE] < detector.thresholds()[NOISE]


class TestHudDetector:
    @pytest.fixture(scope="class")
    def detector(self, scene):
        detector = HudDetector()
        calibrate(detector, scene)
        return detector

    def test_calibration_finds_four_elements(self, detector):
        elements = detector._elements
        assert len(elements) == 4
        texts = {e.text for e in elements.values()}
        assert "CAM01" in texts
        assert "REC" in texts
        blink_rates = sorted(e.toggle_rate for e in elements.values())
        assert blink_rates[0] == pytest.approx(0.0, abs=0.3)   # static labels
        assert blink_rates[-1] == pytest.approx(4.0, abs=0.8)    # 2 Hz REC blinker

    def test_static_label_text_is_read(self, detector):
        texts = {e.text for e in detector._elements.values()}
        assert "CAM 01" in texts or "CAM01" in texts

    def test_anomaly_ids_cover_all_aspects(self, detector):
        thresholds = detector.thresholds()
        for element_id in detector._elements:
            for aspect in ("text", "position", "size", "blink"):
                assert f"hud/{element_id}/{aspect}" in thresholds

    def test_clean_footage_stays_below_thresholds(self, scene):
        detector = HudDetector()
        calibrate(detector, scene)
        peaks = peak_intensities(detector, scene, 10.0, 15.0)
        thresholds = detector.thresholds()
        for aid, peak in peaks.items():
            assert peak < thresholds[aid], aid

    def _fresh(self, scene, warmup_start):
        """Detector with blink windows warmed up on clean footage."""
        detector = HudDetector()
        calibrate(detector, scene)
        peak_intensities(detector, scene, warmup_start, warmup_start + 2.5)
        return detector

    def _label_id(self, detector):
        return next(e.element_id for e in detector._elements.values() if e.text == "CAM01")

    def _blinker_id(self, detector):
        return next(e.element_id for e in detector._elements.values() if e.text == "REC")

    def test_text_change_detected(self, scene):
        detector = self._fresh(scene, 34.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 37.5, 39.5)
        assert peaks[f"hud/{label}/text"] > detector.thresholds()[f"hud/{label}/text"]
        assert peaks[f"hud/{label}/position"] < detector.thresholds()[f"hud/{label}/position"]

    def test_position_change_detected(self, scene):
        detector = self._fresh(scene, 55.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 58.5, 60.5)
        assert peaks[f"hud/{label}/position"] > detector.thresholds()[f"hud/{label}/position"]

    def test_size_change_detected(self, scene):
        detector = self._fresh(scene, 62.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 65.5, 67.5)
        assert peaks[f"hud/{label}/size"] > detector.thresholds()[f"hud/{label}/size"]

    def test_blink_frequency_change_detected(self, scene):
        detector = self._fresh(scene, 41.0)
        blinker = self._blinker_id(detector)
        peaks = peak_intensities(detector, scene, 44.0, 47.0)
        assert peaks[f"hud/{blinker}/blink"] > detector.thresholds()[f"hud/{blinker}/blink"]

    def test_blink_stop_detected(self, scene):
        detector = self._fresh(scene, 48.0)
        blinker = self._blinker_id(detector)
        peaks = peak_intensities(detector, scene, 51.0, 54.0)
        assert peaks[f"hud/{blinker}/blink"] > detector.thresholds()[f"hud/{blinker}/blink"]

    def test_metadata_and_regions(self, detector):
        label = self._label_id(detector)
        metadata = detector.metadata(f"hud/{label}/text")
        assert metadata["element"] == label
        assert detector.regions(f"hud/{label}/text")


class TestSpatialDetector:
    @pytest.fixture(scope="class")
    def detector(self, scene):
        detector = SpatialDetector()
        calibrate(detector, scene)
        return detector

    def test_calibration_finds_keypoints_outside_hud(self, detector):
        points = detector._points.reshape(-1, 2)
        assert len(points) >= 20
        # Most corners should sit on the landscape, not on HUD overlays.
        assert (points[:, 1] > 26).mean() > 0.95

    def test_clean_footage_stays_below_threshold(self, detector, scene):
        threshold = detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(detector, scene, 12.0, 15.0)
        assert peaks["spatial/distortion"] < threshold

    def test_distortion_detected_and_localised(self, detector, scene):
        threshold = detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(detector, scene, 72.5, 74.5)
        assert peaks["spatial/distortion"] > threshold

        regions = detector.regions("spatial/distortion")
        assert regions
        # Affine warp moves terrain features; at least one highlight should sit
        # on the hillside rather than in the top/bottom HUD strips.
        assert any(40 < ry + rh / 2 < 195 for _, ry, _, rh in regions)

    def test_hud_changes_are_ignored(self, detector, scene):
        threshold = detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(detector, scene, 37.5, 39.5)  # HUD text change window
        assert peaks["spatial/distortion"] < threshold
