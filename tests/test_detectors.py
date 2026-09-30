"""Unit tests for the three bundled detectors.

Each detector is calibrated on the clean first seconds of the synthetic
scene, then fed frames from a scheduled anomaly window.  Ground truth comes
from :mod:`amon.synthetic`.
"""

import pytest

from amon.detectors.hud import HudDetector
from amon.detectors.spatial import SpatialDetector
from amon.detectors.temporal import CONTRAST, FLICKER, NOISE, TemporalDetector

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


@pytest.fixture(scope="module")
def temporal_detector(scene):
    detector = TemporalDetector()
    calibrate(detector, scene)
    return detector


@pytest.fixture(scope="module")
def hud_detector(scene):
    detector = HudDetector()
    calibrate(detector, scene)
    return detector


@pytest.fixture(scope="module")
def spatial_detector(scene):
    detector = SpatialDetector()
    calibrate(detector, scene)
    return detector


class TestTemporalDetector:
    def test_calibration_produces_thresholds(self, temporal_detector):
        thresholds = temporal_detector.thresholds()
        assert set(thresholds) == {NOISE, FLICKER, CONTRAST}
        assert all(v > 0 for v in thresholds.values())
        assert temporal_detector.mode == "detection"

    def test_clean_footage_stays_below_thresholds(self, temporal_detector, scene):
        peaks = peak_intensities(temporal_detector, scene, 12.0, 15.0)
        thresholds = temporal_detector.thresholds()
        for aid in thresholds:
            assert peaks[aid] < thresholds[aid], aid

    def test_noise_window_fires_noise(self, temporal_detector, scene):
        peaks = peak_intensities(temporal_detector, scene, 16.5, 18.5)
        assert peaks[NOISE] > temporal_detector.thresholds()[NOISE]
        assert peaks[FLICKER] < temporal_detector.thresholds()[FLICKER]

    def test_flicker_window_fires_flicker(self, temporal_detector, scene):
        peaks = peak_intensities(temporal_detector, scene, 23.5, 25.5)
        assert peaks[FLICKER] > temporal_detector.thresholds()[FLICKER]

    def test_contrast_window_fires_contrast(self, temporal_detector, scene):
        peaks = peak_intensities(temporal_detector, scene, 30.5, 32.5)
        assert peaks[CONTRAST] > temporal_detector.thresholds()[CONTRAST]
        assert peaks[NOISE] < temporal_detector.thresholds()[NOISE]


class TestHudDetector:
    def test_calibration_finds_expected_elements(self, hud_detector):
        elements = hud_detector._elements
        assert len(elements) == 7
        texts = {e.text for e in elements.values()}
        assert "CAM01" in texts
        assert "REC" in texts
        assert "STAT01" in texts
        assert {"S", "A", "X"} <= texts
        assert not any(t.startswith("symbol-") for t in texts)
        blink_rates = sorted(e.toggle_rate for e in elements.values())
        assert blink_rates[0] == pytest.approx(0.0, abs=0.3)  # static labels
        assert blink_rates[-1] == pytest.approx(4.0, abs=0.8)  # 2 Hz REC blinker

    def test_single_letter_elements_are_letters_not_symbols(self, hud_detector):
        """Lone glyphs must calibrate as letter IDs, not symbol-N."""
        by_text = {e.text: e for e in hud_detector._elements.values()}
        for letter in ("S", "A", "X"):
            assert letter in by_text, f"missing lone {letter!r}"
            element = by_text[letter]
            assert not element.element_id.startswith("symbol-")
            assert element.element_id == letter.lower()

    def test_static_label_text_is_read(self, hud_detector):
        texts = {e.text for e in hud_detector._elements.values()}
        assert "CAM 01" in texts or "CAM01" in texts

    def test_stat01_survives_tight_bright_box_via_crop_pad(self, scene):
        """OCR pad restores glyphs when the bright-mask box clips soft edges."""
        import cv2
        import numpy as np
        from amon.synthetic import FPS
        from amon.textocr import read_hud

        detector = HudDetector()
        calibrate(detector, scene)
        element = next(e for e in detector._elements.values() if "STAT" in e.text)
        # Rebuild a max-image frame and inset the bright box (simulates AA fringe
        # excluded by bright_threshold).
        grays = [
            cv2.cvtColor(scene.frame(i / FPS, index=i), cv2.COLOR_BGR2GRAY)
            for i in range(int(10 * FPS))
        ]
        max_img = np.max(np.stack(grays), axis=0)
        x, y, w, h = element.box
        inset = (x + 2, y + 2, max(1, w - 4), max(1, h - 4))
        ix, iy, iw, ih = inset
        tight = max_img[iy : iy + ih, ix : ix + iw]
        assert read_hud(tight).text != element.text
        assert not read_hud(tight).text.startswith("STAT")
        padded = detector._ocr_crop(max_img, inset)
        assert read_hud(padded).text == element.text

    def test_anomaly_ids_cover_all_aspects(self, hud_detector):
        thresholds = hud_detector.thresholds()
        # Calibrated elements expose text/position/size/blink; new overlays are
        # registered dynamically when they appear (no static hud/new threshold).
        for element_id in hud_detector._elements:
            for aspect in ("text", "position", "size", "blink"):
                assert f"hud/{element_id}/{aspect}" in thresholds
        assert not any(aid.endswith("/new") for aid in thresholds)

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
        return next(
            e.element_id for e in detector._elements.values() if e.text == "CAM01"
        )

    def _blinker_id(self, detector):
        return next(
            e.element_id for e in detector._elements.values() if e.text == "REC"
        )

    def test_text_change_detected(self, scene):
        detector = self._fresh(scene, 34.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 37.5, 39.5)
        assert peaks[f"hud/{label}/text"] > detector.thresholds()[f"hud/{label}/text"]
        assert (
            peaks[f"hud/{label}/position"]
            < detector.thresholds()[f"hud/{label}/position"]
        )

    def test_position_change_detected(self, scene):
        detector = self._fresh(scene, 55.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 58.5, 60.5)
        assert (
            peaks[f"hud/{label}/position"]
            > detector.thresholds()[f"hud/{label}/position"]
        )

    def test_size_change_detected(self, scene):
        detector = self._fresh(scene, 62.0)
        label = self._label_id(detector)
        peaks = peak_intensities(detector, scene, 65.5, 67.5)
        assert peaks[f"hud/{label}/size"] > detector.thresholds()[f"hud/{label}/size"]

    def test_new_text_detected(self, scene):
        detector = self._fresh(scene, 66.0)
        peaks = peak_intensities(detector, scene, 69.2, 70.8)
        new_aids = [aid for aid in peaks if aid.endswith("/new")]
        assert len(new_aids) >= 2  # ALERT + WARN
        thresholds = detector.thresholds()
        for aid in new_aids:
            assert peaks[aid] > thresholds[aid]
            assert detector.metadata(aid).get("new") is True
            assert detector.regions(aid)

    def test_new_overlay_keeps_id_when_text_mutates(self, scene):
        """Appear → rewrite glyphs in place → still one frozen /new channel."""
        from amon.synthetic import MUTATE_FLIP_AT

        detector = self._fresh(scene, 89.0)
        seen_ids = set()
        for frame in make_frames(scene, 91.0, 96.0):
            intensities = detector.process(frame)
            new_aids = [aid for aid in intensities if aid.endswith("/new")]
            seen_ids.update(new_aids)
            if 91.2 <= frame.timestamp < MUTATE_FLIP_AT:
                assert "hud/alert/new" in intensities
            if frame.timestamp >= MUTATE_FLIP_AT + 0.2:
                # Still the spawn ID — not hud/1000/new.
                assert "hud/alert/new" in intensities
                assert "hud/1000/new" not in intensities
        assert seen_ids == {"hud/alert/new"}

    def test_non_text_icon_spawns_as_symbol(self, scene):
        """Dots/crosshairs must not be forced into a letter slug."""
        import cv2

        from amon.model import Frame

        detector = self._fresh(scene, 10.0)
        # Build a frame from clean footage with a filled dot outside HUD cover.
        base = scene.frame(10.0, index=200)
        img = base.copy()
        cv2.circle(img, (160, 120), 12, (255, 255, 255), -1)
        intensities = detector.process(
            Frame(index=201, timestamp=10.05, image=img)
        )
        new_aids = [aid for aid in intensities if aid.endswith("/new")]
        assert new_aids == ["hud/symbol-1/new"]
        assert detector.metadata("hud/symbol-1/new").get("text") == "symbol-1"

    def test_scheduled_crosshair_and_dot_are_symbols(self, scene):
        """Synth schedule draws a centre crosshair + side dot as symbol-N /new."""
        from amon.synthetic import SYMBOL_END, SYMBOL_START

        detector = self._fresh(scene, SYMBOL_START - 2.0)
        seen = set()
        for frame in make_frames(scene, SYMBOL_START, SYMBOL_END):
            intensities = detector.process(frame)
            seen.update(aid for aid in intensities if aid.endswith("/new"))
        assert seen == {"hud/symbol-1/new", "hud/symbol-2/new"}
        # Must not invent letter/digit slugs for these icons.
        assert not any(
            aid.split("/")[1] not in {"symbol-1", "symbol-2"} for aid in seen
        )
        for aid in seen:
            assert detector.metadata(aid).get("text", "").startswith("symbol-")

    def test_new_overlay_survives_numeric_steps_and_blink(self, scene):
        """1000 → 2000 in steps, then blink — still a single hud/1000/new."""
        from amon.synthetic import (
            CYCLE_BLINK_START,
            CYCLE_START,
            cycle_hud_text,
        )

        detector = self._fresh(scene, CYCLE_START - 2.0)
        seen_ids = set()
        seen_on_frames = 0
        for frame in make_frames(scene, CYCLE_START, CYCLE_BLINK_START + 3.5):
            intensities = detector.process(frame)
            new_aids = [aid for aid in intensities if aid.endswith("/new")]
            seen_ids.update(new_aids)
            if "hud/1000/new" in intensities:
                seen_on_frames += 1
            assert "hud/2000/new" not in intensities
            if CYCLE_START <= frame.timestamp < CYCLE_BLINK_START:
                assert cycle_hud_text(frame.timestamp) in {
                    "1000",
                    "1200",
                    "1400",
                    "1600",
                    "1800",
                    "2000",
                }
                assert "hud/1000/new" in intensities
        assert seen_ids == {"hud/1000/new"}
        # Blink at 2 Hz still yields many above-threshold frames overall.
        assert seen_on_frames >= 40

    def test_dynamic_new_thresholds_are_forgotten_after_ttl(self, scene):
        """Pruned runtime tracks must not leave thresholds forever."""
        from amon.synthetic import CYCLE_START, SYMBOL_END

        detector = HudDetector({"new_track_ttl_seconds": 0.5})
        calibrate(detector, scene)
        # Drive one cycle overlay briefly.
        for frame in make_frames(scene, CYCLE_START, CYCLE_START + 1.0):
            detector.process(frame)
        assert any(aid.endswith("/new") for aid in detector.thresholds())
        # After the last scheduled /new window + TTL, clean frames forget it.
        for frame in make_frames(scene, SYMBOL_END + 0.5, SYMBOL_END + 2.0):
            detector.process(frame)
        assert not any(aid.endswith("/new") for aid in detector.thresholds())
        assert detector._new_tracks == []

    def test_blink_frequency_change_detected(self, scene):
        detector = self._fresh(scene, 41.0)
        blinker = self._blinker_id(detector)
        peaks = peak_intensities(detector, scene, 44.0, 47.0)
        assert (
            peaks[f"hud/{blinker}/blink"]
            > detector.thresholds()[f"hud/{blinker}/blink"]
        )

    def test_blink_stop_detected(self, scene):
        detector = self._fresh(scene, 48.0)
        blinker = self._blinker_id(detector)
        peaks = peak_intensities(detector, scene, 51.0, 54.0)
        assert (
            peaks[f"hud/{blinker}/blink"]
            > detector.thresholds()[f"hud/{blinker}/blink"]
        )

    def test_metadata_and_regions(self, hud_detector):
        label = self._label_id(hud_detector)
        metadata = hud_detector.metadata(f"hud/{label}/text")
        assert metadata["element"] == label
        assert hud_detector.regions(f"hud/{label}/text")


class TestSpatialDetector:
    def test_calibration_finds_keypoints_outside_hud(self, spatial_detector):
        points = spatial_detector._points.reshape(-1, 2)
        assert len(points) >= 20
        # Most corners should sit on the landscape, not on HUD overlays.
        assert (points[:, 1] > 26).mean() > 0.95

    def test_clean_footage_stays_below_threshold(self, spatial_detector, scene):
        threshold = spatial_detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(spatial_detector, scene, 12.0, 15.0)
        assert peaks["spatial/distortion"] < threshold

    def test_distortion_detected_and_localised(self, spatial_detector, scene):
        threshold = spatial_detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(spatial_detector, scene, 72.5, 74.5)
        assert peaks["spatial/distortion"] > threshold

        regions = spatial_detector.regions("spatial/distortion")
        assert regions
        # Affine warp moves terrain features; at least one highlight should sit
        # on the hillside rather than in the top/bottom HUD strips.
        assert any(40 < ry + rh / 2 < 195 for _, ry, _, rh in regions)

    def test_hud_changes_are_ignored(self, spatial_detector, scene):
        threshold = spatial_detector.thresholds()["spatial/distortion"]
        peaks = peak_intensities(spatial_detector, scene, 37.5, 39.5)  # HUD text change window
        assert peaks["spatial/distortion"] < threshold
