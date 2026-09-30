"""Tests for width-relative geometry scaling."""

from amon.detectors.hud import HudDetector
from amon.detectors.spatial import SpatialDetector
from amon.scale import REFERENCE_WIDTH_PX, rel_width, rel_width_float, rel_width_sq


class TestScaleHelpers:
    def test_rel_round_trip_at_reference(self):
        assert rel_width(15 / REFERENCE_WIDTH_PX, REFERENCE_WIDTH_PX) == 15
        assert rel_width(20 / REFERENCE_WIDTH_PX, REFERENCE_WIDTH_PX) == 20
        assert rel_width_sq(15 / REFERENCE_WIDTH_PX**2, REFERENCE_WIDTH_PX) == 15

    def test_rel_scales_with_frame(self):
        # 15 px @ 1200 → 4 px @ 300 (25% scale)
        assert rel_width(15 / REFERENCE_WIDTH_PX, 300) == 4
        assert rel_width_float(30 / REFERENCE_WIDTH_PX, 300) == 7.5

    def test_rel_round_up_for_max_gates(self):
        # 96/1200 * 320 = 25.6 → ceil keeps glyph headroom after size anomalies
        assert rel_width(96 / REFERENCE_WIDTH_PX, 320, round_up=True) == 26


class TestDetectorRelativeDefaults:
    def test_hud_merge_kernel_scales_down(self):
        detector = HudDetector()
        detector._frame_width = 300  # 25% of 1200
        assert detector._length_px("merge_kernel_rel") == 4
        assert detector._length_px("search_margin_rel") == 5
        assert detector._length_px("glyph_crop_pad_rel") == 2
        assert detector._length_pxf("position_floor_rel") == 1.5
        assert detector._length_pxf("new_match_distance_rel") == 7.5

    def test_hud_defaults_match_legacy_at_1200(self):
        detector = HudDetector()
        detector._frame_width = REFERENCE_WIDTH_PX
        assert detector._length_px("merge_kernel_rel") == 15
        assert detector._length_px("search_margin_rel") == 20
        assert detector._length_px("min_glyph_height_rel") == 8
        assert detector._area_px("min_element_area_rel_sq") == 15
        assert detector._area_px("min_glyph_area_rel_sq") == 20
        assert detector._length_px("glyph_crop_pad_rel") == 6

    def test_spatial_defaults_match_legacy_at_1200(self):
        detector = SpatialDetector()
        detector._frame_width = REFERENCE_WIDTH_PX
        assert detector._length_px("exclusion_dilate_rel") == 21
        assert detector._length_px("corner_min_distance_rel") == 7
        assert detector._length_pxf("fb_max_error_rel") == 1.5
        assert detector._length_pxf("floor_rel") == 2.5
        assert detector._length_px("region_size_rel") == 28
