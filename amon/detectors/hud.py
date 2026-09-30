"""HUD anomaly detector: text, position, size, blink and unexpected overlays.

HUD overlays are assumed to be bright graphics (text, icons) rendered on
top of the scene - the near-universal convention for status HUDs. During
calibration the detector:

1. Collects the per-frame *bright mask* (pixels above ``bright_threshold``)
   and the temporal maximum image;
2. Merges the union of all bright masks into element blobs (dilation +
   connected components), yielding one bounding box per HUD element;
3. Reads each element's text from the maximum image using the offline
   glyph matcher (the maximum image shows blinking
   elements at full brightness) and derives a stable element ID from it;
4. Measures each element's baseline: centroid, box area, text and blink
   toggle rate (visibility changes per second).

In detection mode each element is looked up in a search window around its
calibrated box and four intensities are emitted per element ``<id>``:

- ``hud/<id>/text``: normalised Levenshtein distance between the current
  and calibrated text (0 = identical, 1 = completely different);
- ``hud/<id>/position``: centroid distance to baseline in pixels;
- ``hud/<id>/size``: relative bounding-box area change ``|area/base - 1|``;
- ``hud/<id>/blink``: absolute deviation of the toggle rate (toggles per
  second, measured over a sliding window) from the calibrated rate.  This
  covers frequency changes as well as blink start/stop.

Additionally each unexpected overlay gets a channel ``hud/<slug>/new``
named from the **first** OCR reading; continuity across frames is by
centroid so later text/size changes do not open a second event.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from amon.detectors import Detector
from amon.model import Box, CalibrationResult, Frame
from amon.scale import REFERENCE_WIDTH_PX, of_width, of_width_float, of_width_sq
from amon.stats import robust_threshold
from amon.textocr import DEFAULT_MIN_MATCH_SCORE, levenshtein_norm, read_hud, slugify

# Absolute defaults were authored at REFERENCE_WIDTH_PX; config stores
# fractions of the processed frame width (or width² for areas).
_W = float(REFERENCE_WIDTH_PX)
_W2 = _W * _W


def _is_new_anomaly(anomaly_id: str) -> bool:
    parts = anomaly_id.split("/")
    return len(parts) == 3 and parts[0] == "hud" and parts[2] == "new"


def _box_centroid(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


@dataclass
class CalibratedHudElement:
    """Calibrated baseline of a single HUD element."""

    element_id: str
    box: Box  # (x, y, w, h) of the calibrated bright pixels
    centroid: Tuple[float, float]
    area: float  # calibrated bounding-box area in px^2
    pixel_count: int  # bright pixels when fully visible
    text: str
    toggle_rate: float  # visibility toggles per second (2x blink Hz)
    on_ratio: float
    visibility: deque = field(default_factory=deque)  # (t, visible) sliding window
    last_box: Optional[Box] = None


@dataclass
class _RuntimeNewTrack:
    """Unexpected overlay tracked by centroid; ID frozen at first OCR."""

    anomaly_id: str
    centroid: Tuple[float, float]
    last_box: Box
    last_text: str  # latest OCR (may differ from spawn slug)
    last_seen: float


class HudDetector(Detector):
    """Detects text, position, size, blink and unexpected HUD overlays."""

    name = "hud"

    @classmethod
    def default_config(cls) -> dict:
        return {
            "bright_threshold": 220,  # gray level separating HUD from scene
            # Fractions of processed frame width (tuned as px @ 1200 → /1200).
            "search_margin_of_width": 20 / _W,  # was 20 px
            "blink_window_seconds": 2.0,  # sliding window for the toggle rate
            "min_element_area_of_width_sq": 15 / _W2,  # was 15 px²
            "min_glyph_height_of_width": 8 / _W,  # was 8 px
            "max_glyph_height_of_width": 96 / _W,  # was 64 px; headroom for size anomalies
            "max_glyph_width_of_width": 96 / _W,  # was 64 px
            "min_glyph_area_of_width_sq": 20 / _W2,  # was 20 px²
            # Reject forced letter matches below this correlation×aspect score
            # (icons/dots/crosshairs); those overlays become symbol-1, …
            "min_glyph_match_score": DEFAULT_MIN_MATCH_SCORE,
            # TrueType under amon/fonts/ (or absolute path); default VCR OSD Mono
            "glyph_font": "VCR_OSD_MONO_1.001.ttf",
            "merge_kernel_of_width": 15 / _W,  # was 15 px (~1.25% of width)
            "visible_fraction": 0.25,  # bright-pixel fraction counting as visible
            "sigma_k": 8.0,
            "text_floor": 0.3,  # min normalised text distance
            "position_floor_of_width": 6.0 / _W,  # was 6 px
            "size_floor": 0.25,  # min relative area change
            "blink_floor": 2.0,  # min toggle-rate deviation (1/s)
            "new_floor": 0.5,  # intensity when unexpected text is present
            # Max centroid distance to keep a runtime /new track across frames.
            "new_match_distance_of_width": 30.0 / _W,  # was 30 px
            # Keep an unseen track this long so brief gaps reconnect to the same ID.
            "new_track_ttl_seconds": 1.0,
            # Per-anomaly threshold multipliers (float also accepted).
            "tolerance": {
                "text": 1.0,
                "position": 1.0,
                "size": 1.0,
                "blink": 1.0,
                "new": 1.0,
            },
        }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._grays: List[np.ndarray] = []
        self._times: List[float] = []
        self._max_img: Optional[np.ndarray] = None
        self._elements: Dict[str, CalibratedHudElement] = {}
        self._known_cover: Optional[np.ndarray] = None  # dilated calibrated footprint
        self._last_new: List[Tuple[str, Box, str]] = []  # (anomaly_id, box, text)
        self._new_floor = float(self.config["new_floor"])
        self._new_tracks: List[_RuntimeNewTrack] = []
        self._symbol_serial = 0  # symbol-1, symbol-2, … for non-text overlays
        self._new_frame_t: float = 0.0
        self._frame_width = REFERENCE_WIDTH_PX

    def _set_frame_width(self, image: np.ndarray) -> None:
        self._frame_width = int(image.shape[1])

    def _length_px(self, key: str, *, minimum: int = 1, round_up: bool = False) -> int:
        return of_width(
            self.config[key],
            self._frame_width,
            minimum=minimum,
            round_up=round_up,
        )

    def _area_px(self, key: str, *, minimum: int = 1) -> int:
        return of_width_sq(self.config[key], self._frame_width, minimum=minimum)

    def _length_pxf(self, key: str, *, minimum: float = 0.0) -> float:
        return of_width_float(
            self.config[key], self._frame_width, minimum=minimum
        )

    # --- calibration --------------------------------------------------------
    def _calibrate(self, frame: Frame) -> None:
        # collect the per-frame bright mask and the temporal maximum image
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        self._set_frame_width(gray)
        self._grays.append(gray)
        self._times.append(frame.timestamp)
        self._max_img = (
            gray if self._max_img is None else np.maximum(self._max_img, gray)
        )

    def _finish_calibration(self) -> CalibrationResult:
        if self._max_img is not None:
            self._set_frame_width(self._max_img)
        # Merge all collected bright masks into a single union mask
        thr = int(self.config["bright_threshold"])
        masks = [g > thr for g in self._grays]
        union = np.logical_or.reduce(masks) if masks else np.zeros((1, 1), bool)

        thresholds: Dict[str, float] = {}
        annotations = {"hud_elements": []}
        for box in self._find_element_bounding_boxes(union):
            element = self._aggregate_calibration_frames_for_element_bounding_box(
                box, union, masks
            )
            self._elements[element.element_id] = element
            thresholds.update(self._element_thresholds(element, masks))
            annotations["hud_elements"].append(
                {
                    "id": element.element_id,
                    "box": list(element.box),
                    "text": element.text,
                    "blink_hz": round(element.toggle_rate / 2.0, 2),
                    "on_ratio": round(element.on_ratio, 3),
                }
            )

        self._known_cover = self._build_known_cover(union.shape)
        # Absent during calibration by construction — any later appearance is
        # anomalous.  Per-element ``hud/<slug>/new`` channels share this floor
        # and are registered dynamically when overlays appear.
        self._new_floor = float(self.config["new_floor"])
        annotations["new_floor"] = self._new_floor

        self._grays, self._times = [], []  # free calibration memory
        return CalibrationResult(thresholds=thresholds, annotations=annotations)

    def _build_known_cover(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Binary mask of calibrated HUD footprints (box + search margin)."""
        cover = np.zeros(shape[:2], np.uint8)
        base_margin = self._length_px("search_margin_of_width")
        height, width = shape[:2]
        for element in self._elements.values():
            x, y, w, h = element.box
            # Extra pad scales with element size so modest size anomalies stay
            # inside the cover and do not spawn false ``hud/*/new`` channels.
            margin = base_margin + max(w, h) // 3
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
            cover[y0:y1, x0:x1] = 1
        return cover

    def _find_element_bounding_boxes(self, union: np.ndarray) -> List[Box]:
        """Group the union bright mask into per-element bounding boxes."""
        kernel_size = self._length_px("merge_kernel_of_width")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        blobs = cv2.dilate(union.astype(np.uint8), kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)
        min_area = self._area_px("min_element_area_of_width_sq")
        boxes = []
        for i in range(1, count):
            ys, xs = np.nonzero(union & (labels == i))
            if len(xs) < min_area:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            boxes.append((x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1))
        return sorted(boxes)

    def _aggregate_calibration_frames_for_element_bounding_box(
        self, box: Box, union: np.ndarray, masks: List[np.ndarray]
    ) -> CalibratedHudElement:
        x, y, w, h = box
        text, element_id = self._label_element_crop(
            self._max_img[y : y + h, x : x + w], fallback_id=f"elem{x}x{y}"
        )
        while element_id in self._elements:  # ensure uniqueness
            element_id += "x"

        pixel_count = int(union[y : y + h, x : x + w].sum())
        ys, xs = np.nonzero(union[y : y + h, x : x + w])
        centroid = (x + float(xs.mean()), y + float(ys.mean()))

        # Calculate the toggle rate and on ratio
        visible = [self._is_hud_element_visible(m, box, pixel_count) for m in masks]
        toggles = int(np.sum(np.array(visible[1:]) != np.array(visible[:-1])))
        duration = max(self._times[-1] - self._times[0], 1e-6)

        return CalibratedHudElement(
            element_id=element_id,
            box=box,
            centroid=centroid,
            area=float(w * h),
            pixel_count=pixel_count,
            text=text,
            toggle_rate=toggles / duration,
            on_ratio=float(np.mean(visible)),
        )

    def _element_thresholds(
        self, element: CalibratedHudElement, masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """Derive per-anomaly thresholds from calibration measurement jitter."""
        pos_changes: List[float] = []
        size_changes: List[float] = []
        text_distances: List[float] = []
        toggle_rate_changes: List[float] = []
        rate: List[Tuple[float, bool]] = []

        # Sliding window size for the toggle rate.
        window = self.config["blink_window_seconds"]

        # For each calibration frame, measure the HUD element state and calculate the toggle rate
        # and thresholds for when it is visible.
        for i, mask in enumerate(masks):
            state_change = self._measure_hud_element_changes(
                mask, self._grays[i], element
            )
            if state_change is not None:
                pos_change, size_change, text_dist = state_change
                pos_changes.append(pos_change)
                size_changes.append(size_change)
                text_distances.append(text_dist)

            t = self._times[i]
            is_visible = self._is_hud_element_visible(
                mask, element.box, element.pixel_count
            )
            rate.append((t, is_visible))

            # Drop frames from the sliding window when the window is exceeded.
            while rate and t - rate[0][0] > window:
                rate.pop(0)

            # Calculate the toggle rate and thresholds for when it is visible.
            if len(rate) >= 2 and rate[-1][0] - rate[0][0] >= window / 2:
                toggle_rate = self._get_toggle_rate(rate)
                toggle_rate_change = abs(toggle_rate - element.toggle_rate)
                toggle_rate_changes.append(toggle_rate_change)

        sigma_k, eid = float(self.config["sigma_k"]), element.element_id
        return {
            f"hud/{eid}/text": robust_threshold(
                text_distances, sigma_k, self.config["text_floor"]
            ),
            f"hud/{eid}/position": robust_threshold(
                pos_changes,
                sigma_k,
                self._length_pxf("position_floor_of_width"),
            ),
            f"hud/{eid}/size": robust_threshold(
                size_changes, sigma_k, self.config["size_floor"]
            ),
            f"hud/{eid}/blink": robust_threshold(
                toggle_rate_changes, sigma_k, self.config["blink_floor"]
            ),
        }

    # --- detection ------------------------------------------------------------
    def _detect(self, frame: Frame) -> Dict[str, float]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        self._set_frame_width(gray)
        mask = gray > int(self.config["bright_threshold"])
        window = float(self.config["blink_window_seconds"])

        out: Dict[str, float] = {}
        for eid, element in self._elements.items():
            measured = self._measure_hud_element_changes(
                mask, gray, element, record_box=True
            )
            pos_change, size_change, text_dist = (
                measured if measured is not None else (0.0, 0.0, 0.0)
            )

            visibility = element.visibility
            visibility.append((frame.timestamp, measured is not None))
            while visibility and frame.timestamp - visibility[0][0] > window:
                visibility.popleft()

            # Calculate the blink change if the sliding window has filled up.
            blink_change = 0.0
            sliding_window_filled = (
                len(visibility) >= 2
                and visibility[-1][0] - visibility[0][0] >= window * 0.9
            )
            if sliding_window_filled:
                blink_change = abs(
                    self._get_toggle_rate(visibility) - element.toggle_rate
                )

            out[f"hud/{eid}/text"] = text_dist
            out[f"hud/{eid}/position"] = pos_change
            out[f"hud/{eid}/size"] = size_change
            out[f"hud/{eid}/blink"] = blink_change

        out.update(self._measure_new_text(mask, gray, frame.timestamp))
        return out

    def _measure_new_text(
        self, mask: np.ndarray, gray: np.ndarray, timestamp: float
    ) -> Dict[str, float]:
        """Track unexpected overlays by centroid; freeze ID from first OCR.

        A blob outside ``known_cover`` is accepted when OCR is non-empty (spawn)
        or when its centroid lies near an existing track (continuity even if
        OCR briefly fails or the glyphs change).  Only ``hud/<slug>/new`` is
        emitted — never text/position/size/blink for runtime-only overlays.
        """
        self._last_new = []
        self._new_frame_t = timestamp
        if self._known_cover is None:
            return {}

        match_dist = self._length_pxf("new_match_distance_of_width")
        ttl = float(self.config["new_track_ttl_seconds"])
        self._prune_new_tracks(timestamp, ttl)
        novel = mask & (self._known_cover == 0)
        intensities: Dict[str, float] = {}
        if not novel.any():
            return intensities

        used_track_ids: set = set()
        for box in self._find_element_bounding_boxes(novel):
            x, y, w, h = box
            centroid = _box_centroid(box)
            text, is_symbol = self._read_element_label(gray[y : y + h, x : x + w])

            track = self._nearest_new_track(centroid, match_dist, used_track_ids)
            if track is None:
                # Glyph rewrites can shift the bbox centroid; allow a wider
                # reconnect before spawning a second channel.
                track = self._nearest_new_track(
                    centroid, match_dist * 2.5, used_track_ids
                )
            if track is not None:
                track.centroid = centroid
                track.last_box = box
                track.last_seen = timestamp
                # Freeze ID; only refresh display text when OCR is confident.
                if text and not is_symbol:
                    track.last_text = text
                used_track_ids.add(id(track))
                intensities[track.anomaly_id] = 1.0
                self._last_new.append((track.anomaly_id, box, track.last_text))
                continue

            if not text and not is_symbol:
                continue  # empty crop (no ink in glyph-size range)

            if is_symbol:
                slug = self._next_symbol_id()
                label = slug
            else:
                slug = slugify(text) or f"elem{x}x{y}"
                label = text
            aid = f"hud/{slug}/new"
            # Same frozen ID still alive but unmatched → reconnect, don't fork.
            orphan = next(
                (
                    t
                    for t in self._new_tracks
                    if t.anomaly_id == aid and id(t) not in used_track_ids
                ),
                None,
            )
            if orphan is not None:
                orphan.centroid = centroid
                orphan.last_box = box
                orphan.last_seen = timestamp
                if text and not is_symbol:
                    orphan.last_text = text
                used_track_ids.add(id(orphan))
                intensities[aid] = 1.0
                self._last_new.append((aid, box, orphan.last_text))
                continue
            # Two novel blobs with the same OCR in one frame → disambiguate.
            if aid in intensities:
                if is_symbol:
                    slug = self._next_symbol_id()
                    label = slug
                    aid = f"hud/{slug}/new"
                else:
                    aid = f"hud/{slug}_{x}x{y}/new"
            self._thresholds.setdefault(aid, self._new_floor)
            new_track = _RuntimeNewTrack(
                anomaly_id=aid,
                centroid=centroid,
                last_box=box,
                last_text=label,
                last_seen=timestamp,
            )
            self._new_tracks.append(new_track)
            used_track_ids.add(id(new_track))
            intensities[aid] = 1.0
            self._last_new.append((aid, box, label))

        return intensities

    def _nearest_new_track(
        self,
        centroid: Tuple[float, float],
        match_dist: float,
        used_track_ids: set,
    ) -> Optional[_RuntimeNewTrack]:
        best: Optional[_RuntimeNewTrack] = None
        best_d = match_dist
        for track in self._new_tracks:
            if id(track) in used_track_ids:
                continue
            d = float(
                np.hypot(
                    centroid[0] - track.centroid[0], centroid[1] - track.centroid[1]
                )
            )
            if d <= best_d:
                best_d = d
                best = track
        return best

    def _prune_new_tracks(self, timestamp: float, ttl: float) -> None:
        kept: List[_RuntimeNewTrack] = []
        for track in self._new_tracks:
            if timestamp - track.last_seen <= ttl:
                kept.append(track)
            else:
                # Forget dynamic thresholds so multi-day runs cannot accumulate
                # one map entry per historical overlay slug.
                self._thresholds.pop(track.anomaly_id, None)
        self._new_tracks = kept

    def _measure_hud_element_changes(
        self,
        mask: np.ndarray,
        gray: np.ndarray,
        element: CalibratedHudElement,
        record_box: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Locate the element near its calibrated box and measure the deviation
        from the calibrated state.

        Returns ``(position, size, text)`` intensities, or ``None`` when the
        element is not visible (e.g. mid-blink).
        """

        # Dilate the search margin around the calibrated box to compensate for the
        # element's size and position jitter.
        margin = self._length_px("search_margin_of_width")
        x, y, w, h = element.box
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(mask.shape[1], x + w + margin), min(mask.shape[0], y + h + margin)
        sub = mask[y0:y1, x0:x1]

        # Check if the element is visible by counting the number of bright pixels
        # in the search margin.
        if int(sub.sum()) < self.config["visible_fraction"] * element.pixel_count:
            return None

        # Find the bounding box of the element in the search margin.
        ys, xs = np.nonzero(sub)
        bx0, by0 = x0 + int(xs.min()), y0 + int(ys.min())
        bw, bh = int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1

        # Record the last bounding box of the element.
        if record_box:
            element.last_box = (bx0, by0, bw, bh)

        # Determine position and size of text
        centroid = (x0 + float(xs.mean()), y0 + float(ys.mean()))
        pos_error = float(
            np.hypot(
                centroid[0] - element.centroid[0], centroid[1] - element.centroid[1]
            )
        )
        size_error = abs(bw * bh / element.area - 1.0)

        # Read the text from the bounding box and calculate the Levenshtein distance
        # (character similarity) between the current text and the calibrated text.
        text = self._read_element_text(gray[by0 : by0 + bh, bx0 : bx0 + bw])
        levenshtein_distance = levenshtein_norm(text, element.text)

        return pos_error, size_error, levenshtein_distance

    def _next_symbol_id(self) -> str:
        """Allocate the next ``symbol-N`` label for a non-text overlay."""
        self._symbol_serial += 1
        return f"symbol-{self._symbol_serial}"

    def _label_element_crop(
        self, gray: np.ndarray, *, fallback_id: str
    ) -> Tuple[str, str]:
        """Return ``(display_text, element_id)`` for a calibrated HUD crop."""
        text, is_symbol = self._read_element_label(gray)
        if is_symbol:
            symbol_id = self._next_symbol_id()
            return symbol_id, symbol_id
        element_id = slugify(text) or fallback_id
        return text, element_id

    def _read_element_label(self, gray: np.ndarray) -> Tuple[str, bool]:
        """OCR a HUD crop → ``(text, is_symbol)``."""
        result = read_hud(
            gray,
            min_glyph_height=self._length_px("min_glyph_height_of_width"),
            max_glyph_height=self._length_px(
                "max_glyph_height_of_width", round_up=True
            ),
            max_glyph_width=self._length_px(
                "max_glyph_width_of_width", round_up=True
            ),
            min_glyph_area=self._area_px("min_glyph_area_of_width_sq"),
            glyph_font=str(self.config.get("glyph_font") or ""),
            min_match_score=float(
                self.config.get("min_glyph_match_score", DEFAULT_MIN_MATCH_SCORE)
            ),
        )
        return result.text.strip(), result.is_symbol

    def _read_element_text(self, gray: np.ndarray) -> str:
        """OCR a HUD crop (empty string for pure symbol/icon ink)."""
        text, _is_symbol = self._read_element_label(gray)
        return text

    @staticmethod
    def _is_hud_element_visible(mask: np.ndarray, box: Box, pixel_count: int) -> bool:
        x, y, w, h = box
        return int(mask[y : y + h, x : x + w].sum()) >= 0.25 * pixel_count

    @staticmethod
    def _get_toggle_rate(samples: List[Tuple[float, bool]]) -> float:
        """Visibility toggles per second over a (t, visible) sequence."""
        values = [v for _, v in samples]
        toggles = sum(1 for a, b in zip(values, values[1:]) if a != b)
        span = samples[-1][0] - samples[0][0]
        return toggles / span if span > 0 else 0.0

    # --- event enrichment -------------------------------------------------------
    def metadata(self, anomaly_id: str) -> dict:
        if _is_new_anomaly(anomaly_id):
            for aid, _box, text in self._last_new:
                if aid == anomaly_id:
                    return {
                        "element": anomaly_id.split("/")[1],
                        "text": text,
                        "spawn_id": anomaly_id.split("/")[1],
                        "new": True,
                    }
            return {"element": anomaly_id.split("/")[1], "new": True}
        element = self._element_for(anomaly_id)
        if element is None:
            return {}
        return {
            "element": element.element_id,
            "text": element.text,
            "blink_hz": round(element.toggle_rate / 2.0, 2),
            "calibrated_box": list(element.box),
        }

    def regions(self, anomaly_id: str) -> List[Box]:
        if _is_new_anomaly(anomaly_id):
            return [box for aid, box, _text in self._last_new if aid == anomaly_id]
        element = self._element_for(anomaly_id)
        if element is None:
            return []
        return [element.last_box or element.box]

    def _element_for(self, anomaly_id: str) -> Optional[CalibratedHudElement]:
        parts = anomaly_id.split("/")
        return self._elements.get(parts[1]) if len(parts) == 3 else None
