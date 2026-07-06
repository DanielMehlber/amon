"""HUD anomaly detector: text, position, size and blink changes.

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

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from amon.detectors import Detector
from amon.model import Box, CalibrationResult, Frame
from amon.stats import robust_threshold
from amon.textocr import levenshtein_norm, read_text, slugify


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


class HudDetector(Detector):
    """Detects text, position, size and blink anomalies of HUD overlays."""

    name = "hud"

    @classmethod
    def default_config(cls) -> dict:
        return {
            "bright_threshold": 220,  # gray level separating HUD from scene
            "search_margin": 20,  # px around the calibrated box searched
            "blink_window_seconds": 2.0,  # sliding window for the toggle rate
            "min_element_area": 15,  # ignore bright specks below this size
            "merge_kernel": 15,  # dilation size merging glyphs to elements
            "visible_fraction": 0.25,  # bright-pixel fraction counting as visible
            "sigma_k": 8.0,
            "text_floor": 0.3,  # min normalised text distance
            "position_floor": 6.0,  # min centroid shift (px)
            "size_floor": 0.25,  # min relative area change
            "blink_floor": 2.0,  # min toggle-rate deviation (1/s)
        }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._grays: List[np.ndarray] = []
        self._times: List[float] = []
        self._max_img: Optional[np.ndarray] = None
        self._elements: Dict[str, CalibratedHudElement] = {}

    # --- calibration --------------------------------------------------------
    def _calibrate(self, frame: Frame) -> None:
        # collect the per-frame bright mask and the temporal maximum image
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        self._grays.append(gray)
        self._times.append(frame.timestamp)
        self._max_img = (
            gray if self._max_img is None else np.maximum(self._max_img, gray)
        )

    def _finish_calibration(self) -> CalibrationResult:
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

        self._grays, self._times = [], []  # free calibration memory
        return CalibrationResult(thresholds=thresholds, annotations=annotations)

    def _find_element_bounding_boxes(self, union: np.ndarray) -> List[Box]:
        """Group the union bright mask into per-element bounding boxes."""
        kernel = np.ones((self.config["merge_kernel"],) * 2, np.uint8)
        blobs = cv2.dilate(union.astype(np.uint8), kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)
        boxes = []
        for i in range(1, count):
            ys, xs = np.nonzero(union & (labels == i))
            if len(xs) < self.config["min_element_area"]:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            boxes.append((x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1))
        return sorted(boxes)

    def _aggregate_calibration_frames_for_element_bounding_box(
        self, box: Box, union: np.ndarray, masks: List[np.ndarray]
    ) -> CalibratedHudElement:
        x, y, w, h = box
        text = read_text(self._max_img[y : y + h, x : x + w])
        element_id = slugify(text) or f"elem{x}x{y}"
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
                pos_changes, sigma_k, self.config["position_floor"]
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

        return out

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
        margin = int(self.config["search_margin"])
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
        text = read_text(gray[by0 : by0 + bh, bx0 : bx0 + bw])
        levenshtein_distance = levenshtein_norm(text, element.text)

        return pos_error, size_error, levenshtein_distance

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
        element = self._element_for(anomaly_id)
        if element is None:
            return []
        return [element.last_box or element.box]

    def _element_for(self, anomaly_id: str) -> Optional[CalibratedHudElement]:
        parts = anomaly_id.split("/")
        return self._elements.get(parts[1]) if len(parts) == 3 else None
