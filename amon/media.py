"""GIF generation and annotation for event evidence and calibration review."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from amon.model import AnomalyEvent, Box

HIGHLIGHT = (0, 0, 255)   # BGR red for anomaly regions
KEYPOINT = (0, 200, 0)    # BGR green for tracked feature points
HUD_BOX = (255, 128, 0)   # BGR blue-ish for HUD element markers

#: (timestamp, BGR image) pairs as captured by the pipeline's ring buffer.
TimedFrames = Sequence[Tuple[float, np.ndarray]]


def write_gif(frames: List[np.ndarray], path: Union[str, Path], fps: float) -> str:
    """Write BGR frames as an animated GIF and return the path."""
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    if not images:
        raise ValueError("cannot write a GIF without frames")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return str(path)


def subsample(frames: TimedFrames, source_fps: float, target_fps: float) -> List[Tuple[float, np.ndarray]]:
    """Reduce the frame rate to ``target_fps`` by dropping frames."""
    stride = max(1, int(round(source_fps / max(target_fps, 1e-6))))
    return list(frames)[::stride]


def write_event_gif(
    frames: TimedFrames, event: AnomalyEvent, path: Union[str, Path], fps: float, gif_fps: float
) -> str:
    """Render an event evidence GIF, highlighting affected regions."""
    selected = subsample(frames, fps, gif_fps)
    rendered = []
    for t, image in selected:
        canvas = image.copy()
        if event.start <= t <= event.end:
            for box in event.regions:
                _draw_box(canvas, box, HIGHLIGHT)
            cv2.putText(canvas, event.anomaly_id, (6, canvas.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, HIGHLIGHT, 1, cv2.LINE_AA)
        rendered.append(canvas)
    return write_gif(rendered, path, min(gif_fps, fps))


def annotate_calibration(image: np.ndarray, annotations: dict) -> np.ndarray:
    """Draw keypoints, HUD boxes, texts and blink frequencies on a frame."""
    canvas = image.copy()
    for x, y in annotations.get("keypoints", []):
        cv2.circle(canvas, (int(x), int(y)), 2, KEYPOINT, -1)
    for element in annotations.get("hud_elements", []):
        x, y, w, h = element["box"]
        _draw_box(canvas, (x, y, w, h), HUD_BOX)
        blink = element.get("blink_hz", 0)
        label = f"{element.get('text') or element['id']}" + (f" @ {blink:g} Hz" if blink else "")
        text_y = y + h + 14 if y < 20 else y - 6
        cv2.putText(canvas, label, (max(2, x - 2), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_BOX, 1, cv2.LINE_AA)
    return canvas


def write_calibration_gif(
    frames: TimedFrames, annotations: dict, path: Union[str, Path], fps: float, gif_fps: float
) -> str:
    """Render the annotated calibration review GIF."""
    selected = subsample(frames, fps, gif_fps)
    rendered = [annotate_calibration(image, annotations) for _, image in selected]
    return write_gif(rendered, path, min(gif_fps, fps))


def _draw_box(canvas: np.ndarray, box: Box, color: Tuple[int, int, int]) -> None:
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
