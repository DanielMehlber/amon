"""Standalone HTML report writer with embedded media (no Panel runtime)."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from amon.timefmt import format_wall_time

ACCENT = (198, 40, 40)
MUTED = (158, 158, 158)


def _media_data_uri(path: str) -> Optional[str]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    mime = "image/gif" if file_path.suffix.lower() == ".gif" else "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _intensity_plot_png(event: dict) -> bytes:
    """Render the intensity timeline as a small PNG (no Bokeh dependency)."""
    width, height = 520, 200
    margin = 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    timeline = event["timeline"] or [[event["start"], event["max_intensity"]]]
    xs = [point[0] for point in timeline]
    ys = [point[1] for point in timeline]
    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1.0
    y_max = max(max(ys), event["threshold"]) * 1.15 or 1.0

    def px(x: float) -> int:
        return int(margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin))

    def py(y: float) -> int:
        return int(height - margin - y / y_max * (height - 2 * margin))

    # Threshold line.
    threshold_y = py(event["threshold"])
    draw.line(
        [(margin, threshold_y), (width - margin, threshold_y)], fill=MUTED, width=1
    )
    draw.text((margin, threshold_y - 14), "threshold", fill=MUTED)

    # Intensity polyline.
    points = [(px(x), py(y)) for x, y in zip(xs, ys)]
    if len(points) >= 2:
        draw.line(points, fill=ACCENT, width=2)
    elif points:
        draw.ellipse(
            [points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2],
            fill=ACCENT,
        )

    draw.text((margin, 8), event["anomaly_id"], fill=ACCENT)
    draw.text((margin, height - 28), "session time (s)", fill=(80, 80, 80))
    draw.text((8, height // 2), "intensity", fill=(80, 80, 80))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_html_report(
    session: dict,
    events: list,
    calibration: Optional[dict],
    out_path: Path,
) -> Path:
    """Write a self-contained HTML archive to ``out_path``."""
    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'><head>",
        "<meta charset='utf-8'>",
        f"<title>Report: {html.escape(session['id'])}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222;}",
        "h1,h2,h3{color:#b71c1c;} .meta{color:#555;} .event{border-top:1px solid #eee;padding:1.25rem 0;}",
        "img{max-width:100%;border:2px solid #ffebee;border-radius:6px;}",
        "pre{background:#fafafa;padding:0.75rem;border-radius:6px;overflow:auto;}",
        "table{border-collapse:collapse;width:100%;margin:0.5rem 0;}",
        "td,th{border:1px solid #eee;padding:0.4rem 0.6rem;text-align:left;}",
        "</style></head><body>",
        f"<h1>Monitoring report: {html.escape(session['name'])}</h1>",
        "<p class='meta'>",
        f"<strong>Run:</strong> {html.escape(session['id'])}<br>",
        f"<strong>Status:</strong> {html.escape(session['status'])}<br>",
        f"<strong>Started:</strong> {html.escape(format_wall_time(session['started_at']))}<br>",
    ]
    if session.get("finished_at"):
        parts.append(
            f"<strong>Finished:</strong> {html.escape(format_wall_time(session['finished_at']))}<br>"
        )
    parts += [
        f"<strong>Events:</strong> {len(events)}<br>",
        f"<strong>Source:</strong> <code>{html.escape(session.get('source') or '')}</code>",
        "</p>",
    ]

    parts.append("<h2>Calibration</h2>")
    if calibration is None:
        parts.append("<p><em>Not recorded.</em></p>")
    else:
        if calibration.get("media"):
            uri = _media_data_uri(calibration["media"])
            if uri:
                parts.append(f"<p><img src='{uri}' alt='Calibration review'></p>")
        elements = calibration["annotations"].get("hud_elements", [])
        if elements:
            parts.append("<table><tr><th>HUD</th><th>Text</th><th>Blink (Hz)</th></tr>")
            for element in elements:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(element['id'])}</td>"
                    f"<td>{html.escape(str(element.get('text') or ''))}</td>"
                    f"<td>{html.escape(str(element.get('blink_hz', 0)))}</td>"
                    "</tr>"
                )
            parts.append("</table>")
        keypoints = len(calibration["annotations"].get("keypoints", []))
        parts.append(f"<p><strong>Feature points:</strong> {keypoints}</p>")
        parts.append(
            "<pre>"
            + html.escape(json.dumps(calibration["thresholds"], indent=2))
            + "</pre>"
        )

    parts.append(f"<h2>Events ({len(events)})</h2>")
    for event in events:
        parts += [
            "<section class='event'>",
            f"<h3>{html.escape(event['anomaly_id'])}</h3>",
            "<p>",
            f"{event['start']:.2f}s – {event['end']:.2f}s "
            f"({event['duration']:.2f}s), peak {event['max_intensity']:.3f} "
            f"vs threshold {event['threshold']:.3f}",
            "</p>",
        ]
        if event.get("media"):
            uri = _media_data_uri(event["media"])
            if uri:
                parts.append(f"<p><img src='{uri}' alt='Event evidence'></p>")
        plot_uri = "data:image/png;base64," + base64.b64encode(
            _intensity_plot_png(event)
        ).decode("ascii")
        parts.append(f"<p><img src='{plot_uri}' alt='Intensity plot'></p>")
        if event.get("metadata"):
            parts.append(
                "<pre>"
                + html.escape(json.dumps(event["metadata"], indent=2))
                + "</pre>"
            )
        parts.append("</section>")

    parts.append("</body></html>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
