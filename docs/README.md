# Knowledge base

Short, intuition-first notes on the computer-vision techniques used in this
project. They are written for developers who are not image-processing
specialists and explain *why* each method fits the anomaly-detection problem.

## Algorithms

| Topic | What it detects |
|---|---|
| [Overview](algorithms/overview.md) | How the pieces fit together |
| [Threshold calibration](algorithms/threshold-calibration.md) | Learning “normal” without manual tuning |
| [Temporal anomalies](algorithms/temporal.md) | Noise, flicker, contrast changes |
| [HUD anomalies](algorithms/hud.md) | Text, position, size, blink changes |
| [Spatial anomalies](algorithms/spatial.md) | Background distortion (HUD ignored) |
| [Event aggregation](algorithms/event-aggregation.md) | One event per occurrence, fewer false positives |

Implementation details and API docs remain in [for-dev.md](../for-dev.md).
