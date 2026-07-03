# Temporal anomalies

These detectors look at **the whole frame** (or its global statistics) and
flag disturbances that affect large areas or the entire image at once.

## Frame differencing (noise)

**Idea:** Subtract the previous grayscale frame from the current one. If
nothing moved, every pixel changes by roughly the same amount (global
brightness drift) and the **spatial average** of the difference is near
zero. **Sensor noise** makes pixels jump independently, so the spread of
difference values grows.

**Metric:** After subtracting the mean difference (to cancel uniform
brightness shifts like flicker), take a robust spread (MAD) of the
residual. High spread ⇒ noisy frame.

**Anomaly:** Random speckle, compression artefacts, RF interference on
the video link.

## Global brightness oscillation (flicker)

**Idea:** Flicker is a **rhythmic change in overall brightness** — every
pixel gets brighter and darker together. Track the mean gray level over a
short sliding window and measure how much it **wiggles** frame to frame.

**Metric:** Mean absolute change of the global mean inside the window.

**Anomaly:** Power-line hum on backlights, failing inverters, strobing
room lights reflected into the scene.

Flicker deliberately ignores local changes (a blinking HUD dot barely
moves the global mean).

## Contrast change

**Idea:** **Contrast** is how spread-out pixel intensities are — a flat
gray wall has low contrast; a crisp scene has high. During calibration we
remember the typical standard deviation of gray levels. In monitoring we
flag relative deviations from that baseline.

**Metric:** `|current_std / baseline_std − 1|`.

**Anomaly:** Gamma curve shifts, auto-exposure stuck, washed-out or
crushed video levels.

## How they work together

| Symptom | Primary detector | Why |
|---|---|---|
| Snow / static noise | Noise | Per-pixel jitter |
| Whole image pulsing | Flicker | Global mean oscillation |
| Image looks flat or harsh | Contrast | Std deviation drift |

A strong flicker event can disturb HUD and spatial metrics too; the
[exclusion hierarchy](event-aggregation.md) reports flicker as the root
cause instead of a pile of side-effect events.
