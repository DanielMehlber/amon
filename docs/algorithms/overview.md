# Algorithm overview

This system watches a **mostly static** camera feed: the background scene
changes little, while a HUD (text, icons, blinkers) sits on top. Anomalies
are anything that breaks the pattern learned during a short **calibration**
phase at startup.

## The basic loop

Every incoming frame passes through three independent detectors. Each
detector returns a **number per anomaly type** (an *intensity*). The
pipeline compares that number to a **threshold learned during calibration**.
If the intensity stays above the threshold long enough, one **event** is
recorded (with a GIF as evidence).

```
Frame → [Temporal detector] → intensities → compare to thresholds → events
      → [HUD detector]      → intensities → compare to thresholds → events
      → [Spatial detector]  → intensities → compare to thresholds → events
```

Detectors do not decide “anomaly yes/no” themselves; they only measure
*how unusual* the current frame looks. That keeps plugins small and lets
the framework handle timing, aggregation and storage.

## Why classical CV instead of deep learning?

- **No training data** — calibration on a few seconds of normal footage
  is enough.
- **Fully offline** — no GPU farm, no model downloads.
- **Explainable** — each intensity has a direct geometric or statistical
  meaning (pixel shift, brightness swing, text edit distance).
- **Fast** — suitable for long-running, real-time monitoring.

## Where to read next

| If you want to understand… | Read |
|---|---|
| How thresholds are chosen automatically | [Threshold calibration](threshold-calibration.md) |
| Whole-frame disturbances (noise, flicker, contrast) | [Temporal anomalies](temporal.md) |
| On-screen status overlays | [HUD anomalies](hud.md) |
| Warped or corrupted background | [Spatial anomalies](spatial.md) |
| Why flicker does not spawn ten duplicate events | [Event aggregation](event-aggregation.md) |
