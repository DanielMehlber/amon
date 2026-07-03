# Threshold calibration

**Problem:** Every detector outputs a continuous number. We need a cutoff
that says “this is unusual” without an engineer hand-tuning values for
each camera.

**Intuition:** During calibration the stream should look **normal**. The
detector collects many intensity samples — one per frame — and asks: *how
high did “normal” ever get?* The threshold is set **above** that, with
margin.

## Robust statistics (median + MAD)

A plain average and standard deviation are thrown off by a single glitch
frame. Instead we use:

- **Median** — the typical value; half the samples are below it.
- **MAD** (median absolute deviation) — how far samples usually sit from
  the median. It behaves like a standard deviation but ignores outliers.

Threshold ≈ `median + k × MAD` (with `k` around 8 in this project).

We also require the threshold to beat `1.5 × max(calibration samples)`
so even a lucky spike during calibration cannot trigger false alarms later.

## Why this fits anomaly detection

Calibration assumes the first *N* seconds are representative of healthy
operation. The learned threshold encodes “what normal looked like on *this*
camera, *this* HUD, *this* lighting.” When monitoring starts, only
**sustained** excursions above that personal baseline become events.

No magic — if calibration happens during an incident, the baseline is
wrong. That is why operators can review calibration media in the report UI.
