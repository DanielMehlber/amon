# Spatial anomalies

**Goal:** Notice when the **background scene** is geometrically wrong
(warped, slipped, corrupted) while **ignoring** HUD overlays that are
supposed to move or blink.

## Building a clean background (temporal median)

**Idea:** Over calibration frames, take the **per-pixel median** intensity.
Short-lived things (motion, blink-off frames) disappear; static structure
remains. That median image is the “normal geometry” reference.

## Ignoring the HUD (masking)

**Idea:** Any pixel that was ever bright during calibration is treated as
HUD territory. We **dilate** that mask slightly so anti-aliased edges are
included, then **exclude** it when placing feature points. HUD motion never
creates spatial alarms.

## Feature points (Shi–Tomasi corners)

**Idea:** We need salient spots on the background that are easy to track —
corners of shapes, grid intersections. **Shi–Tomasi** scores each pixel for
“corner-likeness” and picks the best candidates up to a limit.

Think of them as pushpins stuck into the reference photo.

## Tracking motion (Lucas–Kanade optical flow)

**Idea:** For each pushpin in the reference frame, ask: *where did this
patch of pixels move in the new frame?* **Optical flow** estimates that
shift per point. If the scene is rigid and stable, shifts are tiny. A
**local warp** moves a cluster of points together in one direction.

**Forward–backward check:** Track from reference → current → reference
again. If we do not land near the start, the match was unreliable and is
discarded.

## Distortion intensity

**Metric:** Sort point displacements and take a **high rank** (e.g. third
largest), not the maximum. One bad track from glare should not dominate;
a real warp moves many points.

**Evidence:** Regions around displaced points are drawn on the event GIF.

## What this catches

- Lens vibration misalignment
- Partial frame buffer corruption
- Local geometric warping (the synthetic test uses a sinusoidal remap in
  one region)

What it does **not** flag: HUD text edits, global flicker, or noise —
those have dedicated temporal/HUD detectors.
