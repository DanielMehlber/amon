# HUD anomalies

HUD elements are assumed to be **bright graphics** (white text, icons) on
top of a darker scene — the usual layout for status overlays on industrial
or security feeds.

## Finding HUD elements (bright-region segmentation)

**Idea:** Pixels brighter than a threshold are “overlay.” During
calibration we collect these masks across many frames and merge them. A
small **dilation** joins nearby bright blobs (letters in a word) into one
**connected component** per HUD element. Each component gets a bounding box.

**Why it works:** The static scene stays below the brightness cutoff; only
deliberately bright UI pixels qualify.

## Reading text (template matching, not full OCR)

**Idea:** HUD fonts are simple and fixed. We do not need a neural OCR
engine — we **segment** each character as a blob, resize it to a standard
size, and **compare** it to pre-rendered templates of `A–Z` and `0–9`
(drawn with the same font OpenCV uses elsewhere). The best match wins.

**Otsu thresholding** picks a text/background split per crop so
anti-aliased edges survive binarization.

**Anomaly use:** Text change = high **edit distance** between the live
reading and the calibrated string.

## Position and size

**Position:** Compare the **centroid** (average x,y) of bright pixels in
the element’s search window to the calibrated centroid. A shifted label
moves the centroid.

**Size:** Compare the **bounding-box area** to calibration. A zoomed or
shrunk overlay changes area even if text is unchanged.

## Blink detection (toggle rate)

**Idea:** A blinking icon is **visible in some frames and absent in
others**. Count how often visibility flips inside a sliding time window —
that is the **toggle rate** (toggles per second). A 2 Hz blink flips ~4
times per second.

**Anomalies detected with one metric:**

- **Frequency change** — toggle rate drifts from calibration (e.g. alarm
  flashing faster).
- **Blink start/stop** — element stays always on or always off compared
  to a normally blinking icon (toggle rate drops toward zero).

The detector waits until the sliding window is full before judging blink,
so brief gaps during a normal blink do not false-trigger.

## Per-element anomaly IDs

Each HUD element gets its own namespace, e.g. `hud/cam01/text`,
`hud/rec/blink`. Text and position on the same physical label are
separate channels — operators can see *what* changed.
