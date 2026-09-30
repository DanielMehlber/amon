# HUD anomalies

HUD elements are assumed to be **bright graphics** (white text, icons) on
top of a darker scene — the usual layout for status overlays on industrial
or security feeds.

## Finding HUD elements (bright-region segmentation)

**Idea:** Pixels brighter than a threshold are “overlay.” During
calibration we collect these masks across many frames and merge them. A
small **dilation** (`merge_kernel_rel × frame width`) joins nearby
bright blobs (letters in a word) into one **connected component** per HUD
element. Each component gets a bounding box.

**Why it works:** The static scene stays below the brightness cutoff; only
deliberately bright UI pixels qualify.

## Reading text (template matching, not full OCR)

**Idea:** HUD fonts are simple and fixed. We do not need a neural OCR
engine — we **segment** each character as a blob, resize it to a standard
size, and **compare** it to pre-rendered templates of `A–Z` and `0–9`
drawn from a configurable TrueType font (default: bundled **VCR OSD Mono**
in `amon/fonts/`). The best match wins. Drop an alternate `.ttf` in that
folder and set `glyph_font` on `HudDetector` to switch.

**Size gates:** Components shorter than `min_glyph_height_rel × frame
width`, taller than `max_glyph_height_rel × width`, wider than
`max_glyph_width_rel × width`, or with fewer than
`min_glyph_area_rel_sq × width²` bright pixels are ignored — this
filters IR speckles and large glare patches.  Absolute pixel defaults were
authored at ~1200 px width and are stored as fractions of the *processed*
frame width so preprocessing scale does not require retuning.  OCR crops
add `glyph_crop_pad_rel × width` around the bright-mask box so
anti-aliased fringe below `bright_threshold` is not cut off (tight boxes
otherwise mangle letters such as `S`).  Each remaining blob is scored
against the charset;
matches below `min_glyph_match_score` (default ~0.45) are **not** forced to
the nearest letter.  Crops where every blob fails that gate are non-text
icons and get ids `symbol-1`, `symbol-2`, … (a filled dot is no longer
read as `4`).  Calibration uses the same rule: a bright blob whose crop
is icon-only or fails to produce a readable slug becomes `symbol-N`
instead of a text id like `cam01`.
If at least one glyph clears the gate the crop is treated
as text and weaker neighbours still use their best letter so words such as
`STAT01` stay intact.

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

## Unexpected / new text

**Idea:** Calibration records a *known cover* — each element’s box plus the
search margin. Any later bright blob **outside** that cover that OCR reads
as non-empty text (letters or digits) is an unexpected overlay.

**Anomaly ID:** `hud/<slug>/new` — named once from the **first** OCR reading
(e.g. `ALERT` → `hud/alert/new`, `1000` → `hud/1000/new`). Continuity across
frames is by **centroid** (`new_match_distance_rel × frame width`): if
the overlay rewrites
its glyphs, blinks, or resizes in place, the same channel stays open until
the blob disappears. Runtime overlays never emit calibrated-style
`text` / `position` / `size` / `blink` anomalies. Tracks older than
`new_track_ttl_seconds` are pruned and their dynamic thresholds dropped so
multi-day runs cannot accumulate one map entry per historical overlay.

Bright non-text icons (dots, crosshairs, etc.) whose glyph scores stay below
`min_glyph_match_score` spawn as `hud/symbol-N/new` rather than a false
letter slug.  An already-tracked blob may keep its ID briefly even if OCR
fails for a frame.

## Per-element anomaly IDs

Each **calibrated** HUD element gets its own namespace, e.g. `hud/cam01/text`,
`hud/rec/blink`. Unexpected overlays use position-tracked `/new` channels
frozen at first sighting.
