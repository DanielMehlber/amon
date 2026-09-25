# amon - Developer Guide

This document describes the architecture, the image-processing algorithms
and the extension points. The code favours small interfaces and KISS: the
framework owns all generic logic; plugins only implement their algorithm.

## Architecture overview

```
VideoSource ──frames──▶ Pipeline ──intensities──▶ EventAggregator
                          │                            │ events
                          │  Detector plugins          ▼
                          │  (calibration ▸ detection) BackgroundWorker (separate process)
                          │                            ├─ GIF generation (media.py)
                          ▼                            └─ SQLite writes (db.py)
                        ring buffer (evidence clips)

Report UI (Panel, report.py) ◀── reads ── SQLite + media files ── reads ──▶ Exporters
```

- `amon/pipeline.py` - the session lifecycle: calibration phase, switch to
  detection, per-frame aggregation, evidence-clip capture, worker hand-off.
- `amon/aggregate.py` - threshold crossing → event state machine, plus the
  suppression hierarchy.
- `amon/worker.py` - a `multiprocessing.Process` receiving jobs through a
  queue; the pipeline never blocks on I/O (non-functional requirement).
- `amon/db.py` - SQLite persistence (sessions, calibrations, events).
- `amon/report.py`, `amon/panel_offline.py`, `amon/exporters.py`, `amon/plots.py` -
  Panel UI (with explicit offline resource configuration), standalone HTML export.

## Detector interface (`amon/detectors/__init__.py`)

```python
class MyDetector(Detector):
    name = "mine"

    def _calibrate(self, frame): ...            # gather statistics
    def _finish_calibration(self) -> CalibrationResult: ...  # thresholds + baseline
    def _detect(self, frame) -> Dict[str, float]: ...        # anomaly id -> intensity
```

Key points:

- Detectors start in **calibration mode**; the pipeline feeds frames, then
  calls `finish_calibration()`, which returns a `CalibrationResult`
  (thresholds + JSON-serialisable annotations) and switches the detector
  to **detection mode**. Mode handling lives in the base class.
- `process(frame)` returns a mapping of anomaly IDs (e.g.
  `hud/cam01/text`) to intensity values. The *pipeline* queries
  `thresholds()` and decides what constitutes a detection - detectors
  never aggregate events themselves.
- Optional hooks: `metadata(anomaly_id)` (static context stored with the
  event) and `regions(anomaly_id)` (image regions to highlight in the
  evidence GIF).
- Register the detector in the config; no core changes needed:

```yaml
detectors:
  - class: mypackage.MyDetector
    config: {some_option: 3}
```

## Video source interface (`amon/sources/__init__.py`)

Implement `fps` and `frames()` (a generator of `Frame` objects), raise
`SourceError` when the stream cannot be opened. Bundled implementations:

- `VideoFileSource` — pre-recorded files (`amon/sources/file.py`). Optional
  ``processing_fps`` skips frames so the pipeline never sees more than that
  many frames per second of video time; ``realtime`` paces wall-clock delivery.
- `VideoInputStream` — live capture devices such as USB adapters or
  capture cards (`amon/sources/stream.py`).  Prefer numeric ``device``
  indices (portable across Linux/macOS/Windows); OpenCV selects the native
  backend via ``CAP_ANY``.  Native resolution and FPS are auto-detected;
  ``processing_fps`` optionally throttles output.  When the configured
  device is unavailable, open alternatives are listed in the error before
  the pipeline aborts.

Reference a plugin from the config by dotted class path; no core changes needed.

## Preprocessing (`amon/preprocess.py`)

After each frame leaves the video source, the pipeline runs
``FramePreprocessor`` (config key ``preprocessing``):

- ``scale`` — percent of input size, aspect preserved
- ``rotate`` — degrees clockwise (90/180/270 use a fast OpenCV path)
- ``brightness`` / ``contrast`` — ``out = contrast * image + brightness``

Order is rotate → scale → brightness/contrast.  Identity defaults disable
the stage.  Applies to every source type.

## Image-processing algorithms

Intuition-first explanations for developers new to computer vision live in
the [algorithm knowledge base](docs/README.md). The sections below focus on
how those techniques are wired into this codebase.

### Threshold calibration (`amon/stats.py`)

All detectors collect intensity samples on clean calibration footage and
derive thresholds as `max(floor, median + k·MAD·1.4826, 1.5·max(samples))`.
The MAD-based sigma is robust against occasional glitches; the
`1.5·max` term guarantees clearance above everything seen during
calibration; the floor encodes the metric's physical scale. Config key
`tolerance` (default `1.0`, or a per-anomaly mapping keyed by trailing
segment / full ID / `default`) then multiplies calibrated thresholds —
`noise: 1.2` requires that intensity to exceed the learned cutoff by
20%. No manual per-metric absolute tuning is required.

### Temporal detector (`detectors/temporal.py`)

- **Noise**: robust std (`1.4826·MAD`) of the frame difference after
  removing its spatial mean. Global brightness changes cancel out;
  small blinking HUD regions barely move the median - only dense,
  per-pixel change (sensor noise) scores high.
- **Flicker**: mean absolute change of global brightness over a short
  sliding window - uniform oscillation moves the global mean strongly.
- **Contrast**: relative deviation of the intensity std from the
  calibrated baseline.

### HUD detector (`detectors/hud.py`)

HUD overlays are assumed bright. Calibration collects per-frame bright
masks (`gray > bright_threshold`) and the temporal max image; the union
mask is dilated and split into connected components = HUD elements. Per
element it learns: bounding box, centroid, pixel count, text (via the
offline glyph matcher in `textocr.py`, Otsu-binarised, matched against
TrueType templates (default: bundled VCR OSD Mono under `amon/fonts/`;
override with `glyph_font`); components outside
`min_glyph_height`…`max_glyph_height` / `max_glyph_width` or below
`min_glyph_area` are ignored so tiny IR speckles and large non-text
blobs are not read as letters) and the blink toggle
rate. Detection re-locates each element inside a search window around its
calibrated box and emits four intensities: normalised Levenshtein text
distance, centroid shift (px), relative box-area change, and toggle-rate
deviation over a sliding window (covers frequency change and blink
start/stop with a single metric). Unexpected overlays each get a channel
``hud/<slug>/new`` (one event per appearing element) when readable text
appears outside every calibrated element's search window.

### Spatial detector (`detectors/spatial.py`)

Calibration builds the temporal median of sampled frames (baseline image)
and detects Shi-Tomasi corners on it, masking out dilated ever-bright
(HUD) pixels so HUD overlays are ignored. Detection tracks all corners
from the baseline into the current frame with pyramidal Lucas-Kanade flow
plus a forward-backward consistency check; the intensity is the
third-largest displacement, which tolerates single-point outliers but
responds to any locally coherent distortion. Regions around displaced
points are attached as highlights.

## Event aggregation and the suppression hierarchy

`EventAggregator.update(t, readings)` implements a per-anomaly state
machine: an event opens at threshold crossing, stays open while intensity
re-crosses within `cooldown_seconds`, and is dropped if shorter than
`min_duration_seconds`. Continuous anomalies therefore produce exactly
one event. Open events are also written to the database as `status=ongoing`
(refreshed about once per second) so a live report refresh can show
anomalies that have not closed yet; closing promotes the same row to
`completed` and attaches the evidence GIF.

Per-run diagnostic logs go to `logging.dir` / `<session_id>.log` (default
`logs/`). Console and file levels are set separately (`console_level`,
`file_level`; defaults INFO / DEBUG). Use `file_level: DEBUG` (or
`--log-level DEBUG`) to record aggregation decisions (OPEN / CLOSE /
DISCARD / SUPPRESSED) — useful on field machines without a debugger.

The `suppresses` config maps suppressor patterns to target patterns.
``*`` matches any path span with no capture binding; ``#`` matches one
segment and binds across suppressor/target when both sides use the same
``#`` count (`hud/#/size` only suppresses `hud/#/text` of that element).
While a suppressor is raw-active - and for `suppression_linger_seconds`
after a *sustained* suppressor subsides, giving windowed metrics time to
drain - matching targets cannot open events.

## Persistence

SQLite file `amon.sqlite` in `data_dir`; schema in `db.py` (sessions,
calibrations, events). Event timestamps are seconds relative to session
start; the session row holds the wall-clock epoch. Media files are stored
under `media/<session>/` and referenced by path.

## Background processing

`BackgroundWorker` owns a `multiprocessing.Queue` and a worker process.
The pipeline enqueues finalised events together with the already-captured
evidence frames (ring buffer + live capture, clipped to
`media.max_clip_seconds`); the worker encodes GIFs and writes the
database. A media failure never loses the event record.

## Export architecture

`exporters.Exporter` is the interface (`format`, `suffix`, `export()`);
implementations register in the `EXPORTERS` dict and become available to
the CLI (`--format`) and the UI automatically. `HtmlArchiveExporter`
builds a static Panel layout and saves it with inlined Bokeh resources and
base64-embedded media - a single self-contained file.

## Testing

- `amon/synthetic.py` generates a deterministic video with a known
  anomaly schedule (`SCHEDULE`, `EXPECTED_EVENTS`), covering every
  anomaly type plus an overlapping flicker+noise scenario.
- `tests/test_detectors.py` - per-detector unit tests (calibration,
  threshold behaviour, edge cases) on codec-free synthetic frames.
- `tests/test_e2e.py` - full pipeline runs against the encoded video and
  verifies events, timing, suppression, database contents and media.
- Run everything with `python -m pytest`.

## Portable offline distribution

```bash
python3 scripts/bundle_portable.py
```

Build OS must equal target OS. The script automatically:

1. Fetches a relocatable CPython into `dist/cache/` (skipped if already cached).
   This is **not** the host Python: system installs use fixed paths and cannot
   be copied to USB; PBS builds are self-contained (~30–50 MB compressed).
2. Compiles `requirements-runtime.in` → `dist/cache/requirements-runtime.txt`
   when the input is newer than the lockfile.
3. Downloads wheels into `dist/cache/wheels/` only when the lockfile digest
   changes (no double-download on re-runs).
4. Assembles `dist/amon-portable-<platform>/` with `amon` / `amon.bat`
   launchers (`python -m amon`, no venv activate).

If CPython cannot be downloaded (offline build machine, firewall, 403), place
the matching `cpython-*-<triple>-install_only.tar.gz` from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases)
into `dist/cache/` manually; the script prints the exact pattern on failure.

## Technical decisions

- **OpenCV + classical CV** instead of learned models: deterministic,
  fast, fully offline, no training data needed.
- **SQLite** as the local file database: zero-infrastructure, atomic,
  queryable; media stays on the filesystem to keep the DB small.
- **Multiprocessing (not threads)** for I/O: GIF encoding is CPU-bound,
  so a separate process protects the real-time loop from the GIL.
- **Panel** for the UI (project constraint) - declarative, no handwritten
  HTML, Bootstrap theme with explicit offline resource configuration.
- **pip-tools** for reproducible dependency pinning.
- **Portable CPython + wheelhouse** for USB/offline distribution instead of
  a frozen single-file binary (friendlier to OpenCV/Panel and field debug).
