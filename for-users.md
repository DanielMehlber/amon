# amon - User Guide

`amon` monitors a video stream of a static scene, automatically calibrates
itself, detects anomalies (noise, flicker, contrast changes, HUD changes,
spatial distortions) and records each occurrence as a single event with
timestamps, intensity measurements and a GIF as visual evidence.  Results
are reviewed in the browser and can be exported as a standalone HTML
archive.  Everything runs fully offline.

## Installation

### Developer / connected machine

Requires Python 3.9+. Dependencies are managed with pip-tools. Lockfiles are
**not** committed — compile them on your machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile --strip-extras -o requirements.txt requirements.in
pip-sync requirements.txt
```

(Runtime-only inputs for portable builds live in `requirements-runtime.in`;
`scripts/bundle_portable.py` compiles and caches the lockfile automatically.)

### Offline / USB portable bundle

On a machine with network (same OS/arch as the offline target):

```bash
python scripts/bundle_portable.py          # Windows: python scripts\bundle_portable.py
```

This downloads CPython and runtime wheels into `dist/cache/` on the first
run (later runs reuse the cache) and writes `dist/amon-portable-<platform>/`.

Copy that folder to a USB stick. On the offline PC:

```bat
amon.bat synth test-video.avi
amon.bat monitor test.yaml
```

## Quick start

Generate the synthetic demo video and monitor it:

```bash
python -m amon synth test-video.avi
python -m amon monitor test.yaml
python -m amon report test.yaml     # opens the browser UI
```

For live capture-card / USB adapter input use `stream.yaml` instead
(adjust `device`):

```bash
python -m amon monitor stream.yaml
python -m amon report stream.yaml
```

## Configuration

Two ready-made configs ship with the project:

| File | Purpose |
|---|---|
| `test.yaml` | File source — synthetic / recorded `test-video.avi` |
| `stream.yaml` | Live video input (USB adapter, capture card, webcam) |

All behaviour is controlled by the YAML file you pass to the CLI. The most
important keys:

| Key | Meaning |
|---|---|
| `video_source.config.path` | The video file to monitor (file source). |
| `video_source.config.device` | Capture device index or path (stream source). |
| `video_source.config.processing_fps` | Max FPS delivered to the pipeline (stream and file). |
| `video_source.config.realtime` | `true` paces file playback like a live stream. |
| `preprocessing.scale` | Scale percent of frame size (aspect preserved). |
| `preprocessing.rotate` | Clockwise rotation in degrees. |
| `preprocessing.brightness` | Additive brightness offset `[-255, 255]`. |
| `preprocessing.contrast` | Contrast gain (`1.0` = unchanged). |
| `calibration.duration_seconds` | Length of the automatic calibration phase. |
| `detectors` | Which detector plugins to load and their settings. |
| `aggregation.suppresses` | The exclusion hierarchy that prevents false positives. |
| `media.max_clip_seconds` | Evidence clips of long events are cut to this length. |
| `data_dir` | Where the database, media and exports are stored. |
| `logging.console_level` | Terminal verbosity (`INFO` = anomalies only). |
| `logging.file_level` | Per-run file verbosity (`DEBUG` = full detection trace). |
| `logging.dir` | Folder for per-run log files (`logs/<session_id>.log`). |
| `logging.console` | Also print logs to the terminal (`true`/`false`). |

Detector thresholds are learned automatically during calibration. Each
detector accepts an optional ``tolerance`` — a float for all of its
anomalies, or a mapping per anomaly type (trailing ID segment such as
``noise`` / ``text``, full ID, or ``default``). Example: ``1.2`` requires
intensity to exceed the calibrated cutoff by 20%:

```yaml
detectors:
  - class: amon.detectors.temporal.TemporalDetector
    config:
      tolerance:
        noise: 1.2
        flicker: 1.0
        contrast: 1.0
```

### Diagnostic logging

Each monitoring run writes a log file named after the session, e.g.
`logs/Oceanic-Robin.log`. Console and file levels are independent:

```yaml
logging:
  console: true
  console_level: INFO    # OPEN / CLOSE / FINALIZE on the terminal
  file_level: DEBUG      # intensity-vs-threshold + suppression in the file
  dir: logs
```

Overrides: `python -m amon monitor test.yaml --log-level DEBUG` sets the
**file** level (use `--console-log-level` for the terminal).

At console `INFO` you only see session start/finish, calibration complete, and
each detected anomaly (`event <id>: start-end (duration, peak)`). File `DEBUG`
traces aggregation: OPEN / CLOSE / DISCARD / SUPPRESSED (and why a crossing
did not become a reported anomaly). Quiet frames are not logged.

### Preprocessing

Frames from any video source pass through an optional preprocessing stage
before calibration/detection:

```yaml
preprocessing:
  scale: 50        # percent of input size (aspect ratio preserved)
  rotate: 90       # degrees clockwise
  brightness: 10   # additive [-255, 255]
  contrast: 1.2    # multiplicative gain
```

Order: rotate → scale → brightness/contrast.  Defaults leave the image
unchanged.

### Live video input

Use `stream.yaml` and set `device` to a portable index (`0`, `1`, …) or,
on Linux, a path such as `/dev/video0`.  Native resolution and frame rate
are detected automatically; optional `processing_fps` throttles capture.
Image transforms (`scale`, `rotate`, `brightness`, `contrast`) are set
under `preprocessing`.  If the device cannot be opened, available
alternatives are listed in the error message.

## Running a monitoring session

```bash
python -m amon monitor test.yaml      # file demo
python -m amon monitor stream.yaml    # live capture
```

The session starts with the calibration phase (the video should show
*normal* behaviour during this time), then switches to monitoring
automatically. Monitoring runs until the video ends or you press
`Ctrl-C`; sessions may run for days. Detected events are written to
`<data_dir>/amon.sqlite` and their GIFs to `<data_dir>/media/<session>/`
in the background while monitoring continues in real time.

## Viewing reports

```bash
python -m amon report test.yaml     # or stream.yaml
```

opens the report UI in your browser (port from `report.port`, default
5006).  The UI serves all JavaScript, CSS and fonts from the local Panel
process (`report.offline: true` by default) so it works without internet
access.  Set `report.address` to `0.0.0.0` when opening the UI from another
machine on an isolated network (WebSocket origins are configured
automatically).  You can:

- pick any completed or still-running session,
- browse events chronologically and filter by anomaly type or duration,
- open an event to see its GIF, intensity plot with the calibrated
  threshold line, and detector metadata,
- review the calibration: an annotated GIF marks tracked feature points,
  HUD elements, their recognised text and blink frequencies.

## Exporting reports

Either click **Export events as CSV** on the Anomalies tab, choose **csv** or
**html** in the Export tab, or run:

```bash
python -m amon export test.yaml --session <session-id> --format csv
python -m amon export test.yaml --session <session-id> --format html
```

HTML archives embed media and work fully offline.  CSV files contain one
row per anomaly with session-relative and wall-clock timestamps, suitable
for spreadsheets and statistical analysis.  Exports are written to
`<data_dir>/exports/`. Use `--output` to choose a different location. The
export format defaults to `export.format` in the config; additional
formats can be added by developers (see `for-dev.md`).
