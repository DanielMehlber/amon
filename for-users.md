# amon - User Guide

`amon` monitors a video stream of a static scene, automatically calibrates
itself, detects anomalies (noise, flicker, contrast changes, HUD changes,
spatial distortions) and records each occurrence as a single event with
timestamps, intensity measurements and a GIF as visual evidence.  Results
are reviewed in the browser and can be exported as a standalone HTML
archive.  Everything runs fully offline.

## Installation

Requires Python 3.9+. Dependencies are managed with pip-tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-sync requirements.txt        # installs the pinned dependency set
```

(To change dependencies, edit `requirements.in` and run `pip-compile`.)

## Quick start

Generate the synthetic demo video and monitor it:

```bash
python -m amon synth test-video.avi
python -m amon monitor config.yaml
python -m amon report config.yaml     # opens the browser UI
```

## Configuration

All behaviour is controlled by a single YAML file; see `config.yaml` for a
fully commented example. The most important keys:

| Key | Meaning |
|---|---|
| `video_source.config.path` | The video file to monitor. |
| `video_source.config.realtime` | `true` paces playback like a live stream. |
| `calibration.duration_seconds` | Length of the automatic calibration phase. |
| `detectors` | Which detector plugins to load and their settings. |
| `aggregation.suppresses` | The exclusion hierarchy that prevents false positives. |
| `media.max_clip_seconds` | Evidence clips of long events are cut to this length. |
| `data_dir` | Where the database, media and exports are stored. |

Detector thresholds are **not** configured - they are learned automatically
during calibration.

## Running a monitoring session

```bash
python -m amon monitor config.yaml
```

The session starts with the calibration phase (the video should show
*normal* behaviour during this time), then switches to monitoring
automatically. Monitoring runs until the video ends or you press
`Ctrl-C`; sessions may run for days. Detected events are written to
`<data_dir>/amon.sqlite` and their GIFs to `<data_dir>/media/<session>/`
in the background while monitoring continues in real time.

## Viewing reports

```bash
python -m amon report config.yaml
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

Either click **Export report** in the UI's Export tab, or run:

```bash
python -m amon export config.yaml --session <session-id>
```

This writes a standalone HTML archive (media embedded, works offline) to
`<data_dir>/exports/`. Use `--output` to choose a different location. The
export format defaults to `export.format` in the config; additional
formats can be added by developers (see `for-dev.md`).
