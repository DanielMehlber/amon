# amon - video anomaly monitoring framework

Monitors a video stream of a static scene, calibrates itself
automatically, detects temporal / HUD / spatial anomalies and records
each occurrence as a single event with timestamps, intensity data and GIF
evidence. Includes a browser-based report UI (Panel) and standalone HTML
export. Runs fully offline.

- **User guide:** [for-users.md](for-users.md) - installation,
  configuration, running, reports, export.
- **Developer guide:** [for-dev.md](for-dev.md) - architecture,
  algorithms, plugin interfaces, design rationale.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pip-tools && pip-sync requirements.txt

python -m amon synth test-video.avi      # synthetic demo video
python -m amon monitor config.yaml       # run a monitoring session
python -m amon report config.yaml        # inspect results in the browser
```

## Tests

```bash
python -m pytest
```
