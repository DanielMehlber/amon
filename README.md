# AMON - video anomaly monitoring framework

Monitors a video stream of a static scene, calibrates itself
automatically, detects temporal / HUD / spatial anomalies and records
each occurrence as a single event with timestamps, intensity data and GIF
evidence. Includes a browser-based report UI (Panel) and standalone HTML
export. Runs fully offline.

- **User guide:** [for-users.md](for-users.md) - installation,
  configuration, running, reports, export.
- **Developer guide:** [for-dev.md](for-dev.md) - architecture,
  algorithms, plugin interfaces, design rationale.
- **Algorithm knowledge base:** [docs/](docs/) - intuition-first
  explanations of the CV techniques used.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pip-tools
pip-compile --strip-extras -o requirements.txt requirements.in
pip-sync requirements.txt

python -m amon synth test-video.avi      # synthetic demo video
python -m amon monitor test.yaml         # file-based test session
python -m amon report test.yaml          # inspect results in the browser
# Live capture: python -m amon monitor stream.yaml
```

### Offline USB bundle

```bash
python scripts/bundle_portable.py    # builds dist/amon-portable-<platform>/
```

On the offline target:

```bat
amon.bat monitor test.yaml
# or: amon.bat monitor stream.yaml
```

## Tests

```bash
python -m pytest
```
