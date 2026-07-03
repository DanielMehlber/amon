"""amon - a lightweight anomaly-monitoring framework for static video scenes.

The package is organised as a small framework:

- :mod:`amon.sources` - pluggable video sources (``VideoSource`` interface).
- :mod:`amon.detectors` - pluggable anomaly detectors (``Detector`` interface).
- :mod:`amon.pipeline` - the real-time monitoring pipeline (calibration,
  detection, event aggregation and hand-off to background workers).
- :mod:`amon.report` - Panel based reporting UI and export.

See ``for-users.md`` and ``for-dev.md`` in the repository root.
"""

__version__ = "1.0.0"
