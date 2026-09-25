"""Session diagnostic logging: console + per-run log file.

Console and file levels are independent. Typical field setup:

```yaml
logging:
  console: true
  console_level: INFO    # anomalies OPEN/CLOSE/FINALIZE on stderr
  file_level: DEBUG      # full intensity/suppression trace in the log file
  dir: logs              # per-run files: <dir>/<session_id>.log
```

Legacy ``logging.level`` still works as a fallback for both sinks when the
specific ``*_level`` keys are omitted.

Once a monitoring session ID is known, call :func:`attach_session_log` so
messages are written to ``<dir>/<session_id>.log``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_SESSION_HANDLER_ATTR = "_amon_session_file"
_CONSOLE_HANDLER_ATTR = "_amon_console"


def _level(name: str, default: str = "INFO") -> int:
    return getattr(logging, str(name or default).upper(), logging.INFO)


def _sink_levels(cfg: dict) -> Tuple[int, int]:
    """Return ``(console_level, file_level)`` integers."""
    legacy = cfg.get("level")
    console = _level(cfg.get("console_level", legacy or "INFO"))
    file_level = _level(cfg.get("file_level", legacy or "DEBUG"))
    return console, file_level


def configure_logging(config: Optional[dict] = None) -> None:
    """Apply console logging from config (call once at process start)."""
    cfg = (config or {}).get("logging") or {}
    console_level, file_level = _sink_levels(cfg)
    console = bool(cfg.get("console", True))

    amon = logging.getLogger("amon")
    # Logger must admit the most verbose sink; handlers filter further.
    amon.setLevel(min(console_level, file_level))
    amon.propagate = False  # keep amon diagnostics off the root handlers

    # Drop previous console handlers we installed (re-entrant / tests).
    for handler in list(amon.handlers):
        if getattr(handler, _CONSOLE_HANDLER_ATTR, False):
            amon.removeHandler(handler)
            handler.close()

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(console_level)
        stream.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        setattr(stream, _CONSOLE_HANDLER_ATTR, True)
        amon.addHandler(stream)


def attach_session_log(config: Optional[dict], session_id: str) -> Path:
    """Attach a file handler named after the run; returns the log path."""
    cfg = (config or {}).get("logging") or {}
    console_level, file_level = _sink_levels(cfg)
    log_dir = Path(cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{session_id}.log"

    amon = logging.getLogger("amon")
    amon.setLevel(min(console_level, file_level))

    for handler in list(amon.handlers):
        if getattr(handler, _SESSION_HANDLER_ATTR, False):
            amon.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    setattr(file_handler, _SESSION_HANDLER_ATTR, True)
    amon.addHandler(file_handler)

    amon.info("diagnostic log: %s", path.resolve())
    amon.debug(
        "diagnostic log levels: console=%s file=%s "
        "(file DEBUG traces aggregation: OPEN/CLOSE/DISCARD/SUPPRESSED)",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
    )
    return path


def detach_session_log() -> None:
    """Remove the per-run file handler (optional; for tests)."""
    amon = logging.getLogger("amon")
    for handler in list(amon.handlers):
        if getattr(handler, _SESSION_HANDLER_ATTR, False):
            amon.removeHandler(handler)
            handler.close()
