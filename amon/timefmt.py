"""Wall-clock timestamp formatting for reports."""
from __future__ import annotations

from datetime import datetime
from typing import Optional


def format_wall_time(epoch: Optional[float]) -> str:
    """Format a Unix epoch timestamp for display, or em dash when missing."""
    if epoch is None:
        return "—"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
