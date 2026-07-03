"""Frame-level detections are merged into anomaly events here.

An event opens when an anomaly's intensity reaches its calibrated
threshold and closes once the intensity stayed below the threshold for a
configurable cooldown, so a continuous anomaly yields exactly one event.

A configurable *exclusion hierarchy* suppresses side-effects of a primary
anomaly (e.g. a flickering screen would otherwise also trigger HUD and
spatial detections).  Rules map a suppressor pattern to target patterns
where ``*`` greedily matches one or more path segments.  When suppressor
and target patterns contain the same (non-zero) number of wildcards the
captured values carry over, so ``"hud/*/size": ["hud/*/position"]`` only
suppresses the position anomaly of the *same* HUD element.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from amon.model import AnomalyEvent


def _compile(pattern: str) -> re.Pattern:
    parts = [re.escape(p) for p in pattern.split("*")]
    return re.compile("^" + "(.+)".join(parts) + "$")


def _substitute(pattern: str, captures: Tuple[str, ...]) -> str:
    for capture in captures:
        pattern = pattern.replace("*", capture, 1)
    return pattern


class SuppressionRules:
    """Evaluates the exclusion hierarchy from the aggregation config."""

    def __init__(self, rules: Dict[str, List[str]]):
        self._rules = [(_compile(sup), sup.count("*"), targets) for sup, targets in (rules or {}).items()]

    def suppressed(self, anomaly_id: str, active: Iterable[str]) -> bool:
        """True if ``anomaly_id`` is suppressed by any *other* active anomaly."""
        for suppressor in active:
            if suppressor == anomaly_id:
                continue
            for sup_re, sup_stars, targets in self._rules:
                match = sup_re.match(suppressor)
                if not match:
                    continue
                captures = match.groups()
                for target in targets:
                    if sup_stars and target.count("*") == sup_stars:
                        if _substitute(target, captures) == anomaly_id:
                            return True
                    elif _compile(target).match(anomaly_id):
                        return True
        return False


@dataclass
class Reading:
    """One anomaly measurement for the current frame."""

    intensity: float
    threshold: float
    detector: str


@dataclass
class _OpenEvent:
    event: AnomalyEvent
    last_above: float


class EventAggregator:
    """Stateful aggregation of per-frame readings into :class:`AnomalyEvent`.

    Config keys (see ``aggregation`` section of the config file):
    ``cooldown_seconds``, ``min_duration_seconds``, ``max_timeline_points``
    and ``suppresses``.
    """

    def __init__(self, config: dict):
        self.cooldown = float(config.get("cooldown_seconds", 1.0))
        self.min_duration = float(config.get("min_duration_seconds", 0.5))
        self.max_points = int(config.get("max_timeline_points", 2000))
        self.linger = float(config.get("suppression_linger_seconds", 2.5))
        self.rules = SuppressionRules(config.get("suppresses", {}))
        self._open: Dict[str, _OpenEvent] = {}
        self._streaks: Dict[str, Tuple[float, float]] = {}  # aid -> (streak start, last raw)

    def update(self, t: float, readings: Dict[str, Reading]) -> Tuple[List[str], List[AnomalyEvent], List[str]]:
        """Process one frame's readings.

        Returns ``(opened ids, closed events, discarded ids)`` where
        discarded ids belong to events dropped for being shorter than
        ``min_duration_seconds``.
        """
        raw = {aid for aid, r in readings.items() if r.intensity >= r.threshold}
        for aid in raw:
            start, last = self._streaks.get(aid, (t, t))
            self._streaks[aid] = (t if t - last > self.cooldown else start, t)
        # Suppression is instantaneous for currently raw anomalies.  On top,
        # *sustained* suppressors keep their grip for a short linger after
        # subsiding: windowed metrics of suppressed detectors (e.g. HUD blink
        # rates) need time to drain the primary anomaly's side-effects.
        # Momentary spikes get no linger so they cannot mask real follow-ups.
        suppressors = set(raw)
        for aid, (start, last) in self._streaks.items():
            if last - start >= self.min_duration and t - last <= self.linger:
                suppressors.add(aid)
        firing = {aid for aid in raw if not self.rules.suppressed(aid, suppressors)}

        opened: List[str] = []
        closed: List[AnomalyEvent] = []
        discarded: List[str] = []
        for aid, reading in readings.items():
            state = self._open.get(aid)
            if aid in firing:
                if state is None:
                    state = _OpenEvent(
                        event=AnomalyEvent(
                            anomaly_id=aid,
                            detector=reading.detector,
                            start=t,
                            end=t,
                            max_intensity=reading.intensity,
                            threshold=reading.threshold,
                        ),
                        last_above=t,
                    )
                    self._open[aid] = state
                    opened.append(aid)
                state.last_above = t
                state.event.max_intensity = max(state.event.max_intensity, reading.intensity)
                self._append_point(state, t, reading.intensity)
            elif state is not None:
                self._append_point(state, t, reading.intensity)
                if t - state.last_above >= self.cooldown:
                    event = self._close(aid)
                    closed.append(event) if event else discarded.append(aid)
        return opened, closed, discarded

    def flush(self) -> List[AnomalyEvent]:
        """Close all events that are still open (called at stream end)."""
        closed = [self._close(aid) for aid in list(self._open)]
        return [event for event in closed if event]

    def _close(self, aid: str) -> Optional[AnomalyEvent]:
        state = self._open.pop(aid)
        event = state.event
        event.end = state.last_above
        # Trim the cooldown tail from the timeline and drop glitches.
        event.timeline = [(t, v) for t, v in event.timeline if t <= event.end]
        return event if event.duration >= self.min_duration else None

    def _append_point(self, state: _OpenEvent, t: float, intensity: float) -> None:
        timeline = state.event.timeline
        timeline.append((t, intensity))
        if len(timeline) > self.max_points:
            # Halve resolution but keep the most recent point.
            state.event.timeline = timeline[::2] + timeline[-1:]
