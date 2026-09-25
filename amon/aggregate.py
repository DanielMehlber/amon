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

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from amon.model import AnomalyEvent

log = logging.getLogger("amon.aggregate")


def _compile_regex(pattern: str) -> re.Pattern:
    parts = [re.escape(p) for p in pattern.split("*")]
    return re.compile("^" + "(.+)".join(parts) + "$")


def _substitute(pattern: str, captures: Tuple[str, ...]) -> str:
    for capture in captures:
        pattern = pattern.replace("*", capture, 1)
    return pattern


class SuppressionRules:
    """Evaluates the exclusion hierarchy from the aggregation config."""

    def __init__(self, rules: Dict[str, List[str]]):
        self._rules = [
            (_compile_regex(sup), sup.count("*"), targets)
            for sup, targets in (rules or {}).items()
        ]

    def suppressed(self, anomaly_id: str, active: Iterable[str]) -> bool:
        """True if ``anomaly_id`` is suppressed by any *other* active anomaly."""
        return self.suppressor_of(anomaly_id, active) is not None

    def suppressor_of(
        self, anomaly_id: str, active: Iterable[str]
    ) -> Optional[str]:
        """Return the active anomaly ID that suppresses ``anomaly_id``, if any."""
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
                            return suppressor
                    elif _compile_regex(target).match(anomaly_id):
                        return suppressor
        return None


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
        self._streaks: Dict[str, Tuple[float, float]] = (
            {}
        )  # aid -> (streak start, last raw)
        # aid -> suppressor id for the current contiguous suppression streak
        self._suppress_streak: Dict[str, str] = {}

    def update(
        self, t: float, readings: Dict[str, Reading]
    ) -> Tuple[List[str], List[AnomalyEvent], List[str]]:
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

        firing: set = set()
        currently_suppressed: set = set()
        for aid in raw:
            blocker = self.rules.suppressor_of(aid, suppressors)
            reading = readings[aid]
            if blocker is not None:
                currently_suppressed.add(aid)
                detail = (
                    "t=%.3fs SUPPRESSED %s: intensity=%.4f >= threshold=%.4f "
                    "(detector=%s) blocked by suppressor %s "
                    "(raw_above_threshold=%s linger_eligible=%s)"
                    % (
                        t,
                        aid,
                        reading.intensity,
                        reading.threshold,
                        reading.detector,
                        blocker,
                        sorted(raw),
                        sorted(suppressors - raw),
                    )
                )
                if self._suppress_streak.get(aid) != blocker:
                    log.debug(
                        "%s — event will NOT open/continue while this suppressor is active",
                        detail,
                    )
                    self._suppress_streak[aid] = blocker
                else:
                    log.debug("%s", detail)
            else:
                firing.add(aid)

        for aid in list(self._suppress_streak):
            if aid not in currently_suppressed:
                log.debug(
                    "t=%.3fs SUPPRESSION_END %s: no longer blocked (was suppressed by %s)",
                    t,
                    aid,
                    self._suppress_streak.pop(aid),
                )

        for aid, reading in readings.items():
            if aid in raw:
                continue
            margin = reading.threshold - reading.intensity
            log.debug(
                "t=%.3fs BELOW_THRESHOLD %s: intensity=%.4f < threshold=%.4f "
                "(shortfall=%.4f, detector=%s, event_open=%s)",
                t,
                aid,
                reading.intensity,
                reading.threshold,
                margin,
                reading.detector,
                aid in self._open,
            )

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
                    log.debug(
                        "t=%.3fs OPEN %s: intensity=%.4f >= threshold=%.4f "
                        "(detector=%s) — event started; will stay open until "
                        "intensity stays below threshold for cooldown=%.2fs",
                        t,
                        aid,
                        reading.intensity,
                        reading.threshold,
                        reading.detector,
                        self.cooldown,
                    )
                else:
                    log.debug(
                        "t=%.3fs CONTINUE %s: intensity=%.4f >= threshold=%.4f "
                        "(open since %.3fs, peak_so_far=%.4f)",
                        t,
                        aid,
                        reading.intensity,
                        reading.threshold,
                        state.event.start,
                        state.event.max_intensity,
                    )
                state.last_above = t
                state.event.max_intensity = max(
                    state.event.max_intensity, reading.intensity
                )
                self._append_point(state, t, reading.intensity)
            elif state is not None:
                self._append_point(state, t, reading.intensity)
                below_for = t - state.last_above
                if below_for >= self.cooldown:
                    duration_so_far = state.last_above - state.event.start
                    event = self._close(aid)
                    if event:
                        closed.append(event)
                        log.debug(
                            "t=%.3fs CLOSE %s: was above threshold from %.3fs to "
                            "%.3fs (duration=%.3fs, peak=%.4f, threshold=%.4f); "
                            "closed after %.3fs below threshold (cooldown=%.2fs)",
                            t,
                            aid,
                            event.start,
                            event.end,
                            event.duration,
                            event.max_intensity,
                            event.threshold,
                            below_for,
                            self.cooldown,
                        )
                    else:
                        discarded.append(aid)
                        log.debug(
                            "t=%.3fs DISCARD %s: was above threshold from %.3fs to "
                            "%.3fs (duration=%.3fs) but shorter than "
                            "min_duration_seconds=%.2fs — treated as glitch, "
                            "not written as an anomaly event",
                            t,
                            aid,
                            state.event.start,
                            state.last_above,
                            duration_so_far,
                            self.min_duration,
                        )
                else:
                    log.debug(
                        "t=%.3fs COOLDOWN %s: intensity=%.4f < threshold=%.4f "
                        "but only %.3fs below so far (need cooldown=%.2fs); "
                        "event still open since %.3fs",
                        t,
                        aid,
                        reading.intensity,
                        reading.threshold,
                        below_for,
                        self.cooldown,
                        state.event.start,
                    )
        return opened, closed, discarded

    def peek_open(self) -> Dict[str, AnomalyEvent]:
        """Snapshots of still-open events with ``end`` set to last-above time."""
        snapshots: Dict[str, AnomalyEvent] = {}
        for aid, state in self._open.items():
            event = state.event
            snapshots[aid] = AnomalyEvent(
                anomaly_id=event.anomaly_id,
                detector=event.detector,
                start=event.start,
                end=state.last_above,
                max_intensity=event.max_intensity,
                threshold=event.threshold,
                timeline=list(event.timeline),
                metadata=dict(event.metadata),
                regions=list(event.regions),
            )
        return snapshots

    def flush(self) -> List[AnomalyEvent]:
        """Close all events that are still open (called at stream end)."""
        still_open = list(self._open)
        if still_open:
            log.debug(
                "flush: closing %d still-open event(s) at stream end: %s",
                len(still_open),
                still_open,
            )
        closed = []
        for aid in still_open:
            event = self._close(aid)
            if event:
                closed.append(event)
                log.debug(
                    "flush CLOSE %s: %.3fs-%.3fs (duration=%.3fs, peak=%.4f)",
                    aid,
                    event.start,
                    event.end,
                    event.duration,
                    event.max_intensity,
                )
            else:
                log.debug(
                    "flush DISCARD %s: open at stream end but duration shorter "
                    "than min_duration_seconds=%.2fs",
                    aid,
                    self.min_duration,
                )
        return closed

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
