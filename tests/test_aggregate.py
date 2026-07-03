"""Unit tests for event aggregation and the suppression hierarchy."""
import pytest

from amon.aggregate import EventAggregator, Reading, SuppressionRules

RULES = SuppressionRules({
    "temporal/flicker": ["*"],
    "temporal/noise": ["temporal/contrast", "hud/*"],
    "hud/*/size": ["hud/*/text", "hud/*/position"],
})


class TestSuppressionRules:
    def test_wildcard_suppresses_everything_else(self):
        active = {"temporal/flicker", "temporal/noise", "hud/cam01/text"}
        assert RULES.suppressed("temporal/noise", active)
        assert RULES.suppressed("hud/cam01/text", active)
        assert not RULES.suppressed("temporal/flicker", active)

    def test_prefix_wildcard(self):
        active = {"temporal/noise", "hud/cam01/blink"}
        assert RULES.suppressed("hud/cam01/blink", active)

    def test_capture_must_match_same_element(self):
        active = {"hud/cam01/size", "hud/rec/text"}
        assert not RULES.suppressed("hud/rec/text", active)
        assert RULES.suppressed("hud/cam01/text", {"hud/cam01/size", "hud/cam01/text"})

    def test_nothing_suppressed_without_suppressor(self):
        assert not RULES.suppressed("hud/cam01/text", {"hud/cam01/text"})


def make_aggregator(**overrides):
    config = {"cooldown_seconds": 0.5, "min_duration_seconds": 0.4, "suppresses": {}}
    config.update(overrides)
    return EventAggregator(config)


def feed(agg, timestamps, intensity_fn, threshold=1.0, aid="a"):
    """Feed a reading per timestamp; collect opened/closed/discarded."""
    opened, closed, discarded = [], [], []
    for t in timestamps:
        o, c, d = agg.update(t, {aid: Reading(intensity_fn(t), threshold, "det")})
        opened += o
        closed += c
        discarded += d
    return opened, closed, discarded


class TestEventAggregator:
    def test_continuous_anomaly_yields_exactly_one_event(self):
        agg = make_aggregator()
        times = [i * 0.1 for i in range(100)]  # 0 .. 9.9s
        opened, closed, _ = feed(agg, times, lambda t: 5.0 if 2.0 <= t <= 5.0 else 0.0)
        assert opened == ["a"]
        assert len(closed) == 1
        event = closed[0]
        assert event.start == pytest.approx(2.0)
        assert event.end == pytest.approx(5.0)
        assert event.duration == pytest.approx(3.0)
        assert event.max_intensity == 5.0
        assert event.threshold == 1.0
        assert event.timeline[0][0] == pytest.approx(2.0)
        assert event.timeline[-1][0] <= event.end

    def test_short_glitch_is_discarded(self):
        agg = make_aggregator()
        times = [i * 0.1 for i in range(50)]
        opened, closed, discarded = feed(agg, times, lambda t: 5.0 if 2.0 <= t < 2.2 else 0.0)
        assert opened == ["a"]
        assert closed == []
        assert discarded == ["a"]

    def test_gap_shorter_than_cooldown_merges(self):
        agg = make_aggregator()
        times = [i * 0.1 for i in range(100)]
        active = lambda t: 5.0 if (2.0 <= t <= 3.0 or 3.3 <= t <= 4.5) else 0.0
        opened, closed, _ = feed(agg, times, active)
        assert len(closed) == 1
        assert closed[0].end == pytest.approx(4.5)

    def test_flush_closes_open_events(self):
        agg = make_aggregator()
        feed(agg, [i * 0.1 for i in range(30)], lambda t: 5.0 if t >= 1.0 else 0.0)
        closed = agg.flush()
        assert len(closed) == 1
        assert closed[0].end == pytest.approx(2.9)

    def test_suppressed_anomaly_creates_no_event(self):
        agg = make_aggregator(suppresses={"primary": ["secondary"]})
        for i in range(30):
            t = i * 0.1
            agg.update(t, {
                "primary": Reading(5.0, 1.0, "det"),
                "secondary": Reading(5.0, 1.0, "det"),
            })
        closed = agg.flush()
        assert [e.anomaly_id for e in closed] == ["primary"]

    def test_timeline_is_capped(self):
        agg = make_aggregator(max_timeline_points=50)
        feed(agg, [i * 0.01 for i in range(2000)], lambda t: 5.0)
        event = agg.flush()[0]
        assert len(event.timeline) <= 101
