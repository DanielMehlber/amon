"""End-to-end integration tests: full pipeline on the synthetic video.

The synthetic video's anomaly schedule is the ground truth; the pipeline
(including the background worker process and SQLite persistence) must
reproduce it: one event per scheduled anomaly, correct timing, media on
disk and no false positives.
"""

from pathlib import Path

import pytest

from amon.aggregate import SuppressionRules
from amon.db import Database
from amon.synthetic import DURATION, EXPECTED_EVENT_COUNTS, EXPECTED_EVENTS, SCHEDULE

#: Tolerance for event boundaries.  Sliding-window metrics (blink rate)
#: respond up to one window (2 s) late, plus MAD/cooldown slack.
START_TOLERANCE = 2.6
END_TOLERANCE = 3.6

#: Secondary detections that appear when ``aggregation.suppresses`` does not
#: silence them (defaults ship with no suppressions).  Contrast changes the
#: global histogram and can trip a calibrated blink rate without a real blink
#: anomaly — only expected if ``temporal/contrast → hud/*`` is not configured.
KNOWN_SIDE_EFFECTS = [
    ("hud/*/blink", 30.0, 33.0),
]


def matches(anomaly_id: str, pattern: str) -> bool:
    return SuppressionRules({pattern: []})._rules[0][0].match(anomaly_id) is not None


@pytest.fixture(scope="module")
def db_events(completed_session):
    config, session_id = completed_session
    db = Database(Path(config["data_dir"]) / "amon.sqlite")
    yield db, session_id, db.list_events(session_id)
    db.close()


class TestEventDetection:
    def test_every_scheduled_anomaly_is_reported_once(self, db_events):
        _, _, events = db_events
        for key, start, end in SCHEDULE:
            pattern = EXPECTED_EVENTS[key]
            expected = EXPECTED_EVENT_COUNTS.get(key, 1)
            hits = [
                e
                for e in events
                if matches(e["anomaly_id"], pattern)
                and abs(e["start"] - start) <= START_TOLERANCE
                and abs(e["end"] - end) <= END_TOLERANCE
            ]
            assert (
                len(hits) == expected
            ), f"{key}: expected {expected} event(s) matching {pattern}, got {hits}"

    def test_hud_new_emits_one_event_per_overlay(self, db_events):
        _, _, events = db_events
        new_events = [
            e
            for e in events
            if matches(e["anomaly_id"], "hud/*/new")
            and abs(e["start"] - 69.0) <= START_TOLERANCE
        ]
        ids = sorted(e["anomaly_id"] for e in new_events)
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert all(aid.endswith("/new") for aid in ids)

    def test_hud_new_mutate_keeps_spawn_id(self, db_events):
        """Content rewrite mid-lifetime must not open hud/1000/new in that window."""
        _, _, events = db_events
        hits = [
            e
            for e in events
            if e["anomaly_id"].endswith("/new")
            and abs(e["start"] - 91.0) <= START_TOLERANCE
        ]
        assert len(hits) == 1
        event = hits[0]
        assert event["anomaly_id"] == "hud/alert/new"
        assert event["end"] - event["start"] >= 3.0  # spans ALERT and 1000 phases
        assert not any(
            e["anomaly_id"] == "hud/1000/new"
            and abs(e["start"] - 91.0) <= START_TOLERANCE
            for e in events
        )

    def test_hud_new_cycle_keeps_spawn_id_through_steps_and_blink(self, db_events):
        """Stepped 1000→2000 plus blink must yield one hud/1000/new event."""
        _, _, events = db_events
        hits = [
            e
            for e in events
            if e["anomaly_id"].endswith("/new")
            and abs(e["start"] - 98.0) <= START_TOLERANCE
        ]
        assert len(hits) == 1
        event = hits[0]
        assert event["anomaly_id"] == "hud/1000/new"
        # Steps (5s) + blink (~4s) — allow cooldown slack on the end.
        assert event["end"] - event["start"] >= 6.0
        assert not any(e["anomaly_id"] == "hud/2000/new" for e in events)

    def test_hud_new_symbols_are_not_letters(self, db_events):
        """Centre crosshair + side dot must open two symbol-N /new events."""
        _, _, events = db_events
        hits = [
            e
            for e in events
            if matches(e["anomaly_id"], "hud/symbol-*/new")
            and abs(e["start"] - 109.0) <= START_TOLERANCE
        ]
        ids = sorted(e["anomaly_id"] for e in hits)
        assert ids == ["hud/symbol-1/new", "hud/symbol-2/new"]
        # No false letter/digit channels in that window.
        letterish = [
            e
            for e in events
            if e["anomaly_id"].endswith("/new")
            and abs(e["start"] - 109.0) <= START_TOLERANCE
            and not e["anomaly_id"].startswith("hud/symbol-")
        ]
        assert letterish == []

    def test_parallel_hud_changes_are_differentiated(self, db_events):
        _, _, events = db_events
        text_hits = [
            e
            for e in events
            if matches(e["anomaly_id"], "hud/*/text")
            and abs(e["start"] - 87.0) <= START_TOLERANCE
        ]
        position_hits = [
            e
            for e in events
            if matches(e["anomaly_id"], "hud/*/position")
            and abs(e["start"] - 87.0) <= START_TOLERANCE
        ]
        assert len(text_hits) == 1
        assert len(position_hits) == 1
        assert text_hits[0]["anomaly_id"].split("/")[1] != position_hits[0][
            "anomaly_id"
        ].split("/")[1]

    def test_no_unexpected_events(self, db_events):
        _, _, events = db_events
        for event in events:
            windows = [
                (start, end)
                for key, start, end in SCHEDULE
                if matches(event["anomaly_id"], EXPECTED_EVENTS[key])
            ]
            windows.extend(
                (start, end)
                for pattern, start, end in KNOWN_SIDE_EFFECTS
                if matches(event["anomaly_id"], pattern)
            )
            assert any(
                abs(event["start"] - start) <= START_TOLERANCE
                and abs(event["end"] - end) <= END_TOLERANCE
                for start, end in windows
            ), f"unexpected event {event['anomaly_id']} at {event['start']:.1f}s"

    def test_suppression_removed_noise_during_overlap(self, db_events):
        _, _, events = db_events
        overlap = next(
            entry for entry in SCHEDULE if entry[0] == "overlap_flicker_noise"
        )
        noise_events = [
            e
            for e in events
            if e["anomaly_id"] == "temporal/noise"
            and e["end"] > overlap[1] - 1
            and e["start"] < overlap[2] + 1
        ]
        assert noise_events == []

    def test_durations_are_calculated(self, db_events):
        _, _, events = db_events
        for event in events:
            assert event["duration"] == pytest.approx(
                event["end"] - event["start"], abs=1e-6
            )
            assert event["duration"] > 0


class TestEventRecords:
    def test_metadata_and_intensity_recorded(self, db_events):
        _, _, events = db_events
        for event in events:
            assert event["max_intensity"] >= event["threshold"] > 0
            assert len(event["timeline"]) >= 2
            peak = max(v for _, v in event["timeline"])
            assert peak == pytest.approx(event["max_intensity"])
            assert isinstance(event["metadata"], dict)

    def test_media_generated_for_every_event(self, db_events):
        _, _, events = db_events
        for event in events:
            assert event["media"], f"no media for {event['anomaly_id']}"
            path = Path(event["media"])
            assert path.exists() and path.stat().st_size > 0

    def test_spatial_event_has_highlight_regions(self, db_events):
        _, _, events = db_events
        spatial = [e for e in events if e["anomaly_id"] == "spatial/distortion"]
        assert spatial and spatial[0]["regions"]


class TestSessionAndCalibration:
    def test_session_is_completed(self, db_events):
        db, session_id, _ = db_events
        session = db.get_session(session_id)
        assert session["status"] == "completed"
        assert session["finished_at"] is not None
        assert session["fps"] == pytest.approx(20.0)
        assert "-" in session_id
        assert session["started_at"] > 0

    def test_no_ongoing_events_after_session_ends(self, db_events):
        """Events still open at EOF must be finalized — never left ongoing."""
        _, _, events = db_events
        ongoing = [e for e in events if e.get("status") == "ongoing"]
        assert ongoing == []
        assert all(e.get("status") == "completed" for e in events)

    def test_anomaly_open_at_eof_is_completed(self, db_events):
        """hud_text_until_end runs into EOF; flush must close it as completed."""
        _, _, events = db_events
        hits = [
            e
            for e in events
            if matches(e["anomaly_id"], "hud/*/text")
            and abs(e["start"] - 110.0) <= START_TOLERANCE
        ]
        assert len(hits) == 1
        event = hits[0]
        assert event["status"] == "completed"
        # Last synthetic frame is just under DURATION (114 s @ 20 fps).
        assert event["end"] == pytest.approx(DURATION - 1.0 / 20.0, abs=0.15)
        assert event["duration"] >= 3.0

    def test_calibration_record(self, db_events):
        db, session_id, _ = db_events
        calibration = db.get_calibration(session_id)
        assert calibration is not None
        assert calibration["media"] and Path(calibration["media"]).exists()

        annotations = calibration["annotations"]
        assert len(annotations["keypoints"]) >= 20
        frame = annotations["frame"]
        assert frame["width"] == 320
        assert frame["height"] == 240
        assert frame["fps"] == pytest.approx(20.0)
        elements = annotations["hud_elements"]
        assert len(elements) == 4
        blink_rates = sorted(e["blink_hz"] for e in elements)
        assert blink_rates[0] == pytest.approx(0.0, abs=0.2)
        assert blink_rates[-1] == pytest.approx(2.0, abs=0.4)
        assert any("CAM01" in e["text"] for e in elements)
        assert any("REC" in e["text"] for e in elements)

        thresholds = calibration["thresholds"]
        assert set(thresholds) == {"temporal", "hud", "spatial"}
        for detector_thresholds in thresholds.values():
            assert all(v > 0 for v in detector_thresholds.values())

    def test_events_persist_across_reopen(self, db_events, completed_session):
        config, session_id = completed_session
        _, _, events = db_events
        fresh = Database(Path(config["data_dir"]) / "amon.sqlite")
        assert len(fresh.list_events(session_id)) == len(events)
        fresh.close()
