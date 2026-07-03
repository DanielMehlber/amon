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
from amon.synthetic import EXPECTED_EVENTS, SCHEDULE

#: Tolerance for event boundaries.  Sliding-window metrics (blink rate)
#: respond up to one window (2 s) late, plus MAD/cooldown slack.
START_TOLERANCE = 2.6
END_TOLERANCE = 3.6


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
            hits = [
                e for e in events
                if matches(e["anomaly_id"], pattern)
                and abs(e["start"] - start) <= START_TOLERANCE
                and abs(e["end"] - end) <= END_TOLERANCE
            ]
            assert len(hits) == 1, f"{key}: expected 1 event matching {pattern}, got {hits}"

    def test_no_unexpected_events(self, db_events):
        _, _, events = db_events
        for event in events:
            windows = [
                (start, end) for key, start, end in SCHEDULE
                if matches(event["anomaly_id"], EXPECTED_EVENTS[key])
            ]
            assert any(
                abs(event["start"] - start) <= START_TOLERANCE
                and abs(event["end"] - end) <= END_TOLERANCE
                for start, end in windows
            ), f"unexpected event {event['anomaly_id']} at {event['start']:.1f}s"

    def test_suppression_removed_noise_during_overlap(self, db_events):
        _, _, events = db_events
        overlap = next(entry for entry in SCHEDULE if entry[0] == "overlap_flicker_noise")
        noise_events = [
            e for e in events
            if e["anomaly_id"] == "temporal/noise" and e["end"] > overlap[1] - 1 and e["start"] < overlap[2] + 1
        ]
        assert noise_events == []

    def test_durations_are_calculated(self, db_events):
        _, _, events = db_events
        for event in events:
            assert event["duration"] == pytest.approx(event["end"] - event["start"], abs=1e-6)
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

    def test_calibration_record(self, db_events):
        db, session_id, _ = db_events
        calibration = db.get_calibration(session_id)
        assert calibration is not None
        assert calibration["media"] and Path(calibration["media"]).exists()

        annotations = calibration["annotations"]
        assert len(annotations["keypoints"]) >= 20
        elements = annotations["hud_elements"]
        assert len(elements) == 2
        blink_rates = sorted(e["blink_hz"] for e in elements)
        assert blink_rates[0] == pytest.approx(0.0, abs=0.2)
        assert blink_rates[1] == pytest.approx(2.0, abs=0.4)
        assert any("CAM 01" in e["text"] for e in elements)

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
