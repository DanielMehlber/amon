"""Tests for the dropping capture buffer and overload warnings."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from amon.sources.frame_buffer import DroppingFrameBuffer, OverloadMonitor


def _sequential_reader(n: int, size=(8, 12)):
    """Yield n unique frames then signal end-of-stream."""
    lock = threading.Lock()
    state = {"i": 0}

    def read():
        with lock:
            i = state["i"]
            if i >= n:
                return False, None
            state["i"] = i + 1
        image = np.full((*size, 3), i, dtype=np.uint8)
        return True, image

    return read


class TestDroppingFrameBuffer:
    def test_delivers_frames_in_order_when_consumer_keeps_up(self):
        # Buffer large enough that a fast producer does not collapse the sequence.
        buf = DroppingFrameBuffer(_sequential_reader(5), maxsize=8)
        buf.start()
        try:
            got = []
            for _ in range(5):
                item = buf.get(timeout=2.0)
                assert item is not None
                got.append(int(item[0][0, 0, 0]))
            assert got == [0, 1, 2, 3, 4]
            assert buf.dropped == 0
            assert buf.get(timeout=1.0) is None
            assert buf.ended
        finally:
            buf.close()

    def test_drops_oldest_when_consumer_is_behind(self):
        released = threading.Event()
        lock = threading.Lock()
        state = {"i": 0}

        def read():
            # Produce a burst quickly, then block until the test is done so
            # we do not spin after the stream "ends" for our purposes.
            with lock:
                i = state["i"]
                if i >= 20:
                    released.wait(timeout=2.0)
                    return False, None
                state["i"] = i + 1
            return True, np.full((4, 4, 3), i, dtype=np.uint8)

        buf = DroppingFrameBuffer(read, maxsize=1)
        buf.start()
        try:
            # Let the producer overflow the single-slot buffer.
            deadline = time.monotonic() + 2.0
            while buf.pushed < 10 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert buf.pushed >= 10
            assert buf.dropped >= 9  # maxsize=1 → every push after the first drops

            item = buf.get(timeout=1.0)
            assert item is not None
            # Latest (or near-latest) frame — not the first ones produced.
            assert int(item[0][0, 0, 0]) >= 9
        finally:
            released.set()
            buf.close()

    def test_maxsize_must_be_positive(self):
        with pytest.raises(ValueError):
            DroppingFrameBuffer(_sequential_reader(1), maxsize=0)


class TestOverloadMonitor:
    def test_warns_when_drop_ratio_is_high(self):
        mon = OverloadMonitor(
            width=1920,
            height=1080,
            source_fps=30.0,
            window_seconds=5.0,
            warn_drop_ratio=0.25,
            warn_interval_seconds=0.0,
        )
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        with patch("amon.sources.frame_buffer.time.monotonic", side_effect=fake_monotonic):
            with patch("amon.sources.frame_buffer.log.warning") as warn:
                # Simulate steady emits while cumulative drops climb fast.
                dropped = 0
                for i in range(20):
                    clock["t"] = i * 0.1
                    dropped += 3  # 3 drops per emit → 75% drop ratio
                    mon.note_emit(dropped)
                assert warn.called
                message = warn.call_args[0][0] % warn.call_args[0][1:]
                assert "too slow" in message
                assert "1920x1080" in message

    def test_quiet_when_drops_are_rare(self):
        mon = OverloadMonitor(
            width=640,
            height=480,
            source_fps=25.0,
            window_seconds=5.0,
            warn_drop_ratio=0.25,
            warn_interval_seconds=0.0,
        )
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        with patch("amon.sources.frame_buffer.time.monotonic", side_effect=fake_monotonic):
            with patch("amon.sources.frame_buffer.log.warning") as warn:
                for i in range(30):
                    clock["t"] = i * 0.1
                    mon.note_emit(dropped_total=i // 20)  # almost no drops
                assert not warn.called
