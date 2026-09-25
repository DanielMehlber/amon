"""Tests for per-run diagnostic logging setup."""

import logging

from amon.logging_setup import (
    attach_session_log,
    configure_logging,
    detach_session_log,
)


def test_attach_session_log_writes_named_file(tmp_path):
    configure_logging(
        {"logging": {"console": False, "console_level": "INFO", "file_level": "INFO"}}
    )
    path = attach_session_log(
        {
            "logging": {
                "console": False,
                "console_level": "INFO",
                "file_level": "INFO",
                "dir": str(tmp_path),
            }
        },
        "Test-Session",
    )
    assert path == tmp_path / "Test-Session.log"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "diagnostic log:" in text
    detach_session_log()


def test_file_debug_console_info_are_independent(tmp_path, capsys):
    configure_logging(
        {"logging": {"console": True, "console_level": "INFO", "file_level": "DEBUG"}}
    )
    path = attach_session_log(
        {
            "logging": {
                "console": True,
                "console_level": "INFO",
                "file_level": "DEBUG",
                "dir": str(tmp_path),
            }
        },
        "Split-Levels",
    )
    logging.getLogger("amon.aggregate").debug("probe-below-threshold-detail")
    logging.getLogger("amon.pipeline").info(
        "event hud/cam01/text: 37.0s-40.0s (3.0s, peak 1.00)"
    )
    detach_session_log()

    file_text = path.read_text(encoding="utf-8")
    assert "probe-below-threshold-detail" in file_text
    assert "event hud/cam01/text" in file_text

    captured = capsys.readouterr()
    assert "event hud/cam01/text" in captured.err
    assert "probe-below-threshold-detail" not in captured.err
