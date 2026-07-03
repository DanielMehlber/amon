"""CLI tests: startup errors and the export command."""
from pathlib import Path

import yaml

from amon.__main__ import main


def write_config(path: Path, **overrides) -> str:
    config = {"video_source": {"class": "amon.sources.file.VideoFileSource",
                               "config": {"path": "does-not-exist.avi"}}}
    config.update(overrides)
    path.write_text(yaml.safe_dump(config))
    return str(path)


class TestCli:
    def test_monitor_reports_unopenable_source(self, tmp_path, capsys):
        config = write_config(tmp_path / "c.yaml")
        assert main(["monitor", config]) == 1
        assert "cannot open" in capsys.readouterr().err

    def test_export_command(self, completed_session, tmp_path, capsys):
        session_config, session_id = completed_session
        config = write_config(tmp_path / "c.yaml", data_dir=session_config["data_dir"])
        out = tmp_path / "report.html"
        assert main(["export", config, "--session", session_id, "--output", str(out)]) == 0
        assert out.exists()
