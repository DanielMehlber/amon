"""Command line interface: ``python -m amon <command>``.

Commands:

- ``monitor <config>``: run a monitoring session (Ctrl-C stops gracefully).
- ``report <config>``: launch the browser-based report UI.
- ``export <config> --session <id>``: export a session report.
- ``synth <path>``: generate the synthetic test video.
"""
from __future__ import annotations

import argparse
import logging
import sys

from amon.config import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="amon", description="Video anomaly monitoring framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_monitor = sub.add_parser("monitor", help="run a monitoring session")
    p_monitor.add_argument("config", help="path to the YAML configuration file")
    p_monitor.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    p_report = sub.add_parser("report", help="launch the report UI in the browser")
    p_report.add_argument("config", help="path to the YAML configuration file")

    p_export = sub.add_parser("export", help="export a session report")
    p_export.add_argument("config", help="path to the YAML configuration file")
    p_export.add_argument("--session", required=True, help="session ID to export")
    p_export.add_argument("--format", default=None, help="export format (default: from config)")
    p_export.add_argument("--output", default=None, help="output file path")

    p_synth = sub.add_parser("synth", help="generate the synthetic test video")
    p_synth.add_argument("path", help="output video path (.avi or .mp4)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.command == "synth":
        from amon.synthetic import write_video

        print(write_video(args.path))
        return 0

    config = load_config(args.config)
    if args.command == "monitor":
        return _monitor(config, args.max_frames)
    if args.command == "report":
        from amon.report import serve

        serve(config)
        return 0
    if args.command == "export":
        from amon.exporters import export_session

        out = export_session(config, args.session, args.format, args.output)
        print(out)
        return 0
    return 2


def _monitor(config: dict, max_frames) -> int:
    from amon.pipeline import Pipeline
    from amon.sources import SourceError

    try:
        pipeline = Pipeline(config)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        session_id = pipeline.run(max_frames=max_frames)
    except KeyboardInterrupt:
        print("stopped by user", file=sys.stderr)
        return 0
    print(f"session {session_id} completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
