"""Command-line entry point for offline gameplay event extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pipeline import PipelineConfig, _DEFAULT_DETECTOR_CONFIG, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gameplay-events",
        description="Extract auditable enemy and coin events from a gameplay video.",
    )
    parser.add_argument("--video", type=Path, required=True, help="source gameplay video")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_DETECTOR_CONFIG,
        help="sprite detector JSON configuration",
    )
    parser.add_argument("--start", type=float, default=2.0, help="absolute start time in seconds")
    parser.add_argument("--end", type=float, default=92.0, help="absolute end time in seconds")
    parser.add_argument("--fps", type=float, default=15.0, help="analysis/output frame rate")
    parser.add_argument("--workers", type=int, default=2, help="parallel detector workers")
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="disable event confirmation and disappearance grace",
    )
    parser.add_argument("--no-hud", action="store_true", help="disable HUD coin-count events")
    parser.add_argument("--cache", type=Path, help="validated raw detection JSONL to replay")
    parser.add_argument("--no-video", action="store_true", help="skip annotated MP4 output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        video=args.video,
        output=args.output,
        detector_config=args.config,
        start=args.start,
        end=args.end,
        fps=args.fps,
        workers=args.workers,
        smoothing=not args.no_smoothing,
        hud=not args.no_hud,
        cache=args.cache,
        write_video=not args.no_video,
    )
    try:
        run_pipeline(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"gameplay-events: error: {exc}", file=sys.stderr)
        return 2
    return 0
