#!/usr/bin/env python3
"""Score saved held-out predictions against frozen annotation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gameplay_events.evaluation import (
    evaluate_detections,
    evaluate_events,
    evaluate_tracking,
)


def _read_records(path: Path, collection_key: str) -> list[dict[str, Any]]:
    """Read a JSON document, JSONL file, or directory of JSON documents."""

    if path.is_dir():
        records: list[dict[str, Any]] = []
        files = sorted(child for child in path.iterdir() if child.suffix == ".json")
        if not files:
            raise ValueError(f"no .json files found in {path}")
        for child in files:
            records.extend(_records_from_value(_read_json(child), collection_key, child))
        return records
    if path.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            records.extend(_records_from_value(value, collection_key, path))
        return records
    return _records_from_value(_read_json(path), collection_key, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"input does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _records_from_value(
    value: Any, collection_key: str, source: Path
) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "_meta" in value:
        return []  # A pipeline cache header is not an annotated frame.
    if isinstance(value, dict) and collection_key in value:
        value = value[collection_key]
    elif isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(
            f"{source} must be an object with '{collection_key}', a record, or a record array"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-ground-truth", type=Path)
    parser.add_argument("--detection-predictions", type=Path)
    parser.add_argument("--event-ground-truth", type=Path)
    parser.add_argument("--event-predictions", type=Path)
    parser.add_argument("--tracking-ground-truth", type=Path)
    parser.add_argument("--tracking-predictions", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--event-tolerance", type=float, default=0.5)
    parser.add_argument(
        "--require-coin-count-value-match",
        action="store_true",
        help="require matching old/new HUD counts for coin_count_changed true positives",
    )
    parser.add_argument("--output", type=Path, help="optional metric JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paired_inputs = (
        (
            "detection",
            args.detection_ground_truth,
            args.detection_predictions,
        ),
        ("event", args.event_ground_truth, args.event_predictions),
        ("tracking", args.tracking_ground_truth, args.tracking_predictions),
    )
    for name, ground_truth, predictions in paired_inputs:
        if (ground_truth is None) != (predictions is None):
            raise SystemExit(
                f"{name} evaluation requires both ground-truth and prediction inputs"
            )
    if not any(ground_truth is not None for _, ground_truth, _ in paired_inputs):
        raise SystemExit("provide at least one ground-truth/prediction input pair")

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "inputs": {},
    }
    if args.detection_ground_truth is not None:
        detection_truth = _read_records(args.detection_ground_truth, "frames")
        detection_predictions = _read_records(args.detection_predictions, "frames")
        result["detection"] = evaluate_detections(
            detection_truth,
            detection_predictions,
            iou_threshold=args.iou_threshold,
        )
        result["inputs"]["detection_ground_truth"] = str(
            args.detection_ground_truth
        )
        result["inputs"]["detection_predictions"] = str(
            args.detection_predictions
        )
    if args.event_ground_truth is not None:
        event_truth = _read_records(args.event_ground_truth, "events")
        event_predictions = _read_records(args.event_predictions, "events")
        result["events"] = evaluate_events(
            event_truth,
            event_predictions,
            tolerance_seconds=args.event_tolerance,
            require_coin_count_value_match=args.require_coin_count_value_match,
        )
        result["inputs"]["event_ground_truth"] = str(args.event_ground_truth)
        result["inputs"]["event_predictions"] = str(args.event_predictions)
    if args.tracking_ground_truth is not None:
        tracking_truth = _read_records(args.tracking_ground_truth, "frames")
        tracking_predictions = _read_records(args.tracking_predictions, "frames")
        result["tracking"] = evaluate_tracking(
            tracking_truth,
            tracking_predictions,
            iou_threshold=args.iou_threshold,
        )
        result["inputs"]["tracking_ground_truth"] = str(args.tracking_ground_truth)
        result["inputs"]["tracking_predictions"] = str(
            args.tracking_predictions
        )

    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
