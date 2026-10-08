#!/usr/bin/env python3
"""Evaluate the frozen coin HUD reader on independently labelled fixed frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data/annotations/hud_evaluation.json"
DEFAULT_VIDEO = ROOT / "data/videos/evaluation.webm"
DEFAULT_FREEZE = ROOT / "data/freeze.json"
DEFAULT_OUTPUT = ROOT / "outputs/metrics/hud.json"
FROZEN_INPUTS = (
    "src/gameplay_events/hud.py",
    "assets/hud/coin_digits_v1.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_freeze(path: Path) -> dict[str, str]:
    freeze = read_json(path)
    expected = freeze.get("files", {})
    verified: dict[str, str] = {}
    for relative in FROZEN_INPUTS:
        actual = sha256(ROOT / relative)
        if expected.get(relative) != actual:
            raise RuntimeError(
                f"frozen input hash mismatch for {relative}: "
                f"expected {expected.get(relative)}, found {actual}"
            )
        verified[relative] = actual
    return verified


def evaluate(
    annotation_path: Path,
    video_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    verified_inputs = verify_freeze(freeze_path)
    annotations = read_json(annotation_path)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("annotation document must contain a non-empty 'frames' list")

    # Import only after frozen artifact hashes have been verified.
    from gameplay_events.hud import CoinCounter

    counter = CoinCounter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    requested = {int(item["frame_index"]): item for item in frames}
    if len(requested) != len(frames):
        raise ValueError("duplicate frame_index in HUD annotations")
    last_requested = max(requested)
    results: list[dict[str, Any]] = []
    frame_index = 0
    while frame_index <= last_requested:
        ok, frame = capture.read()
        if not ok:
            break
        truth = requested.get(frame_index)
        if truth is not None:
            observation = counter.read(frame)
            ground_truth = truth.get("count")
            if ground_truth is None:
                status = "label_ambiguous"
            elif observation is None:
                status = "rejected"
            elif observation.count == int(ground_truth):
                status = "correct"
            else:
                status = "wrong"
            results.append(
                {
                    "sample": int(truth["sample"]),
                    "frame_index": frame_index,
                    "timestamp": float(truth["timestamp"]),
                    "ground_truth_count": ground_truth,
                    "predicted_count": None if observation is None else observation.count,
                    "confidence": None if observation is None else observation.confidence,
                    "status": status,
                }
            )
        frame_index += 1
    capture.release()
    if len(results) != len(frames):
        decoded = {item["frame_index"] for item in results}
        raise RuntimeError(f"failed to decode annotated frames: {sorted(set(requested) - decoded)}")

    evaluable = sum(item["status"] != "label_ambiguous" for item in results)
    ambiguous = sum(item["status"] == "label_ambiguous" for item in results)
    rejected = sum(item["status"] == "rejected" for item in results)
    correct = sum(item["status"] == "correct" for item in results)
    wrong = sum(item["status"] == "wrong" for item in results)
    accepted = correct + wrong

    return {
        "schema_version": "1.0",
        "evaluation": "frozen_coin_counter_fixed_frames",
        "inputs": {
            "annotations": str(annotation_path.relative_to(ROOT)),
            "annotations_sha256": sha256(annotation_path),
            "video": str(video_path.relative_to(ROOT)),
            "freeze": str(freeze_path.relative_to(ROOT)),
            "verified_frozen_inputs": verified_inputs,
            "source_fps": fps,
            "source_size": list(size),
        },
        "annotation_provenance": annotations.get("annotator"),
        "counts": {
            "sampled_frames": len(results),
            "evaluable_labels": evaluable,
            "label_ambiguous": ambiguous,
            "accepted": accepted,
            "correct": correct,
            "wrong": wrong,
            "rejected": rejected,
        },
        "metrics": {
            "coverage": accepted / evaluable if evaluable else None,
            "accepted_read_exact_accuracy": correct / accepted if accepted else None,
            "overall_exact_rate_including_rejections": correct / evaluable if evaluable else None,
        },
        "metric_notes": {
            "coverage": "accepted / evaluable labels",
            "accepted_read_exact_accuracy": "exact count matches / accepted reads; abstentions excluded",
            "overall_exact_rate_including_rejections": "exact count matches / evaluable labels; abstentions count as misses",
            "confidence": "Frozen reader heuristic, not a calibrated probability",
        },
        "frames": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.annotations.resolve(), args.video.resolve(), args.freeze.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))
    print(json.dumps(result["metrics"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
