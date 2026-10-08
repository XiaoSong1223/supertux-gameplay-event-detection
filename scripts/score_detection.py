"""Run the frozen detector on exactly the 40 annotated held-out source frames.

Only frame identifiers/timestamps select inputs. Boxes are used by the scorer
after predictions have been saved. This avoids scoring unannotated video frames
as empty ground truth or silently using a neighboring 15 fps frame.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2

from gameplay_events.detection import SpriteDetector
from gameplay_events.evaluation import evaluate_detections

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    frozen = json.loads((ROOT / "data/freeze.json").read_text())
    for filename, expected in frozen["files"].items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen file changed: {filename}; retain and label any rerun")
    cv2.setNumThreads(1)
    truth_path = ROOT / "data/annotations/detection_evaluation.json"
    truth = json.loads(truth_path.read_text())
    selections = {f["frame_index"]: f["timestamp"] for f in truth["frames"]}
    detector = SpriteDetector(ROOT / "assets/templates/detector.json")
    capture = cv2.VideoCapture(str(ROOT / "data/videos/evaluation.webm"))
    if not capture.isOpened():
        raise FileNotFoundError("Run python scripts/fetch_assets.py first")
    images = []
    for index in range(max(selections) + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Decode failed at source frame {index}")
        if index in selections:
            if frame.shape[1] != 854:
                frame = cv2.resize(frame, (854, round(frame.shape[0] * 854 / frame.shape[1])))
            images.append((index, selections[index], frame))
    capture.release()

    def predict(item):
        index, timestamp, frame = item
        return {"video_id": "evaluation", "frame_index": index,
                "timestamp": timestamp, "processed_size": [854, 480],
                "detections": [asdict(d) for d in detector.detect(frame)]}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(predict, images))
    destination = ROOT / "outputs/metrics"
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "detection_predictions.json"
    prediction_path.write_text(json.dumps({"frames": predictions}, indent=2) + "\n")
    result = {
        "schema_version": "1.0",
        "scope": "40 fixed held-out frames; AI-assisted visual labels without human audit",
        "ground_truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "frozen_files_verified": True,
        "detection_wall_seconds": round(time.perf_counter() - started, 3),
        "detection": evaluate_detections(truth["frames"], predictions, iou_threshold=0.5),
    }
    (destination / "detection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
