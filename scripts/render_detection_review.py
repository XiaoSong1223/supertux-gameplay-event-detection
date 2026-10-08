"""Render all held-out frame predictions beside the frozen reference boxes."""
import json
from pathlib import Path

import cv2
import numpy as np

from gameplay_events.evaluation import evaluate_detections

ROOT = Path(__file__).resolve().parents[1]


def main():
    truth = json.loads((ROOT / "data/annotations/detection_evaluation.json").read_text())["frames"]
    predictions = json.loads((ROOT / "outputs/metrics/detection_predictions.json").read_text())["frames"]
    selected = {g["frame_index"]: (g, p) for g, p in zip(truth, predictions, strict=True)}
    output = ROOT / "outputs/review/detection"
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / "data/videos/evaluation.webm"))
    summaries = []
    for index in range(max(selected) + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"decode failed at {index}")
        if index not in selected:
            continue
        gt, pred = selected[index]
        metrics = evaluate_detections([gt], [pred])["micro"]
        canvas = np.full((540, 854, 3), (24, 24, 24), dtype=np.uint8)
        canvas[60:] = frame
        cv2.putText(canvas, f"SOURCE {gt['timestamp']:.3f}s | TP {metrics['tp']} FP {metrics['fp']} FN {metrics['fn']}",
                    (16, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GREEN: visual reference   MAGENTA: prediction   IoU >= 0.5", (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 210, 210), 1, cv2.LINE_AA)
        for items, color in [(gt["objects"], (90, 240, 90)), (pred["detections"], (240, 80, 240))]:
            for obj in items:
                x1, y1, x2, y2 = [round(x) for x in obj["bbox"]]
                cv2.rectangle(canvas, (x1, y1 + 60), (x2, y2 + 60), color, 1)
        cv2.imwrite(str(output / f"frame_{index:05}.jpg"), canvas)
        summaries.append({"frame_index": index, "timestamp": gt["timestamp"], **metrics})
    capture.release()
    (output.parent / "detection_frame_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    # Chosen after scoring to illustrate failures; this is not a new test set.
    examples = [951, 2058, 2889, 4966]
    panels = [cv2.imread(str(output / f"frame_{i:05}.jpg")) for i in examples]
    contact = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(ROOT / "outputs/review/failure_examples.jpg"), contact)


if __name__ == "__main__":
    main()
