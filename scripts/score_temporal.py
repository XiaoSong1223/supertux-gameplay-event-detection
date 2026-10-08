"""Score only annotated tracking frames and explicitly declared event windows."""
import hashlib
import json
from pathlib import Path

from gameplay_events.evaluation import evaluate_events, evaluate_tracking

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def checksum(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    output = ROOT / "outputs/metrics"
    output.mkdir(parents=True, exist_ok=True)
    tracking_path = "data/annotations/tracking_evaluation.json"
    gt = read(tracking_path)
    indices = {f["frame_index"] for f in gt["frames"]}
    rows = [json.loads(line) for line in (ROOT / "outputs/evaluation/tracks.jsonl").read_text().splitlines() if line]
    # Include retained predicted boxes through temporary misses; observed flag is
    # preserved for inspection. Ground truth controls only the scoring cadence.
    predictions = [{**row, "video_id": "evaluation"} for row in rows if row.get("frame_index") in indices]
    if {f["frame_index"] for f in predictions} != indices:
        raise ValueError("tracking run lacks annotated source frames")
    (output / "tracking_predictions.json").write_text(json.dumps({"frames": predictions}, indent=2) + "\n")
    tracking = {"ground_truth_sha256": checksum(tracking_path),
                "scope": "45 frames at 5 fps in three predefined 3-second windows; AI visual labels",
                "prediction_policy": "Observed and temporarily predicted tracks are both scored",
                "tracking": evaluate_tracking(gt["frames"], predictions)}
    (output / "tracking.json").write_text(json.dumps(tracking, indent=2, allow_nan=False) + "\n")

    event_path = "data/annotations/events_evaluation_windows.json"
    references = read(event_path)
    windows = references["scoring_intervals"]
    names = references["event_types"]

    def in_scope(event):
        return event["event"] in names and any(a <= event["timestamp"] < b for a, b in windows)

    truth = [e for e in references["events"] if in_scope(e)]
    result = {"ground_truth_sha256": checksum(event_path), "scoring_intervals": windows,
              "scope": "Three 2-second interiors; 5 fps visual reference, <=0.2 s sampling uncertainty; not full-video accuracy",
              "matching": "One-to-one same-type timestamp match within 0.5 s; event-to-object identity is not scored",
              "full_run_event_counts": {}}
    from collections import Counter
    for variant, directory in [("smoothing", "evaluation"), ("no_smoothing", "evaluation_no_smoothing")]:
        all_events = read(f"outputs/{directory}/events.json")
        selected = [e for e in all_events if in_scope(e)]
        result[variant] = evaluate_events(truth, selected, event_types=names, tolerance_seconds=.5)
        result["full_run_event_counts"][variant] = dict(Counter(e["event"] for e in all_events))
        (output / f"events_{variant}_scored.json").write_text(json.dumps({"events": selected}, indent=2) + "\n")
    (output / "events.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tracking": {k: v for k, v in tracking["tracking"].items() if k != "tracks"},
                      "event_counts": result["full_run_event_counts"]}, indent=2))


if __name__ == "__main__":
    main()
