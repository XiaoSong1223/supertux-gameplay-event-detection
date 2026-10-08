"""Re-render saved analysis with readable IDs; leave inference outputs unchanged."""
import argparse
import json
from pathlib import Path

import cv2

from gameplay_events.hud import CoinCounter
from gameplay_events.pipeline import _OverlayWriter
from gameplay_events.types import Event, Track

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "evaluation"], required=True)
    args = parser.parse_args()
    output = ROOT / "outputs" / args.split
    rows = {r["frame_index"]: r for r in (json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines())}
    events = [Event(**e) for e in json.loads((output / "events.json").read_text())]
    counter = CoinCounter()
    capture = cv2.VideoCapture(str(ROOT / "data/videos" / f"{args.split}.webm"))
    temp = output / "demo_rendering.mp4"
    writer = _OverlayWriter(temp, 854, 480, 15)
    recent = []
    pointer = 0
    written = 0
    try:
        for index in range(max(rows) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"decode failed at {index}")
            if index not in rows:
                continue
            row = rows[index]
            timestamp = row["timestamp"]
            if row.get("scene_cut"):
                counter.reset()
            counter.update(frame, timestamp)
            while pointer < len(events) and events[pointer].confirmed_at <= timestamp + 1e-9:
                recent.append(events[pointer])
                pointer += 1
            writer.write(frame, [Track(**t) for t in row["tracks"]], timestamp, True, counter.last_count, recent[-5:])
            written += 1
    finally:
        capture.release()
        writer.close()
    if writer.error:
        raise RuntimeError(writer.error)
    if written != len(rows):
        raise RuntimeError("incomplete demo render")
    temp.replace(output / "demo.mp4")
    print(f"{args.split}: rendered {written} saved analysis frames; detection/tracking/event outputs unchanged")


if __name__ == "__main__":
    main()
