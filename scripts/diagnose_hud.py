#!/usr/bin/env python3
"""Create a post-evaluation HUD failure diagnostic without changing the model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from gameplay_events.hud import CoinCounter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/diagnostics/hud_posthoc.png"
SAMPLES = (("development", 5.0), ("development", 15.0), ("evaluation", 5.0), ("evaluation", 15.0))
SLOT_XS = (815, 823, 831, 839)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs() -> None:
    freeze = json.loads((ROOT / "data/freeze.json").read_text(encoding="utf-8"))
    for relative in ("src/gameplay_events/hud.py", "assets/hud/coin_digits_v1.npz"):
        actual = sha256(ROOT / relative)
        if freeze["files"][relative] != actual:
            raise RuntimeError(f"frozen artifact changed: {relative}")


def slot_predictions(counter: CoinCounter, frame: np.ndarray) -> list[tuple[str, float]]:
    predictions = []
    for x in SLOT_XS:
        scores = counter._feature(frame, x) @ counter._weights
        order = np.argsort(scores)
        label = int(order[-1])
        margin = float(scores[order[-1]] - scores[order[-2]])
        confidence = 1.0 - math.exp(-max(margin, 0.0) / counter._margin_scale)
        predictions.append(("N" if label == 10 else str(label), confidence))
    return predictions


def make_panel(counter: CoinCounter, video_id: str, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(ROOT / f"data/videos/{video_id}.webm"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = int(timestamp * fps + 0.5)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode {video_id} frame {frame_index}")

    crop_x1, crop_x2, crop_y2 = 790, 854, 28
    crop = frame[:crop_y2, crop_x1:crop_x2].copy()
    enlarged = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    for x in SLOT_XS:
        x1 = (x - crop_x1) * 8
        cv2.rectangle(enlarged, (x1, 7 * 8), (x1 + 8 * 8 - 1, 19 * 8 - 1), (0, 255, 0), 2)
    panel = np.zeros((315, enlarged.shape[1], 3), dtype=np.uint8)
    panel[: enlarged.shape[0]] = enlarged

    predictions = slot_predictions(counter, frame)
    reading = counter.read(frame)
    read_text = "REJECT" if reading is None else str(reading.count)
    cv2.putText(
        panel,
        f"{video_id}  t={timestamp:.1f}s  frame={frame_index}  read={read_text}",
        (8, 251),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    detail = "  ".join(
        f"x{x}: {label}/{confidence:.2f}" for x, (label, confidence) in zip(SLOT_XS, predictions)
    )
    cv2.putText(
        panel,
        detail,
        (8, 282),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "green boxes = frozen 8x12 slots; N = icon/non-digit class",
        (8, 307),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    verify_frozen_inputs()
    counter = CoinCounter()
    panels = [make_panel(counter, video_id, timestamp) for video_id, timestamp in SAMPLES]
    image = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError(f"could not write {OUTPUT}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
