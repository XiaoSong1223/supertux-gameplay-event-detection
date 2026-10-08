"""Build the tiny SuperTux coin-counter OCR model.

The labels below were manually read from ``data/videos/development.webm``.
No evaluation-video frame is used.  The generated model is deliberately small:
ridge-regression weights over four 8x12 HUD patches (digits 0-9 plus a
non-digit class).  Re-run this script only when changing the development clip
or the canonical 854x480 crop geometry.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/videos/development.webm"
OUTPUT = ROOT / "assets/hud/coin_digits_v1.npz"

# Independent visual reads of stable HUD frames in the development clip.
LABELS = (
    (2.0, "105"),
    (5.0, "108"),
    (10.0, "108"),
    (20.0, "123"),
    (26.0, "123"),
    (29.0, "134"),
    (30.0, "135"),
    (40.0, "158"),
    (50.0, "165"),
    (60.0, "279"),
    (70.0, "288"),
    (80.0, "293"),
    (90.0, "310"),
    (97.0, "310"),
)

CANONICAL_SIZE = (854, 480)
SLOT_X = 823
SLOT_Y = 7
SLOT_W = 8
SLOT_H = 12
BACKGROUND_Y = (22, 28)
NON_DIGIT = 10


def feature(frame: np.ndarray, x: int) -> np.ndarray:
    patch = frame[SLOT_Y : SLOT_Y + SLOT_H, x : x + SLOT_W].astype(np.float32)
    background = np.median(
        frame[BACKGROUND_Y[0] : BACKGROUND_Y[1], x : x + SLOT_W].reshape(-1, 3),
        axis=0,
    ).astype(np.float32)
    delta = (patch - background) / 128.0
    hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    white = np.clip((120.0 - hsv[:, :, 1]) / 120.0, 0.0, 1.0)
    white *= np.clip((hsv[:, :, 2] - 70.0) / 185.0, 0.0, 1.0)
    signed_luma = np.clip(delta.mean(axis=2), -2.0, 2.0)
    color_distance = np.clip(np.linalg.norm(delta, axis=2), 0.0, 3.0)
    return np.concatenate((signed_luma.ravel(), color_distance.ravel(), white.ravel()))


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {VIDEO}")

    samples: list[np.ndarray] = []
    targets: list[int] = []
    # Nearby frames add compression variation while staying inside the stable
    # manually labelled interval.
    for timestamp, text in LABELS:
        for offset in (-0.12, -0.06, 0.0, 0.06, 0.12):
            capture.set(cv2.CAP_PROP_POS_MSEC, (timestamp + offset) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"Could not decode development frame at {timestamp + offset:.3f}s")
            if (frame.shape[1], frame.shape[0]) != CANONICAL_SIZE:
                frame = cv2.resize(frame, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
            # The counter is right aligned. All calibration labels have three
            # digits; the immediately preceding slot contains the icon's 'x'.
            samples.append(feature(frame, SLOT_X - SLOT_W))
            targets.append(NON_DIGIT)
            for index, character in enumerate(text):
                samples.append(feature(frame, SLOT_X + index * SLOT_W))
                targets.append(int(character))
    capture.release()

    x = np.stack(samples).astype(np.float32)
    x = np.column_stack((x, np.ones(len(x), dtype=np.float32)))
    y = np.eye(11, dtype=np.float32)[np.asarray(targets)]
    regularization = 100.0
    weights = np.linalg.solve(
        x.T @ x + regularization * np.eye(x.shape[1], dtype=np.float32), x.T @ y
    ).astype(np.float32)

    train_scores = x @ weights
    ordered = np.sort(train_scores, axis=1)
    correct = np.argmax(train_scores, axis=1) == np.asarray(targets)
    margins = ordered[:, -1] - ordered[:, -2]
    # A lower-tail scale turns margins into a conservative heuristic score at
    # runtime. It is not a probability or a reported accuracy metric.
    margin_scale = float(max(np.percentile(margins[correct], 10), 0.02))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        weights=weights,
        margin_scale=np.float32(margin_scale),
        canonical_size=np.asarray(CANONICAL_SIZE, dtype=np.int32),
        slot_x=np.int32(SLOT_X),
        slot_y=np.int32(SLOT_Y),
        slot_w=np.int32(SLOT_W),
        slot_h=np.int32(SLOT_H),
        background_y=np.asarray(BACKGROUND_Y, dtype=np.int32),
        model_version=np.asarray("coin_digits_v1"),
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(samples)} labelled patches)")


if __name__ == "__main__":
    main()
