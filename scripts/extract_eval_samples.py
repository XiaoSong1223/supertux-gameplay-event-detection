#!/usr/bin/env python3
"""Extract the frozen held-out detection sample frames without running a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


SAMPLE_COUNT = 40
START_SECONDS = 2.0
END_SECONDS = 92.0
EXPECTED_SIZE = (854, 480)
EXPECTED_FPS = 60.0
TRACKING_SEGMENTS = (("track-01", 8.0, 11.0), ("track-02", 38.0, 41.0), ("track-03", 68.0, 71.0))
TRACKING_ANNOTATION_FPS = 5.0


def sample_targets(fps: float) -> list[dict[str, float | int]]:
    targets = []
    for index in range(SAMPLE_COUNT):
        target_seconds = START_SECONDS + index * (END_SECONDS - START_SECONDS) / (
            SAMPLE_COUNT - 1
        )
        # Nearest source frame; target values in this schedule never land at an
        # exact half-frame, so this is independent of tie-breaking convention.
        frame_index = int(target_seconds * fps + 0.5)
        targets.append(
            {
                "sample": index + 1,
                "target_seconds": target_seconds,
                "frame_index": frame_index,
                "timestamp": frame_index / fps,
            }
        )
    return targets


def tracking_targets(fps: float) -> list[dict[str, float | int | str]]:
    targets: list[dict[str, float | int | str]] = []
    sample_index = 1
    for segment_id, start, end in TRACKING_SEGMENTS:
        frame_count = round((end - start) * TRACKING_ANNOTATION_FPS)
        for segment_frame_index in range(frame_count):
            target_seconds = start + segment_frame_index / TRACKING_ANNOTATION_FPS
            frame_index = int(target_seconds * fps + 0.5)
            targets.append(
                {
                    "sample": sample_index,
                    "segment_id": segment_id,
                    "segment_frame_index": segment_frame_index,
                    "target_seconds": target_seconds,
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps,
                }
            )
            sample_index += 1
    return targets


def extract(
    video_path: Path,
    output_dir: Path,
    *,
    targets: list[dict[str, float | int | str]] | None = None,
    selection: str = "40-point linspace from 2.0 to 92.0 s; nearest source frame",
) -> list[dict[str, float | int | str]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(fps - EXPECTED_FPS) > 1e-3:
        raise RuntimeError(f"expected {EXPECTED_FPS:g} fps, found {fps:g}")
    if (width, height) != EXPECTED_SIZE:
        raise RuntimeError(f"expected {EXPECTED_SIZE}, found {(width, height)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = targets or sample_targets(fps)
    by_frame = {int(item["frame_index"]): item for item in targets}
    last_frame = max(by_frame)
    manifest: list[dict[str, float | int | str]] = []
    frame_index = 0
    while frame_index <= last_frame:
        ok, frame = capture.read()
        if not ok:
            break
        target = by_frame.get(frame_index)
        if target is not None:
            filename = f"sample_{target['sample']:02d}_frame_{frame_index:06d}.png"
            path = output_dir / filename
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"could not write frame: {path}")
            manifest.append({**target, "file": filename})
        frame_index += 1
    capture.release()

    if len(manifest) != len(targets):
        missing = sorted(set(by_frame) - {int(item["frame_index"]) for item in manifest})
        raise RuntimeError(
            f"decoded {len(manifest)} of {len(targets)} samples; missing {missing}"
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "source_fps": fps,
                "source_size": [width, height],
                "selection": selection,
                "samples": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_contact_sheets(output_dir, manifest)
    return manifest


def _write_contact_sheets(
    output_dir: Path, manifest: list[dict[str, float | int | str]], per_sheet: int = 4
) -> None:
    """Write review-only 2-column sheets; pixels remain at source scale."""

    for sheet_index in range(math.ceil(len(manifest) / per_sheet)):
        entries = manifest[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        cells = []
        for entry in entries:
            image = cv2.imread(str(output_dir / str(entry["file"])))
            if image is None:
                raise RuntimeError(f"could not reread extracted frame: {entry['file']}")
            label = (
                f"sample {entry['sample']:02d} | frame {entry['frame_index']} | "
                f"t={entry['timestamp']:.6f}s"
            )
            cv2.rectangle(image, (0, 0), (390, 27), (0, 0, 0), thickness=-1)
            cv2.putText(
                image,
                label,
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(image)
        while len(cells) < per_sheet:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[index : index + 2]) for index in range(0, per_sheet, 2)]
        sheet = np.vstack(rows)
        path = output_dir / f"contact_{sheet_index + 1:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"could not write contact sheet: {path}")


def render_ground_truth_overlays(
    annotation_path: Path,
    sample_dir: Path,
    manifest: list[dict[str, float | int | str]],
) -> Path:
    """Render review overlays containing ground truth only."""

    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = {int(frame["frame_index"]): frame for frame in document["frames"]}
    overlay_dir = sample_dir / "gt_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_manifest: list[dict[str, float | int | str]] = []
    colors = {"enemy": (40, 220, 40), "coin": (0, 190, 255)}
    for entry in manifest:
        frame_index = int(entry["frame_index"])
        image = cv2.imread(str(sample_dir / str(entry["file"])))
        if image is None:
            raise RuntimeError(f"could not read extracted frame: {entry['file']}")
        frame = annotations.get(frame_index)
        if frame is None:
            raise RuntimeError(f"annotation is missing source frame {frame_index}")
        for item_index, item in enumerate(frame["objects"], start=1):
            x1, y1, x2, y2 = (int(value) for value in item["bbox"])
            color = colors[item["label"]]
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            text_label = f"{item['label']} {item_index}"
            cv2.putText(
                image,
                text_label,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        output_path = overlay_dir / str(entry["file"])
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"could not write overlay: {output_path}")
        overlay_manifest.append(dict(entry))
    _write_contact_sheets(overlay_dir, overlay_manifest)
    return overlay_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/videos/evaluation.webm"),
        help="held-out source video",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotation_work/evaluation_detection_samples"),
        help="directory for raw PNG samples and manifest",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="optional ground-truth JSON to render as review-only overlays",
    )
    parser.add_argument(
        "--mode",
        choices=("detection", "tracking"),
        default="detection",
        help="frozen sampling schedule to extract",
    )
    args = parser.parse_args()
    if args.mode == "tracking":
        targets = tracking_targets(EXPECTED_FPS)
        selection = (
            "5 fps in frozen half-open segments [8,11), [38,41), [68,71); "
            "nearest source frame"
        )
    else:
        targets = sample_targets(EXPECTED_FPS)
        selection = "40-point linspace from 2.0 to 92.0 s; nearest source frame"
    manifest = extract(
        args.video,
        args.output_dir,
        targets=targets,
        selection=selection,
    )
    print(f"extracted {len(manifest)} raw frames to {args.output_dir}")
    if args.annotations is not None:
        overlay_dir = render_ground_truth_overlays(
            args.annotations, args.output_dir, manifest
        )
        print(f"rendered ground-truth-only overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
