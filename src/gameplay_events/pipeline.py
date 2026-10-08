"""Ordered video-to-event pipeline and auditable output writers."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import warnings

import cv2
import imageio_ffmpeg
import numpy as np

from .camera import CameraMotion
from .detection import SpriteDetector
from .events import EventEngine
from .tracking import Tracker
from .types import Detection, Event, Track


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parents[1]
_CACHE_SCHEMA_VERSION = 1


def _find_default_detector_config(
    asset_roots: Iterable[Path] | None = None,
) -> Path:
    """Locate detector assets in source, wheel, shared-prefix, or cwd layouts."""

    roots = list(asset_roots) if asset_roots is not None else [
        _PACKAGE_DIR / "assets",
        _PROJECT_ROOT / "assets",
        Path(sys.prefix) / "share" / "gameplay-events" / "assets",
        Path.cwd() / "assets",
    ]
    candidates = [root / "templates" / "detector.json" for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return a stable expected path so normal validation can produce a useful
    # error instead of failing while this module is imported.
    return candidates[0]


_DEFAULT_DETECTOR_CONFIG = _find_default_detector_config()


@dataclass(frozen=True)
class PipelineConfig:
    video: Path | str
    output: Path | str
    detector_config: Path | str = field(default_factory=lambda: _DEFAULT_DETECTOR_CONFIG)
    start: float = 0.0
    end: float | None = None
    fps: float = 15.0
    workers: int = 2
    smoothing: bool = True
    hud: bool = True
    cache: Path | str | None = None
    write_video: bool = True


@dataclass
class _FrameWork:
    frame_index: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection] | Future[list[Detection]]


def sample_frame_indices(
    source_fps: float,
    frame_count: int,
    start: float,
    end: float | None,
    processing_fps: float,
) -> tuple[int, list[int]]:
    """Return an exact integer sampling stride and absolute source indices.

    The requested interval is half open: ``start <= timestamp < end``.
    Sampling remains aligned to source frame zero, so timestamps do not drift
    when the start time changes.
    """

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source FPS must be finite and positive")
    if frame_count < 0:
        raise ValueError("frame count must be non-negative")
    if not math.isfinite(processing_fps) or processing_fps <= 0:
        raise ValueError("processing FPS must be finite and positive")
    if processing_fps > source_fps:
        raise ValueError("processing FPS cannot exceed source FPS")
    ratio = source_fps / processing_fps
    stride = int(round(ratio))
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"processing FPS {processing_fps:g} must exactly divide source FPS "
            f"{source_fps:g}"
        )
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be finite and non-negative")
    duration = frame_count / source_fps
    resolved_end = duration if end is None else end
    if not math.isfinite(resolved_end) or resolved_end < start:
        raise ValueError("end must be finite and greater than or equal to start")
    if resolved_end > duration + 1e-6:
        raise ValueError(
            f"end {resolved_end:g}s exceeds source duration {duration:.3f}s"
        )

    start_index = max(0, int(math.ceil(start * source_fps - 1e-9)))
    stop_index = min(frame_count, int(math.ceil(resolved_end * source_fps - 1e-9)))
    first_index = ((start_index + stride - 1) // stride) * stride
    return stride, list(range(first_index, stop_index, stride))


def run_pipeline(
    config: PipelineConfig,
    *,
    detector: Any | None = None,
    camera: CameraMotion | Any | None = None,
    tracker: Tracker | Any | None = None,
    event_engine: EventEngine | Any | None = None,
    coin_counter: Any | None = None,
) -> dict[str, Any]:
    """Run analysis and write raw detections, tracks, events, metadata, and demo."""

    video_path = Path(config.video).expanduser().resolve()
    output_dir = Path(config.output).expanduser().resolve()
    detector_config_path = Path(config.detector_config).expanduser().resolve()
    cache_path = Path(config.cache).expanduser().resolve() if config.cache else None
    _validate_pipeline_config(config, video_path, detector_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = _sha256_file(video_path)
    config_sha256 = _sha256_file(detector_config_path)
    detector_code_sha256 = _sha256_file(Path(__file__).with_name("detection.py"))
    detector_settings = json.loads(detector_config_path.read_text(encoding="utf-8"))
    processed_width = int(detector_settings["processed_width"])
    hud_height = int(detector_settings.get("hud_exclusion_height", 28))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {video_path}")

    overlay_writer: _OverlayWriter | None = None
    raw_file = None
    executor: ThreadPoolExecutor | None = None
    analysis_started = time.perf_counter()
    all_events: list[Event] = []
    recent_events: deque[Event] = deque(maxlen=5)
    event_counts: Counter[str] = Counter()
    scene_cuts: list[dict[str, float | int]] = []
    processed_frames = 0
    resized_height_warning = False
    cache_used = cache_path is not None

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        stride, frame_indices = sample_frame_indices(
            source_fps,
            source_frame_count,
            config.start,
            config.end,
            config.fps,
        )
        resolved_end = (
            source_frame_count / source_fps if config.end is None else float(config.end)
        )
        cache_metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "detector_code_sha256": detector_code_sha256,
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "processing_fps": float(config.fps),
            "start_seconds": float(config.start),
            "end_seconds": resolved_end,
            "sample_stride": stride,
            "processed_width": processed_width,
        }
        cached_rows = (
            _load_detection_cache(cache_path, cache_metadata) if cache_path else None
        )
        if cached_rows is not None:
            _validate_cache_frame_set(cached_rows, frame_indices)

        if cached_rows is None:
            detector = detector if detector is not None else SpriteDetector(detector_config_path)
            cv2.setNumThreads(1)
            executor = ThreadPoolExecutor(
                max_workers=config.workers,
                thread_name_prefix="sprite-detection",
            )

        camera = camera if camera is not None else CameraMotion(hud_height=hud_height)
        tracker = tracker if tracker is not None else Tracker(
            max_age_seconds=0.35,
            confirmation_hits=1,
            max_center_distance=35.0,
        )
        event_engine = event_engine if event_engine is not None else EventEngine(
            appearance_confirmation_hits=3 if config.smoothing else 1,
            disappearance_grace_seconds=0.35 if config.smoothing else 0.0,
        )
        if config.hud and coin_counter is None:
            coin_counter = _create_coin_counter()
        elif not config.hud:
            coin_counter = None

        raw_path = output_dir / "raw_detections.jsonl"
        raw_file = raw_path.open("w", encoding="utf-8")
        raw_file.write(json.dumps({"_meta": cache_metadata}, sort_keys=True) + "\n")
        tracks_path = output_dir / "tracks.jsonl"

        if config.write_video and frame_indices:
            overlay_writer = _OverlayWriter(
                output_dir / "demo.mp4",
                processed_width,
                _processed_height(source_width, source_height, processed_width),
                config.fps,
            )

        pending: deque[_FrameWork] = deque()
        max_pending = max(1, config.workers * 2)
        next_requested = iter(frame_indices)
        wanted = next(next_requested, None)

        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            source_index = 0
            while wanted is not None and source_index <= frame_indices[-1]:
                ok, source_frame = capture.read()
                if not ok:
                    break
                if source_index == wanted:
                    frame = _resize_for_detector(source_frame, processed_width)
                    if frame.shape[0] != 480 and not resized_height_warning:
                        warnings.warn(
                            "processed frame height is not the calibrated 480 pixels; "
                            "detector accuracy may be reduced",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        resized_height_warning = True
                    timestamp = source_index / source_fps
                    if cached_rows is None:
                        assert executor is not None
                        detections: list[Detection] | Future[list[Detection]] = (
                            executor.submit(detector.detect, frame)
                        )
                    else:
                        if source_index not in cached_rows:
                            raise ValueError(
                                f"cache is missing sampled source frame {source_index}"
                            )
                        cached_timestamp, cached_detections = cached_rows[source_index]
                        if not math.isclose(
                            cached_timestamp, timestamp, rel_tol=0.0, abs_tol=1e-9
                        ):
                            raise ValueError(
                                f"cache timestamp mismatch for source frame {source_index}"
                            )
                        detections = cached_detections
                    pending.append(_FrameWork(source_index, timestamp, frame, detections))
                    wanted = next(next_requested, None)
                    if len(pending) >= max_pending:
                        processed_frames += _consume_work(
                            pending.popleft(),
                            raw_file,
                            tracks_file,
                            camera,
                            tracker,
                            event_engine,
                            coin_counter,
                            overlay_writer,
                            all_events,
                            recent_events,
                            event_counts,
                            scene_cuts,
                        )
                        _report_progress(processed_frames, pending, raw_file)
                source_index += 1

            while pending:
                processed_frames += _consume_work(
                    pending.popleft(),
                    raw_file,
                    tracks_file,
                    camera,
                    tracker,
                    event_engine,
                    coin_counter,
                    overlay_writer,
                    all_events,
                    recent_events,
                    event_counts,
                    scene_cuts,
                )
                _report_progress(processed_frames, pending, raw_file)

            if wanted is not None:
                raise RuntimeError(
                    "video decoding ended at source frame "
                    f"{source_index} before requested frame {wanted}"
                )

        # EOF is not evidence that an enemy disappeared.
        trailing_events = event_engine.finish()
        if trailing_events:
            all_events.extend(trailing_events)
            for event in trailing_events:
                event_counts[event.event] += 1

        analysis_wall_seconds = time.perf_counter() - analysis_started
        events_payload = [event.to_dict() for event in all_events]
        _write_json(output_dir / "events.json", events_payload)
        _write_events_csv(output_dir / "events.csv", all_events)
        if overlay_writer is not None:
            overlay_writer.close()
        demo_video_error = None if overlay_writer is None else overlay_writer.error
        run_metadata: dict[str, Any] = {
            "sources": {
                "video": str(video_path),
                "video_sha256": source_sha256,
                "detector_config": str(detector_config_path),
                "detector_config_sha256": config_sha256,
                "detector_code_sha256": detector_code_sha256,
                "raw_detection_cache": str(cache_path) if cache_path else None,
            },
            "parameters": {
                "start_seconds": float(config.start),
                "end_seconds": resolved_end,
                "source_fps": source_fps,
                "source_frame_count": source_frame_count,
                "source_duration_seconds": source_frame_count / source_fps,
                "processing_fps": float(config.fps),
                "sample_stride": stride,
                "workers": config.workers,
                "smoothing": config.smoothing,
                "hud_enabled": coin_counter is not None,
                "video_enabled": config.write_video,
                "processed_width": processed_width,
                "source_width": source_width,
                "source_height": source_height,
                "tracker": _component_settings(
                    tracker,
                    (
                        "max_age_seconds",
                        "confirmation_hits",
                        "max_center_distance",
                        "min_iou",
                        "distance_weight",
                        "velocity_smoothing",
                    ),
                ),
                "event_engine": _component_settings(
                    event_engine,
                    (
                        "appearance_confirmation_hits",
                        "appearance_confirmation_seconds",
                        "disappearance_grace_seconds",
                    ),
                ),
                "camera_motion": _component_settings(
                    camera,
                    (
                        "hud_height",
                        "max_corners",
                        "quality_level",
                        "min_distance",
                        "min_points",
                        "fb_max_error",
                        "min_inlier_ratio",
                        "max_translation",
                        "cut_correlation_threshold",
                        "cut_difference_threshold",
                    ),
                ),
                "coin_counter": None
                if coin_counter is None
                else _component_settings(
                    coin_counter,
                    (
                        "confirm_frames",
                        "confirm_seconds",
                        "min_read_confidence",
                        "max_digits",
                    ),
                ),
            },
            "timings": {"analysis_wall_seconds": analysis_wall_seconds},
            "processed_frames": processed_frames,
            "event_counts": dict(sorted(event_counts.items())),
            "scene_cuts": scene_cuts,
            "cache_used": cache_used,
            "demo_audio": "omitted",
            "demo_video_error": demo_video_error,
            "confidence_note": (
                "Detection, track, HUD, and event confidence values are "
                "uncalibrated heuristic evidence scores, not probabilities."
            ),
        }
        _write_json(output_dir / "run.json", run_metadata)
        if demo_video_error is not None:
            raise RuntimeError(
                "annotated video encoding failed; analysis artifacts were saved: "
                + demo_video_error
            )
        return run_metadata
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if raw_file is not None:
            raw_file.flush()
            raw_file.close()
        if overlay_writer is not None:
            overlay_writer.close()
        capture.release()


def _consume_work(
    work: _FrameWork,
    raw_file: Any,
    tracks_file: Any,
    camera: Any,
    tracker: Any,
    event_engine: Any,
    coin_counter: Any | None,
    overlay_writer: "_OverlayWriter | None",
    all_events: list[Event],
    recent_events: deque[Event],
    event_counts: Counter[str],
    scene_cuts: list[dict[str, float | int]],
) -> int:
    detections = (
        work.detections.result()
        if isinstance(work.detections, Future)
        else work.detections
    )
    detections = list(detections)
    raw_record = {
        "frame_index": work.frame_index,
        "timestamp": work.timestamp,
        "detections": [asdict(detection) for detection in detections],
    }
    raw_file.write(json.dumps(raw_record, sort_keys=True) + "\n")

    camera_shift = camera.update(work.frame)
    scene_cut = bool(camera.scene_cut)
    if scene_cut:
        tracker.reset()
        event_engine.reset()
        if coin_counter is not None:
            coin_counter.reset()
        scene_cuts.append({"frame_index": work.frame_index, "timestamp": work.timestamp})
        camera_shift = (0.0, 0.0)

    tracks = tracker.update(detections, work.timestamp, camera_shift=camera_shift)
    frame_events = list(event_engine.update(tracks, work.timestamp))
    if coin_counter is not None:
        hud_events = coin_counter.update(work.frame, work.timestamp)
        if hud_events:
            frame_events.extend(hud_events)

    for event in frame_events:
        if not isinstance(event, Event):
            raise TypeError("event producers must return Event values")
        all_events.append(event)
        recent_events.append(event)
        event_counts[event.event] += 1

    track_record = {
        "frame_index": work.frame_index,
        "timestamp": work.timestamp,
        "camera_shift": [float(camera_shift[0]), float(camera_shift[1])],
        "scene_cut": scene_cut,
        "tracks": [asdict(track) for track in tracks],
    }
    tracks_file.write(json.dumps(track_record, sort_keys=True) + "\n")

    if overlay_writer is not None:
        overlay_writer.write(
            work.frame,
            tracks,
            work.timestamp,
            coin_counter is not None,
            None if coin_counter is None else coin_counter.last_count,
            list(recent_events),
        )
    return 1


def _report_progress(
    processed_frames: int,
    pending: deque[_FrameWork],
    raw_file: Any,
) -> None:
    if processed_frames > 0 and processed_frames % 30 == 0:
        raw_file.flush()
        pending_note = f", {len(pending)} queued" if pending else ""
        print(
            f"Processed {processed_frames} analysis frames{pending_note}",
            file=sys.stderr,
            flush=True,
        )


def _validate_pipeline_config(
    config: PipelineConfig,
    video_path: Path,
    detector_config_path: Path,
) -> None:
    if not video_path.is_file():
        raise ValueError(f"video does not exist: {video_path}")
    if not detector_config_path.is_file():
        raise ValueError(f"detector config does not exist: {detector_config_path}")
    if not math.isfinite(config.start) or config.start < 0:
        raise ValueError("start must be finite and non-negative")
    if config.end is not None and (
        not math.isfinite(config.end) or config.end < config.start
    ):
        raise ValueError("end must be finite and greater than or equal to start")
    if not math.isfinite(config.fps) or config.fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(config.workers, bool) or not isinstance(config.workers, int):
        raise ValueError("workers must be a positive integer")
    if config.workers < 1:
        raise ValueError("workers must be a positive integer")


def _processed_height(source_width: int, source_height: int, processed_width: int) -> int:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("video reports invalid frame dimensions")
    if source_width == processed_width:
        return source_height
    return max(1, int(round(source_height * processed_width / source_width)))


def _resize_for_detector(frame: np.ndarray, processed_width: int) -> np.ndarray:
    if frame.shape[1] == processed_width:
        return frame
    height = _processed_height(frame.shape[1], frame.shape[0], processed_width)
    interpolation = cv2.INTER_AREA if frame.shape[1] > processed_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (processed_width, height), interpolation=interpolation)


def _create_coin_counter() -> Any | None:
    try:
        from .hud import CoinCounter
    except ImportError:
        warnings.warn("HUD counter module is unavailable; continuing without HUD events")
        return None
    return CoinCounter()


def _component_settings(component: Any, names: Iterable[str]) -> dict[str, Any]:
    """Read the effective public scalar settings from an instantiated component."""

    settings: dict[str, Any] = {}
    for name in names:
        if hasattr(component, name):
            value = getattr(component, name)
            if value is None or isinstance(value, (bool, int, float, str)):
                settings[name] = value
    return settings


def _validate_cache_frame_set(
    rows: dict[int, tuple[float, list[Detection]]],
    frame_indices: list[int],
) -> None:
    expected = set(frame_indices)
    actual = set(rows)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} sampled frame rows")
        if unexpected:
            details.append(f"contains {len(unexpected)} unexpected frame rows")
        raise ValueError("detection cache frame set mismatch: " + "; ".join(details))


def _load_detection_cache(
    path: Path,
    expected_metadata: dict[str, Any],
) -> dict[int, tuple[float, list[Detection]]]:
    if not path.is_file():
        raise ValueError(f"detection cache does not exist: {path}")
    with path.open("r", encoding="utf-8") as cache_file:
        first_line = cache_file.readline()
        if not first_line:
            raise ValueError("detection cache is empty")
        try:
            metadata_record = json.loads(first_line)
        except json.JSONDecodeError as exc:
            raise ValueError("detection cache metadata is invalid JSON") from exc
        metadata = metadata_record.get("_meta")
        if not isinstance(metadata, dict):
            raise ValueError("detection cache is missing its metadata record")
        mismatches = [
            key
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "detection cache metadata mismatch: " + ", ".join(sorted(mismatches))
            )

        rows: dict[int, tuple[float, list[Detection]]] = {}
        for line_number, line in enumerate(cache_file, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame_index_value = record["frame_index"]
                if (
                    isinstance(frame_index_value, bool)
                    or not isinstance(frame_index_value, int)
                    or frame_index_value < 0
                ):
                    raise ValueError("frame_index must be a non-negative integer")
                frame_index = frame_index_value
                timestamp = float(record["timestamp"])
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError("timestamp must be finite and non-negative")
                detections = [Detection(**item) for item in record["detections"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid detection cache row at line {line_number}"
                ) from exc
            if frame_index in rows:
                raise ValueError(f"duplicate cache row for source frame {frame_index}")
            rows[frame_index] = (timestamp, detections)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_events_csv(path: Path, events: Iterable[Event]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "event",
                "confidence",
                "confirmed_at",
                "track_id",
                "evidence",
            ],
        )
        writer.writeheader()
        for event in events:
            row = event.to_dict()
            row["evidence"] = json.dumps(row["evidence"], sort_keys=True)
            writer.writerow(row)


class _OverlayWriter:
    def __init__(self, path: Path, frame_width: int, frame_height: int, fps: float) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = 300
        self.output_width = frame_width + self.panel_width
        self.output_height = frame_height
        if self.output_width % 2:
            self.output_width += 1
        if self.output_height % 2:
            self.output_height += 1
        self._writer = imageio_ffmpeg.write_frames(
            str(path),
            (self.output_width, self.output_height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=fps,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="warning",
        )
        self._writer.send(None)
        self._closed = False
        self.error: str | None = None

    def write(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        timestamp: float,
        hud_enabled: bool,
        coin_count: int | None,
        recent_events: list[Event],
    ) -> None:
        if self._closed or self.error is not None:
            return
        try:
            canvas = np.full(
                (self.output_height, self.output_width, 3),
                (18, 18, 22),
                dtype=np.uint8,
            )
            canvas[: self.frame_height, : self.frame_width] = frame
            for track in tracks:
                x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
                if track.observed:
                    color = (40, 220, 255) if track.label == "coin" else (70, 80, 245)
                    label = str(track.track_id)
                    thickness = 2
                else:
                    color = (115, 115, 115)
                    label = str(track.track_id)
                    thickness = 1
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    canvas,
                    label,
                    (max(0, min(x1, self.frame_width - 24)), max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            panel_x = self.frame_width + 18
            _panel_text(
                canvas,
                f"TIME  {timestamp:8.3f} s",
                panel_x,
                32,
                (235, 235, 235),
                0.62,
            )
            if not hud_enabled:
                hud_text = "HUD  disabled"
            elif coin_count is None:
                hud_text = "LAST CONFIRMED HUD  unavailable"
            else:
                hud_text = f"LAST CONFIRMED HUD  {coin_count}"
            _panel_text(canvas, hud_text, panel_x, 66, (40, 220, 255), 0.40)
            _panel_text(canvas, "IDs: red enemy / gold coin / gray predicted", panel_x, 87, (180, 180, 190), 0.31)
            _panel_text(canvas, "RECENT EVENTS", panel_x, 124, (180, 180, 190), 0.50)
            y = 153
            for event in recent_events[-5:]:
                color = _event_color(event.event)
                _panel_text(
                    canvas,
                    f"{event.timestamp:7.2f}  {event.event}",
                    panel_x,
                    y,
                    color,
                    0.43,
                )
                if event.track_id is not None:
                    _panel_text(
                        canvas,
                        f"track #{event.track_id} | score {event.confidence:.2f}",
                        panel_x + 55,
                        y + 18,
                        color,
                        0.38,
                    )
                    y += 44
                else:
                    y += 28

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            self._writer.send(np.ascontiguousarray(rgb))
        except Exception as exc:  # ffmpeg errors must not discard analysis output
            self.error = f"{type(exc).__name__}: {exc}"
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._writer.close()
            except Exception as exc:  # preserve the first, most relevant failure
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"


def _panel_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _event_color(event_name: str) -> tuple[int, int, int]:
    if event_name == "enemy_detected":
        return 90, 100, 255
    if event_name == "enemy_disappeared":
        return 170, 130, 255
    return 70, 225, 255
