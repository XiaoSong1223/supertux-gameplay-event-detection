import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from gameplay_events.pipeline import (
    PipelineConfig,
    _find_default_detector_config,
    run_pipeline,
    sample_frame_indices,
)
from gameplay_events.types import Detection


class FakeDetector:
    def __init__(self, *, delayed: bool = False) -> None:
        self.calls = 0
        self.delayed = delayed

    def detect(self, frame: np.ndarray) -> list[Detection]:
        marker = int(frame[0, 0, 0])
        if self.delayed and marker < 20:
            time.sleep(0.04)
        self.calls += 1
        return [Detection("enemy", (5.0, 5.0, 15.0, 15.0), 0.9, "snowball")]


class NoMotionCamera:
    scene_cut = False

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        self.scene_cut = False
        return 0.0, 0.0


class CutOnSecondFrameCamera:
    def __init__(self) -> None:
        self.calls = 0
        self.scene_cut = False

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        self.calls += 1
        self.scene_cut = self.calls == 2
        return 0.0, 0.0


def make_video(path: Path, *, fps: float = 8.0, frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()


def make_detector_config(path: Path) -> None:
    path.write_text(
        json.dumps({"processed_width": 64, "hud_exclusion_height": 4}),
        encoding="utf-8",
    )


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_exact_time_sampling_uses_absolute_source_indices() -> None:
    stride, indices = sample_frame_indices(60.0, 600, 0.05, 0.25, 15.0)
    assert stride == 4
    assert indices == [4, 8, 12]
    assert [index / 60.0 for index in indices] == pytest.approx(
        [1 / 15, 2 / 15, 3 / 15]
    )

    with pytest.raises(ValueError, match="exactly divide"):
        sample_frame_indices(59.94, 600, 0.0, 1.0, 15.0)


def test_tiny_pipeline_preserves_order_and_eof_emits_no_disappearance(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    detector_config = tmp_path / "detector.json"
    output = tmp_path / "output"
    make_video(video)
    make_detector_config(detector_config)
    detector = FakeDetector(delayed=True)

    run = run_pipeline(
        PipelineConfig(
            video=video,
            output=output,
            detector_config=detector_config,
            start=0.0,
            end=0.5,
            fps=4.0,
            workers=2,
            smoothing=False,
            hud=False,
            write_video=False,
        ),
        detector=detector,
        camera=NoMotionCamera(),
    )

    raw = jsonl(output / "raw_detections.jsonl")
    tracks = jsonl(output / "tracks.jsonl")
    events = json.loads((output / "events.json").read_text())
    assert detector.calls == 2
    assert [row["frame_index"] for row in raw[1:]] == [0, 2]
    assert [row["frame_index"] for row in tracks] == [0, 2]
    assert [event["event"] for event in events] == ["enemy_detected"]
    assert run["event_counts"] == {"enemy_detected": 1}
    assert run["parameters"]["source_fps"] == pytest.approx(8.0)
    assert run["parameters"]["processing_fps"] == 4.0
    assert run["parameters"]["tracker"]["confirmation_hits"] == 1
    assert run["parameters"]["tracker"]["max_age_seconds"] == 0.35
    assert run["parameters"]["event_engine"] == {
        "appearance_confirmation_hits": 1,
        "appearance_confirmation_seconds": None,
        "disappearance_grace_seconds": 0.0,
    }
    assert run["demo_audio"] == "omitted"


def test_cache_replay_skips_detector_and_rejects_metadata_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    detector_config = tmp_path / "detector.json"
    first_output = tmp_path / "first"
    replay_output = tmp_path / "replay"
    make_video(video)
    make_detector_config(detector_config)
    common = dict(
        video=video,
        detector_config=detector_config,
        start=0.0,
        end=1.0,
        fps=2.0,
        smoothing=False,
        hud=False,
        write_video=False,
    )
    run_pipeline(
        PipelineConfig(output=first_output, **common),
        detector=FakeDetector(),
        camera=NoMotionCamera(),
    )
    cache = first_output / "raw_detections.jsonl"

    detector_that_must_not_run = FakeDetector()
    replay = run_pipeline(
        PipelineConfig(output=replay_output, cache=cache, **common),
        detector=detector_that_must_not_run,
        camera=NoMotionCamera(),
    )
    assert detector_that_must_not_run.calls == 0
    assert replay["cache_used"] is True
    assert (replay_output / "events.json").read_text() == (
        first_output / "events.json"
    ).read_text()

    with pytest.raises(ValueError, match="processing_fps"):
        run_pipeline(
            PipelineConfig(
                output=tmp_path / "mismatch",
                cache=cache,
                **{**common, "fps": 4.0},
            ),
            camera=NoMotionCamera(),
        )

    lines = cache.read_text().splitlines()
    incomplete_cache = tmp_path / "incomplete.jsonl"
    incomplete_cache.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing 1 sampled frame rows"):
        run_pipeline(
            PipelineConfig(output=tmp_path / "incomplete", cache=incomplete_cache, **common),
            camera=NoMotionCamera(),
        )


def test_scene_cut_reset_does_not_fabricate_disappearance(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    detector_config = tmp_path / "detector.json"
    output = tmp_path / "cut"
    make_video(video)
    make_detector_config(detector_config)

    class FirstOnlyDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame: np.ndarray) -> list[Detection]:
            self.calls += 1
            if self.calls == 1:
                return [Detection("enemy", (5.0, 5.0, 15.0, 15.0), 0.9)]
            return []

    run_pipeline(
        PipelineConfig(
            video=video,
            output=output,
            detector_config=detector_config,
            start=0.0,
            end=1.0,
            fps=2.0,
            workers=1,
            smoothing=False,
            hud=False,
            write_video=False,
        ),
        detector=FirstOnlyDetector(),
        camera=CutOnSecondFrameCamera(),
    )

    events = json.loads((output / "events.json").read_text())
    assert [event["event"] for event in events] == ["enemy_detected"]
    run = json.loads((output / "run.json").read_text())
    assert run["scene_cuts"] == [{"frame_index": 4, "timestamp": 0.5}]


def test_truncated_decode_is_not_reported_as_a_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "truncated.fake"
    video.write_bytes(b"test source identity")
    detector_config = tmp_path / "detector.json"
    make_detector_config(detector_config)

    class TruncatedCapture:
        def __init__(self, path: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            values = {
                cv2.CAP_PROP_FPS: 8.0,
                cv2.CAP_PROP_FRAME_COUNT: 8.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            }
            return values[prop]

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= 2:
                return False, None
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            self.index += 1
            return True, frame

        def release(self) -> None:
            pass

    monkeypatch.setattr("gameplay_events.pipeline.cv2.VideoCapture", TruncatedCapture)
    with pytest.raises(
        RuntimeError,
        match="ended at source frame 2 before requested frame 4",
    ):
        run_pipeline(
            PipelineConfig(
                video=video,
                output=tmp_path / "truncated-output",
                detector_config=detector_config,
                start=0.0,
                end=1.0,
                fps=2.0,
                workers=1,
                smoothing=False,
                hud=False,
                write_video=False,
            ),
            detector=FakeDetector(),
            camera=NoMotionCamera(),
        )


def test_default_detector_config_supports_package_local_assets(tmp_path: Path) -> None:
    package_assets = tmp_path / "gameplay_events" / "assets"
    config = package_assets / "templates" / "detector.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")

    assert _find_default_detector_config([package_assets]) == config
