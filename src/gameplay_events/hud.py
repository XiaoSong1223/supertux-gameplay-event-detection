"""Conservative OCR and temporal events for the SuperTux coin HUD."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .types import Event


@dataclass(frozen=True)
class CoinRead:
    """One accepted OCR observation; confidence is a heuristic, not probability."""

    count: int
    confidence: float
    bbox: tuple[int, int, int, int]
    digits: str


def _default_model_path() -> Path:
    relative = Path("assets/hud/coin_digits_v1.npz")
    candidates = (
        Path(__file__).resolve().parents[2] / relative,
        Path.cwd() / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "HUD OCR model not found; run `.venv/bin/python scripts/build_hud_templates.py`"
    )


class CoinCounter:
    """Read a right-aligned coin count and emit confirmed count-change events.

    The OCR is fixed to the SuperTux v0.6.3 HUD geometry represented by the
    checked-in model. Frames of other sizes are normalized to 854x480 first.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        confirm_frames: int = 3,
        confirm_seconds: float = 0.08,
        min_read_confidence: float = 0.58,
        max_digits: int = 4,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if not math.isfinite(confirm_seconds) or confirm_seconds < 0:
            raise ValueError("confirm_seconds must be finite and non-negative")
        if not 0.0 <= min_read_confidence <= 1.0:
            raise ValueError("min_read_confidence must be in [0, 1]")
        if not 1 <= max_digits <= 4:
            raise ValueError("max_digits must be in [1, 4]")

        path = Path(model_path) if model_path is not None else _default_model_path()
        with np.load(path, allow_pickle=False) as model:
            self._weights = model["weights"].astype(np.float32)
            self._margin_scale = float(model["margin_scale"])
            self._canonical_size = tuple(int(v) for v in model["canonical_size"])
            self._slot_x = int(model["slot_x"])
            self._slot_y = int(model["slot_y"])
            self._slot_w = int(model["slot_w"])
            self._slot_h = int(model["slot_h"])
            self._background_y = tuple(int(v) for v in model["background_y"])

        self.confirm_frames = confirm_frames
        self.confirm_seconds = confirm_seconds
        self.min_read_confidence = min_read_confidence
        self.max_digits = max_digits
        self.reset()

    @property
    def last_count(self) -> int | None:
        return self._last_count

    def reset(self) -> None:
        self._last_count: int | None = None
        self._last_timestamp: float | None = None
        self._pending_count: int | None = None
        self._pending_since: float | None = None
        self._pending_frames = 0
        self._pending_confidences: list[float] = []

    def _feature(self, frame: np.ndarray, x: int) -> np.ndarray:
        patch = frame[
            self._slot_y : self._slot_y + self._slot_h,
            x : x + self._slot_w,
        ].astype(np.float32)
        by1, by2 = self._background_y
        background = np.median(
            frame[by1:by2, x : x + self._slot_w].reshape(-1, 3), axis=0
        ).astype(np.float32)
        delta = (patch - background) / 128.0
        hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        white = np.clip((120.0 - hsv[:, :, 1]) / 120.0, 0.0, 1.0)
        white *= np.clip((hsv[:, :, 2] - 70.0) / 185.0, 0.0, 1.0)
        signed_luma = np.clip(delta.mean(axis=2), -2.0, 2.0)
        color_distance = np.clip(np.linalg.norm(delta, axis=2), 0.0, 3.0)
        return np.concatenate(
            (signed_luma.ravel(), color_distance.ravel(), white.ravel(), np.ones(1))
        ).astype(np.float32)

    def read(self, frame_bgr: np.ndarray) -> CoinRead | None:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a non-empty HxWx3 NumPy array")
        if frame_bgr.size == 0 or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be a non-empty uint8 BGR image")

        width, height = self._canonical_size
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (width, height):
            frame = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_bgr

        right_slot = self._slot_x + 2 * self._slot_w
        slots: list[tuple[int, float]] = []
        for offset in range(self.max_digits - 1, -1, -1):
            x = right_slot - offset * self._slot_w
            scores = self._feature(frame, x) @ self._weights
            order = np.argsort(scores)
            label = int(order[-1])
            margin = float(scores[order[-1]] - scores[order[-2]])
            # Bounded evidence score based on classifier separation. The scale
            # comes only from training margins and is not probability calibrated.
            confidence = float(1.0 - math.exp(-max(margin, 0.0) / self._margin_scale))
            slots.append((label, confidence))

        # Read the contiguous right-aligned run. Class 10 denotes icon/blank.
        digits_reversed: list[str] = []
        confidences: list[float] = []
        for label, confidence in reversed(slots):
            if label == 10:
                break
            if confidence < self.min_read_confidence or label > 9:
                return None
            digits_reversed.append(str(label))
            confidences.append(confidence)
        if not digits_reversed:
            return None
        digits = "".join(reversed(digits_reversed))
        if len(digits) > 1 and digits[0] == "0":
            return None

        # The slot immediately left of a short number must be non-digit with
        # useful separation. This rejects partial strings and transition blur.
        boundary_index = self.max_digits - len(digits) - 1
        if boundary_index >= 0:
            boundary_label, boundary_confidence = slots[boundary_index]
            if boundary_label != 10 or boundary_confidence < self.min_read_confidence * 0.8:
                return None
            confidences.append(boundary_confidence)

        start_x = right_slot - (len(digits) - 1) * self._slot_w
        confidence = float(min(confidences))
        return CoinRead(
            count=int(digits),
            confidence=confidence,
            bbox=(start_x, self._slot_y, right_slot + self._slot_w, self._slot_y + self._slot_h),
            digits=digits,
        )

    def update(self, frame_bgr: np.ndarray, timestamp: float) -> list[Event]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self._last_timestamp = timestamp

        observation = self.read(frame_bgr)
        if observation is None or observation.confidence < self.min_read_confidence:
            self._clear_pending()
            return []
        if observation.count == self._last_count:
            self._clear_pending()
            return []

        if observation.count != self._pending_count:
            self._pending_count = observation.count
            self._pending_since = timestamp
            self._pending_frames = 1
            self._pending_confidences = [observation.confidence]
        else:
            self._pending_frames += 1
            self._pending_confidences.append(observation.confidence)

        assert self._pending_since is not None
        if (
            self._pending_frames < self.confirm_frames
            or timestamp - self._pending_since < self.confirm_seconds
        ):
            return []

        old_count = self._last_count
        new_count = observation.count
        confidence = float(min(self._pending_confidences))
        first_timestamp = self._pending_since
        support_frames = self._pending_frames
        self._last_count = new_count
        self._clear_pending()
        if old_count is None:
            return []
        return [
            Event(
                event="coin_count_changed",
                timestamp=first_timestamp,
                confirmed_at=timestamp,
                confidence=confidence,
                evidence={
                    "old_count": old_count,
                    "new_count": new_count,
                    "delta": new_count - old_count,
                    "support_frames": support_frames,
                    "ocr_bbox": observation.bbox,
                    "ocr_digits": observation.digits,
                },
            )
        ]

    def _clear_pending(self) -> None:
        self._pending_count = None
        self._pending_since = None
        self._pending_frames = 0
        self._pending_confidences = []
