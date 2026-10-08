"""Small class-aware tracker for sparse gameplay detections.

The tracker deliberately models only translation.  Its velocity is measured in
processed-frame pixels per second after subtracting the supplied camera shift.
This keeps a scrolling camera from being learned a second time as object motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Detection, Track


_SUPPORTED_LABELS = frozenset({"enemy", "coin"})
_INVALID_COST = 1_000_000.0


@dataclass
class _TrackState:
    track: Track
    velocity: tuple[float, float] = (0.0, 0.0)
    last_update: float = 0.0


class Tracker:
    """Associate detections over time with a lightweight constant-velocity model.

    ``camera_shift`` is the image-space translation from the preceding update to
    the current update.  For example, pass ``(-4, 0)`` when the static scene has
    moved four processed-frame pixels left since the preceding frame.
    """

    def __init__(
        self,
        max_age_seconds: float = 0.35,
        confirmation_hits: int = 2,
        max_center_distance: float = 80.0,
        min_iou: float = 0.01,
        distance_weight: float = 0.35,
        velocity_smoothing: float = 0.5,
    ) -> None:
        if not _is_finite_nonnegative(max_age_seconds):
            raise ValueError("max_age_seconds must be finite and non-negative")
        if (
            isinstance(confirmation_hits, bool)
            or not isinstance(confirmation_hits, int)
            or confirmation_hits < 1
        ):
            raise ValueError("confirmation_hits must be at least 1")
        if not _is_finite_positive(max_center_distance):
            raise ValueError("max_center_distance must be finite and positive")
        if not _is_unit_interval(min_iou):
            raise ValueError("min_iou must be between 0 and 1")
        if not _is_unit_interval(distance_weight):
            raise ValueError("distance_weight must be between 0 and 1")
        if not _is_unit_interval(velocity_smoothing):
            raise ValueError("velocity_smoothing must be between 0 and 1")

        self.max_age_seconds = float(max_age_seconds)
        self.confirmation_hits = int(confirmation_hits)
        self.max_center_distance = float(max_center_distance)
        self.min_iou = float(min_iou)
        self.distance_weight = float(distance_weight)
        self.velocity_smoothing = float(velocity_smoothing)

        self._states: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._last_timestamp: float | None = None

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        camera_shift: tuple[float, float] = (0.0, 0.0),
    ) -> list[Track]:
        """Advance the tracker and return all live observed or predicted tracks.

        Updates must have strictly increasing timestamps.  A missed track is
        returned with ``observed=False`` until ``max_age_seconds`` after its last
        detection.  Public ``Track`` values are copies and are safe to mutate.
        """

        timestamp = _validate_timestamp(timestamp, self._last_timestamp)
        shift = _validate_camera_shift(camera_shift)
        checked_detections = [_validate_detection(detection) for detection in detections]

        # Validation is complete before any mutable state is advanced.
        # Expire stale identities before association so a detection following a
        # long timestamp gap cannot revive an episode beyond the age tolerance.
        expired = [
            track_id
            for track_id, state in self._states.items()
            if timestamp - state.track.last_seen > self.max_age_seconds
        ]
        for track_id in expired:
            del self._states[track_id]

        predicted: dict[int, tuple[float, float, float, float]] = {}
        for track_id, state in self._states.items():
            dt = timestamp - state.last_update
            dx = shift[0] + state.velocity[0] * dt
            dy = shift[1] + state.velocity[1] * dt
            predicted[track_id] = _translate(state.track.bbox, dx, dy)

        matches, unmatched_track_ids, unmatched_detection_indices = self._associate(
            checked_detections, predicted
        )

        for track_id, detection_index in matches:
            state = self._states[track_id]
            detection = checked_detections[detection_index]
            dt = timestamp - state.last_update
            previous_center = _center(state.track.bbox)
            detected_center = _center(detection.bbox)
            measured_velocity = (
                (detected_center[0] - previous_center[0] - shift[0]) / dt,
                (detected_center[1] - previous_center[1] - shift[1]) / dt,
            )
            alpha = self.velocity_smoothing
            state.velocity = (
                alpha * measured_velocity[0] + (1.0 - alpha) * state.velocity[0],
                alpha * measured_velocity[1] + (1.0 - alpha) * state.velocity[1],
            )
            state.track.bbox = detection.bbox
            state.track.confidence = detection.confidence
            state.track.last_seen = timestamp
            state.track.hits += 1
            state.track.confirmed = state.track.hits >= self.confirmation_hits
            state.track.observed = True
            if detection.subtype:
                state.track.subtype = detection.subtype
            state.last_update = timestamp

        for track_id in unmatched_track_ids:
            state = self._states[track_id]
            state.track.bbox = predicted[track_id]
            state.track.observed = False
            state.last_update = timestamp

        for detection_index in unmatched_detection_indices:
            detection = checked_detections[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            track = Track(
                track_id=track_id,
                label=detection.label,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen=timestamp,
                last_seen=timestamp,
                hits=1,
                confirmed=self.confirmation_hits == 1,
                observed=True,
                subtype=detection.subtype,
            )
            self._states[track_id] = _TrackState(track=track, last_update=timestamp)

        self._last_timestamp = timestamp
        return [replace(self._states[track_id].track) for track_id in sorted(self._states)]

    def reset(self) -> None:
        """Drop active tracks for a scene cut while retaining monotonic IDs."""

        self._states.clear()
        self._last_timestamp = None

    def _associate(
        self,
        detections: list[Detection],
        predicted: dict[int, tuple[float, float, float, float]],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        unmatched_tracks = set(self._states)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        for label in _SUPPORTED_LABELS:
            track_ids = [
                track_id
                for track_id, state in self._states.items()
                if state.track.label == label
            ]
            detection_indices = [
                index for index, detection in enumerate(detections) if detection.label == label
            ]
            if not track_ids or not detection_indices:
                continue

            costs = np.full(
                (len(track_ids), len(detection_indices)), _INVALID_COST, dtype=np.float64
            )
            for row, track_id in enumerate(track_ids):
                predicted_bbox = predicted[track_id]
                predicted_center = _center(predicted_bbox)
                for column, detection_index in enumerate(detection_indices):
                    detection = detections[detection_index]
                    distance = math.dist(predicted_center, _center(detection.bbox))
                    overlap = _iou(predicted_bbox, detection.bbox)
                    within_distance = distance <= self.max_center_distance
                    sufficient_overlap = self.min_iou > 0.0 and overlap >= self.min_iou
                    if not within_distance and not sufficient_overlap:
                        continue
                    normalized_distance = min(distance / self.max_center_distance, 1.0)
                    costs[row, column] = (
                        self.distance_weight * normalized_distance
                        + (1.0 - self.distance_weight) * (1.0 - overlap)
                    )

            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if costs[row, column] >= _INVALID_COST:
                    continue
                track_id = track_ids[row]
                detection_index = detection_indices[column]
                matches.append((track_id, detection_index))
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(detection_index)

        return matches, unmatched_tracks, unmatched_detections


def _validate_timestamp(timestamp: float, previous: float | None) -> float:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise TypeError("timestamp must be a finite number")
    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and non-negative")
    if previous is not None and timestamp <= previous:
        raise ValueError("timestamps must be strictly increasing")
    return timestamp


def _validate_camera_shift(camera_shift: tuple[float, float]) -> tuple[float, float]:
    try:
        values = tuple(camera_shift)
    except TypeError as exc:
        raise TypeError("camera_shift must contain two finite numbers") from exc
    if len(values) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("camera_shift must contain two finite numbers")
    return float(values[0]), float(values[1])


def _validate_detection(detection: Detection) -> Detection:
    if not isinstance(detection, Detection):
        raise TypeError("detections must contain Detection values")
    if detection.label not in _SUPPORTED_LABELS:
        raise ValueError(f"unsupported detection label: {detection.label!r}")
    bbox = _validate_bbox(detection.bbox)
    if isinstance(detection.confidence, bool) or not isinstance(
        detection.confidence, (int, float)
    ):
        raise TypeError("detection confidence must be a finite number")
    confidence = float(detection.confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("detection confidence must be between 0 and 1")
    if not isinstance(detection.subtype, str):
        raise TypeError("detection subtype must be a string")
    return replace(detection, bbox=bbox, confidence=confidence)


def _validate_bbox(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    try:
        values = tuple(bbox)
    except TypeError as exc:
        raise TypeError("bbox must contain four finite numbers") from exc
    if len(values) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("bbox must contain four finite numbers")
    result = tuple(float(value) for value in values)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox must have positive width and height")
    return result  # type: ignore[return-value]


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _translate(
    bbox: tuple[float, float, float, float], dx: float, dy: float
) -> tuple[float, float, float, float]:
    return bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_positive(value: object) -> bool:
    return _is_finite_nonnegative(value) and float(value) > 0.0


def _is_unit_interval(value: object) -> bool:
    return _is_finite_nonnegative(value) and float(value) <= 1.0
