"""Temporal gameplay event confirmation built on tracker output."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .types import Event, Track


_CONFIDENCE_NOTE = "uncalibrated heuristic score"


@dataclass
class _EventState:
    label: str
    subtype: str
    first_observed: float
    last_seen: float
    last_bbox: tuple[float, float, float, float]
    last_confidence: float
    consecutive_observations: int = 1
    appeared: bool = False
    missing_since: float | None = None
    disappeared: bool = False


class EventEngine:
    """Confirm appearances and delayed enemy disappearances.

    Appearance confirmation succeeds when either configured threshold is met.
    Set one threshold to ``None`` to use only the other.  Confidence values are
    evidence-ranking heuristics copied from the latest supporting track; they
    are not calibrated probabilities.
    """

    def __init__(
        self,
        appearance_confirmation_hits: int | None = 2,
        appearance_confirmation_seconds: float | None = None,
        disappearance_grace_seconds: float = 0.35,
    ) -> None:
        if appearance_confirmation_hits is not None and (
            isinstance(appearance_confirmation_hits, bool)
            or not isinstance(appearance_confirmation_hits, int)
            or appearance_confirmation_hits < 1
        ):
            raise ValueError("appearance_confirmation_hits must be at least 1 or None")
        if appearance_confirmation_seconds is not None and not _finite_nonnegative(
            appearance_confirmation_seconds
        ):
            raise ValueError(
                "appearance_confirmation_seconds must be finite, non-negative, or None"
            )
        if appearance_confirmation_hits is None and appearance_confirmation_seconds is None:
            raise ValueError("at least one appearance confirmation threshold is required")
        if not _finite_nonnegative(disappearance_grace_seconds):
            raise ValueError("disappearance_grace_seconds must be finite and non-negative")

        self.appearance_confirmation_hits = appearance_confirmation_hits
        self.appearance_confirmation_seconds = (
            None
            if appearance_confirmation_seconds is None
            else float(appearance_confirmation_seconds)
        )
        self.disappearance_grace_seconds = float(disappearance_grace_seconds)
        self._states: dict[int, _EventState] = {}
        self._last_timestamp: float | None = None

    def update(self, tracks: list[Track], timestamp: float) -> list[Event]:
        """Consume a tracker update and return newly confirmed events only."""

        timestamp = _validate_timestamp(timestamp, self._last_timestamp)
        by_id = _validate_tracks(tracks, timestamp)
        events: list[Event] = []

        for track_id, track in by_id.items():
            state = self._states.get(track_id)
            if state is None:
                # first_seen is the first supporting time even when an engine is
                # attached after the tracker has already accumulated hits.
                state = _EventState(
                    label=track.label,
                    subtype=track.subtype,
                    first_observed=track.first_seen,
                    last_seen=track.last_seen,
                    last_bbox=track.bbox,
                    last_confidence=track.confidence,
                    consecutive_observations=1 if track.observed else 0,
                )
                self._states[track_id] = state
            elif state.label != track.label:
                raise ValueError(f"track {track_id} changed label")

            if track.observed:
                if state.consecutive_observations == 0:
                    state.first_observed = timestamp
                    state.consecutive_observations = 1
                elif track.last_seen > state.last_seen:
                    state.consecutive_observations += 1
                state.missing_since = None
                state.last_seen = track.last_seen
                state.last_bbox = track.bbox
                state.last_confidence = track.confidence
                if track.subtype:
                    state.subtype = track.subtype

                if not state.appeared and self._appearance_confirmed(state, timestamp):
                    state.appeared = True
                    event_name = "enemy_detected" if state.label == "enemy" else "item_detected"
                    events.append(
                        Event(
                            timestamp=state.first_observed,
                            event=event_name,
                            confidence=state.last_confidence,
                            confirmed_at=timestamp,
                            track_id=track_id,
                            evidence={
                                "label": state.label,
                                "subtype": state.subtype,
                                "bbox": list(state.last_bbox),
                                "consecutive_observations": state.consecutive_observations,
                                "visible_seconds": timestamp - state.first_observed,
                                "confidence_note": _CONFIDENCE_NOTE,
                            },
                        )
                    )
            else:
                state.consecutive_observations = 0
                if state.appeared and state.label == "enemy" and not state.disappeared:
                    if state.missing_since is None:
                        state.missing_since = timestamp

        # An expired track is absent rather than present with observed=False.
        # Preserve an earlier missing timestamp, or start one at this update.
        for track_id, state in self._states.items():
            if (
                track_id not in by_id
                and state.appeared
                and state.label == "enemy"
                and not state.disappeared
            ):
                if state.missing_since is None:
                    state.missing_since = timestamp
                    state.consecutive_observations = 0

            if (
                state.appeared
                and state.label == "enemy"
                and not state.disappeared
                and state.missing_since is not None
                and timestamp - state.missing_since >= self.disappearance_grace_seconds
            ):
                state.disappeared = True
                events.append(
                    Event(
                        timestamp=state.missing_since,
                        event="enemy_disappeared",
                        confidence=state.last_confidence,
                        confirmed_at=timestamp,
                        track_id=track_id,
                        evidence={
                            "label": state.label,
                            "subtype": state.subtype,
                            "last_bbox": list(state.last_bbox),
                            "last_seen": state.last_seen,
                            "missing_seconds": timestamp - state.missing_since,
                            "grace_seconds": self.disappearance_grace_seconds,
                            "confidence_note": _CONFIDENCE_NOTE,
                        },
                    )
                )

        self._last_timestamp = timestamp
        return sorted(
            events,
            key=lambda event: (event.confirmed_at, event.track_id or -1, event.event),
        )

    def finish(self) -> list[Event]:
        """End a stream without interpreting EOF as evidence of disappearance."""

        return []

    def reset(self) -> None:
        """Clear temporal evidence at a scene cut without emitting events."""

        self._states.clear()
        self._last_timestamp = None

    def _appearance_confirmed(self, state: _EventState, timestamp: float) -> bool:
        by_hits = (
            self.appearance_confirmation_hits is not None
            and state.consecutive_observations >= self.appearance_confirmation_hits
        )
        by_time = (
            self.appearance_confirmation_seconds is not None
            and timestamp - state.first_observed >= self.appearance_confirmation_seconds
        )
        return by_hits or by_time


def _validate_timestamp(timestamp: float, previous: float | None) -> float:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise TypeError("timestamp must be a finite number")
    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and non-negative")
    if previous is not None and timestamp <= previous:
        raise ValueError("timestamps must be strictly increasing")
    return timestamp


def _validate_tracks(tracks: list[Track], timestamp: float) -> dict[int, Track]:
    by_id: dict[int, Track] = {}
    for track in tracks:
        if not isinstance(track, Track):
            raise TypeError("tracks must contain Track values")
        if (
            isinstance(track.track_id, bool)
            or not isinstance(track.track_id, int)
            or track.track_id < 1
        ):
            raise ValueError("track IDs must be positive integers")
        if track.track_id in by_id:
            raise ValueError(f"duplicate track ID: {track.track_id}")
        if track.label not in {"enemy", "coin"}:
            raise ValueError(f"unsupported track label: {track.label!r}")
        if (
            not _finite_nonnegative(track.first_seen)
            or not _finite_nonnegative(track.last_seen)
            or track.first_seen > track.last_seen
            or track.last_seen > timestamp
        ):
            raise ValueError(f"track {track.track_id} has invalid observation times")
        if (
            not _finite_nonnegative(track.confidence)
            or float(track.confidence) > 1.0
        ):
            raise ValueError(f"track {track.track_id} has invalid confidence")
        if (
            len(track.bbox) != 4
            or any(not _finite_number(value) for value in track.bbox)
            or track.bbox[2] <= track.bbox[0]
            or track.bbox[3] <= track.bbox[1]
        ):
            raise ValueError(f"track {track.track_id} has invalid bbox")
        by_id[track.track_id] = track
    return by_id


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
