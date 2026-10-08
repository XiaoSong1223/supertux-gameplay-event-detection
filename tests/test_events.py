import pytest

from gameplay_events.events import EventEngine
from gameplay_events.types import Track


def track(
    track_id: int,
    timestamp: float,
    *,
    label: str = "enemy",
    observed: bool = True,
    first_seen: float = 0.0,
    last_seen: float | None = None,
    confidence: float = 0.8,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        bbox=(10.0, 10.0, 20.0, 20.0),
        confidence=confidence,
        first_seen=first_seen,
        last_seen=timestamp if last_seen is None else last_seen,
        hits=1,
        confirmed=True,
        observed=observed,
        subtype="snowball" if label == "enemy" else "",
    )


def test_appearance_uses_first_observation_time_and_suppresses_duplicates() -> None:
    engine = EventEngine(appearance_confirmation_hits=2)
    assert engine.update([track(7, 0.0)], 0.0) == []

    events = engine.update([track(7, 0.08)], 0.08)
    assert len(events) == 1
    assert events[0].event == "enemy_detected"
    assert events[0].timestamp == 0.0
    assert events[0].confirmed_at == 0.08
    assert events[0].evidence["confidence_note"] == "uncalibrated heuristic score"

    assert engine.update([track(7, 0.31)], 0.31) == []


def test_coin_emits_item_detected_and_never_disappearance() -> None:
    engine = EventEngine(appearance_confirmation_hits=1, disappearance_grace_seconds=0.1)
    events = engine.update([track(2, 0.0, label="coin")], 0.0)
    assert [event.event for event in events] == ["item_detected"]

    assert engine.update([], 0.2) == []
    assert engine.update([], 0.5) == []


def test_enemy_disappearance_requires_continuous_missing_grace() -> None:
    engine = EventEngine(appearance_confirmation_hits=1, disappearance_grace_seconds=0.3)
    assert [event.event for event in engine.update([track(1, 0.0)], 0.0)] == [
        "enemy_detected"
    ]

    assert engine.update([track(1, 0.1, observed=False, last_seen=0.0)], 0.1) == []
    assert engine.update([track(1, 0.35, observed=False, last_seen=0.0)], 0.35) == []
    events = engine.update([], 0.41)

    assert len(events) == 1
    assert events[0].event == "enemy_disappeared"
    assert events[0].timestamp == 0.1
    assert events[0].confirmed_at == 0.41
    assert events[0].evidence["missing_seconds"] == pytest.approx(0.31)


def test_reappearance_inside_grace_cancels_pending_disappearance() -> None:
    engine = EventEngine(appearance_confirmation_hits=1, disappearance_grace_seconds=0.3)
    engine.update([track(1, 0.0)], 0.0)
    engine.update([track(1, 0.1, observed=False, last_seen=0.0)], 0.1)
    engine.update([track(1, 0.25)], 0.25)

    assert engine.update([track(1, 0.5)], 0.5) == []
    assert engine.update([track(1, 0.6, observed=False, last_seen=0.5)], 0.6) == []
    events = engine.update([], 0.91)
    assert len(events) == 1
    assert events[0].timestamp == 0.6


def test_unconfirmed_observation_streak_resets_after_a_miss() -> None:
    engine = EventEngine(appearance_confirmation_hits=2)
    assert engine.update([track(4, 0.0)], 0.0) == []
    assert engine.update([track(4, 0.1, observed=False, last_seen=0.0)], 0.1) == []
    assert engine.update([track(4, 0.2)], 0.2) == []
    events = engine.update([track(4, 0.3)], 0.3)

    assert len(events) == 1
    assert events[0].timestamp == 0.2


def test_elapsed_time_confirmation_handles_irregular_frame_times() -> None:
    engine = EventEngine(
        appearance_confirmation_hits=None,
        appearance_confirmation_seconds=0.2,
    )
    assert engine.update([track(5, 1.0, first_seen=1.0)], 1.0) == []
    assert engine.update([track(5, 1.07, first_seen=1.0)], 1.07) == []
    events = engine.update([track(5, 1.26, first_seen=1.0)], 1.26)

    assert len(events) == 1
    assert events[0].timestamp == 1.0
    assert events[0].confirmed_at == 1.26


def test_finish_and_reset_do_not_fabricate_disappearance() -> None:
    engine = EventEngine(appearance_confirmation_hits=1, disappearance_grace_seconds=0.0)
    engine.update([track(1, 0.0)], 0.0)
    assert engine.finish() == []

    engine.reset()
    assert engine.update([], 10.0) == []


def test_duplicate_ids_and_time_regression_are_rejected_without_advancing_state() -> None:
    engine = EventEngine(appearance_confirmation_hits=2)
    first = track(1, 1.0, first_seen=1.0)
    engine.update([first], 1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.update([], 1.0)
    with pytest.raises(ValueError, match="duplicate track ID"):
        engine.update([track(1, 1.1), track(1, 1.1)], 1.1)

    events = engine.update([track(1, 1.1, first_seen=1.0)], 1.1)
    assert len(events) == 1
