from dataclasses import replace

import pytest

from gameplay_events.tracking import Tracker
from gameplay_events.types import Detection


def detection(
    x: float,
    *,
    label: str = "enemy",
    confidence: float = 0.9,
    subtype: str = "",
) -> Detection:
    return Detection(label, (x, 10.0, x + 10.0, 20.0), confidence, subtype)


def by_id(tracks):
    return {track.track_id: track for track in tracks}


def test_class_aware_global_assignment_preserves_two_competing_tracks() -> None:
    tracker = Tracker(
        confirmation_hits=1,
        max_center_distance=50.0,
        min_iou=0.0,
        distance_weight=1.0,
        velocity_smoothing=1.0,
    )
    initial = tracker.update([detection(0), detection(5)], 0.0)
    left_id, right_id = [track.track_id for track in initial]

    # Both tracks locally prefer x=4. A track-ordered greedy match would give it
    # to the left track and force the right track to x=-5 (total distance 14).
    # The global assignment selects the lower total distance of 6.
    tracks = by_id(tracker.update([detection(4), detection(-5)], 0.1))

    assert tracks[left_id].bbox[0] == -5.0
    assert tracks[right_id].bbox[0] == 4.0
    assert tracks[left_id].observed and tracks[right_id].observed
    assert len(tracks) == 2


def test_association_is_class_aware_even_for_identical_boxes() -> None:
    tracker = Tracker(confirmation_hits=1)
    first = tracker.update([detection(5, label="enemy"), detection(5, label="coin")], 0.0)
    ids = {track.label: track.track_id for track in first}

    second = by_id(
        tracker.update([detection(6, label="coin"), detection(6, label="enemy")], 0.1)
    )

    assert second[ids["enemy"]].label == "enemy"
    assert second[ids["enemy"]].bbox[0] == 6.0
    assert second[ids["coin"]].label == "coin"
    assert second[ids["coin"]].bbox[0] == 6.0


def test_short_miss_returns_predicted_track_then_reuses_identity() -> None:
    tracker = Tracker(
        max_age_seconds=0.4,
        confirmation_hits=2,
        velocity_smoothing=1.0,
        distance_weight=1.0,
    )
    track_id = tracker.update([detection(0)], 0.0)[0].track_id
    tracker.update([detection(10)], 0.1)

    missed = tracker.update([], 0.2)
    assert len(missed) == 1
    assert missed[0].track_id == track_id
    assert missed[0].bbox[0] == pytest.approx(20.0)
    assert missed[0].observed is False
    assert missed[0].last_seen == 0.1

    reacquired = tracker.update([detection(30)], 0.3)
    assert reacquired[0].track_id == track_id
    assert reacquired[0].observed is True
    assert reacquired[0].hits == 3


def test_camera_shift_is_not_learned_as_velocity() -> None:
    tracker = Tracker(
        max_age_seconds=1.0,
        confirmation_hits=1,
        velocity_smoothing=1.0,
        distance_weight=1.0,
    )
    track_id = tracker.update([detection(100)], 0.0)[0].track_id
    tracker.update([detection(90)], 0.1, camera_shift=(-10.0, 0.0))

    # With zero residual object motion, the missed-frame prediction applies the
    # new camera shift once.  Learning the first shift as velocity gives x=70.
    predicted = tracker.update([], 0.2, camera_shift=(-10.0, 0.0))
    assert predicted[0].track_id == track_id
    assert predicted[0].bbox[0] == pytest.approx(80.0)


def test_expiry_creates_a_new_monotonic_id_on_later_detection() -> None:
    tracker = Tracker(max_age_seconds=0.2, confirmation_hits=1)
    first_id = tracker.update([detection(0)], 0.0)[0].track_id
    # The next update itself is beyond max age; the stale track must not be
    # revived by the spatially identical detection.
    second_id = tracker.update([detection(0)], 0.21)[0].track_id

    tracker.reset()
    third_id = tracker.update([detection(0)], 1.0)[0].track_id
    assert first_id < second_id < third_id


def test_zero_iou_gate_still_enforces_center_distance() -> None:
    tracker = Tracker(
        max_age_seconds=1.0,
        confirmation_hits=1,
        max_center_distance=20.0,
        min_iou=0.0,
    )
    first_id = tracker.update([detection(0)], 0.0)[0].track_id
    tracks = tracker.update([detection(200)], 0.1)

    assert len(tracks) == 2
    assert by_id(tracks)[first_id].observed is False


def test_irregular_timestamps_drive_motion_in_seconds() -> None:
    tracker = Tracker(
        max_age_seconds=1.0,
        confirmation_hits=1,
        velocity_smoothing=1.0,
        distance_weight=1.0,
    )
    tracker.update([detection(0)], 0.0)
    tracker.update([detection(20)], 0.2)
    predicted = tracker.update([], 0.65)

    assert predicted[0].bbox[0] == pytest.approx(65.0)


def test_invalid_update_does_not_advance_tracker_state() -> None:
    tracker = Tracker(confirmation_hits=1)
    first = tracker.update([detection(0)], 1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update([detection(10)], 1.0)
    with pytest.raises(ValueError, match="positive width"):
        tracker.update([replace(detection(10), bbox=(10, 10, 10, 20))], 1.1)

    recovered = tracker.update([detection(10)], 1.1)
    assert recovered[0].track_id == first[0].track_id
    assert recovered[0].hits == 2
