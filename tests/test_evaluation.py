import json

import pytest

from gameplay_events.evaluation import (
    box_iou,
    classification_metrics,
    evaluate_detections,
    evaluate_events,
    evaluate_tracking,
)
from gameplay_events.types import Event


def _frame(frame_index, objects, *, prediction=False, segment_id=None):
    frame = {
        "video_id": "clip",
        "frame_index": frame_index,
        "timestamp": frame_index / 15,
        "processed_size": [854, 480],
        "detections" if prediction else "objects": objects,
    }
    if segment_id is not None:
        frame["segment_id"] = segment_id
    return frame


def _object(label, bbox, **extra):
    return {"label": label, "bbox": bbox, **extra}


def test_box_iou_and_invalid_box_validation():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0
    with pytest.raises(ValueError, match="x2 > x1"):
        box_iou([0, 0, 0, 10], [0, 0, 10, 10])


def test_metric_denominators_are_null_only_when_undefined():
    assert classification_metrics(0, 0, 2) == {
        "tp": 0,
        "fp": 0,
        "fn": 2,
        "precision": None,
        "recall": 0,
        "f1": 0,
    }
    assert classification_metrics(0, 0, 0)["f1"] is None


def test_detection_matching_is_class_aware_one_to_one():
    truth = [
        _frame(
            10,
            [
                _object("enemy", [0, 0, 10, 10]),
                _object("coin", [0, 0, 10, 10]),
            ],
        )
    ]
    predictions = [
        _frame(
            10,
            [
                _object("enemy", [0, 0, 10, 10], confidence=0.9),
                _object("enemy", [0, 0, 10, 10], confidence=0.8),
            ],
            prediction=True,
        )
    ]

    result = evaluate_detections(truth, predictions)

    assert result["per_class"]["enemy"]["tp"] == 1
    assert result["per_class"]["enemy"]["fp"] == 1
    assert result["per_class"]["enemy"]["precision"] == 0.5
    assert result["per_class"]["coin"]["fn"] == 1
    assert result["per_class"]["coin"]["precision"] is None
    assert result["per_class"]["coin"]["recall"] == 0


def test_detection_assignment_maximizes_valid_pairs_before_iou():
    truth = [
        _frame(
            0,
            [
                _object("enemy", [0, 0, 10, 10]),
                _object("enemy", [4, 0, 14, 10]),
            ],
        )
    ]
    predictions = [
        _frame(
            0,
            [
                _object("enemy", [2, 0, 12, 10]),
                _object("enemy", [5, 0, 15, 10]),
            ],
            prediction=True,
        )
    ]

    result = evaluate_detections(truth, predictions, labels=("enemy",))

    assert result["micro"]["tp"] == 2
    assert result["micro"]["fp"] == 0
    assert result["micro"]["fn"] == 0


def test_ignored_truth_suppresses_only_overlapping_unmatched_prediction():
    truth = [
        _frame(0, [_object("coin", [0, 0, 10, 10], ignore=True)])
    ]
    predictions = [
        _frame(
            0,
            [
                _object("coin", [0, 0, 10, 10]),
                _object("coin", [20, 20, 30, 30]),
            ],
            prediction=True,
        )
    ]

    result = evaluate_detections(truth, predictions, labels=("coin",))

    assert result["per_class"]["coin"]["ignored_predictions"] == 1
    assert result["per_class"]["coin"]["fp"] == 1


def test_event_assignment_avoids_greedy_cardinality_failure():
    references = [
        {"event": "enemy_detected", "timestamp": 0.0},
        {"event": "enemy_detected", "timestamp": 0.6},
    ]
    predictions = [
        Event(
            event="enemy_detected",
            timestamp=0.4,
            confirmed_at=0.5,
            confidence=0.8,
        ),
        Event(
            event="enemy_detected",
            timestamp=0.9,
            confirmed_at=1.0,
            confidence=0.8,
        ),
    ]

    result = evaluate_events(
        references,
        predictions,
        event_types=("enemy_detected",),
        tolerance_seconds=0.5,
    )

    metric = result["per_event_type"]["enemy_detected"]
    assert metric["tp"] == 2
    assert [(match["reference_index"], match["prediction_index"]) for match in metric["matches"]] == [
        (0, 0),
        (1, 1),
    ]
    assert metric["matches"][0]["signed_error_seconds"] == pytest.approx(0.4)
    assert metric["matches"][0]["confirmation_latency_seconds"] == pytest.approx(0.5)
    assert metric["matches"][0]["detector_confirmation_delay_seconds"] == pytest.approx(0.1)


def test_event_assignment_minimizes_distance_among_maximum_matches():
    references = [
        {"event": "item_detected", "timestamp": 0.0},
        {"event": "item_detected", "timestamp": 1.0},
    ]
    predictions = [
        {"event": "item_detected", "timestamp": 0.1},
        {"event": "item_detected", "timestamp": 0.6},
    ]

    metric = evaluate_events(
        references,
        predictions,
        event_types=("item_detected",),
        tolerance_seconds=1.0,
    )["per_event_type"]["item_detected"]

    assert [(match["reference_index"], match["prediction_index"]) for match in metric["matches"]] == [
        (0, 0),
        (1, 1),
    ]
    assert metric["mean_absolute_error_seconds"] == pytest.approx(0.25)


def test_coin_count_value_agreement_is_reported_and_can_be_required():
    references = [
        {
            "event": "coin_count_changed",
            "timestamp": 2.0,
            "details": {"old_count": 3, "new_count": 4},
        }
    ]
    predictions = [
        {
            "event": "coin_count_changed",
            "timestamp": 2.1,
            "evidence": {"old_count": 3, "new_count": 5},
        }
    ]

    loose = evaluate_events(
        references,
        predictions,
        event_types=("coin_count_changed",),
    )["per_event_type"]["coin_count_changed"]
    strict = evaluate_events(
        references,
        predictions,
        event_types=("coin_count_changed",),
        require_coin_count_value_match=True,
    )["per_event_type"]["coin_count_changed"]

    assert loose["tp"] == 1
    assert loose["coin_count_value_mismatches"] == 1
    assert loose["coin_count_value_accuracy"] == 0
    assert strict["tp"] == 0
    assert strict["fp"] == 1
    assert strict["fn"] == 1


def test_tracking_counts_switches_and_fragments_with_dense_frames():
    truth = []
    predictions = []
    for frame_index in range(5):
        truth.append(
            _frame(
                frame_index,
                [_object("enemy", [0, 0, 10, 10], object_id="enemy-1")],
                segment_id="segment-1",
            )
        )
        predicted_objects = []
        if frame_index in (0, 1):
            predicted_objects = [_object("enemy", [0, 0, 10, 10], track_id=7)]
        elif frame_index in (3, 4):
            predicted_objects = [_object("enemy", [0, 0, 10, 10], track_id=9)]
        predictions.append(
            _frame(
                frame_index,
                predicted_objects,
                prediction=True,
                segment_id="segment-1",
            )
        )

    result = evaluate_tracking(truth, predictions, labels=("enemy",))

    assert result["identity_metrics_available"] is True
    assert result["matched_observations"] == 4
    assert result["observation_recall"] == 0.8
    assert result["identity_switches"] == 1
    assert result["fragments"] == 1


def test_tracking_identity_metrics_are_null_when_ids_are_missing():
    truth = [_frame(0, [_object("enemy", [0, 0, 10, 10])])]
    predictions = [
        _frame(0, [_object("enemy", [0, 0, 10, 10])], prediction=True)
    ]

    result = evaluate_tracking(truth, predictions, labels=("enemy",))

    assert result["identity_metrics_available"] is False
    assert result["identity_switches"] is None
    assert result["fragments"] is None
    json.dumps(result, allow_nan=False)
