"""Reproducible metrics for frame detections, visibility events, and tracks.

The functions in this module operate on JSON-shaped dictionaries and return only
JSON-serializable values.  Matching is class/type aware and one-to-one.  A large
penalty on invalid edges makes assignment lexicographic: first maximize the
number of valid matches, then minimize total localization or timing error.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from math import isfinite
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


DETECTION_LABELS = ("enemy", "coin")
EVENT_TYPES = (
    "enemy_detected",
    "enemy_disappeared",
    "item_detected",
    "coin_count_changed",
)


def box_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return IoU for two ``[x1, y1, x2, y2]`` boxes.

    Coordinates are continuous processed-frame pixels.  ``x2`` and ``y2`` are
    the right and bottom edges, so width and height are simple differences.
    """

    a = _validated_box(first, "first box")
    b = _validated_box(second, "second box")
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (
        b[3] - b[1]
    ) - intersection
    return intersection / union


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Compute count-based precision, recall, and F1.

    A metric is ``None`` exactly when its own denominator is zero.  Thus a class
    with references but no predictions has recall and F1 of 0, while precision
    is undefined rather than a misleading 0.
    """

    if min(tp, fp, fn) < 0:
        raise ValueError("tp, fp, and fn must be non-negative")
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / precision_denominator if precision_denominator else None,
        "recall": tp / recall_denominator if recall_denominator else None,
        "f1": 2 * tp / f1_denominator if f1_denominator else None,
    }


def evaluate_detections(
    ground_truth_frames: Sequence[Mapping[str, Any]],
    prediction_frames: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] = DETECTION_LABELS,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate class-aware detections at fixed annotated frames.

    Ground-truth frame objects live under ``objects``.  Prediction items should
    live under ``detections`` (``objects`` and ``tracks`` are accepted for
    interoperability).  A ground-truth object with ``ignore: true`` is excluded;
    an otherwise-unmatched prediction overlapping such an object at the IoU
    threshold is also excluded from false positives.
    """

    label_order = _validated_names(labels, "labels")
    _validate_iou_threshold(iou_threshold)
    truth_by_frame = _index_frames(ground_truth_frames)
    prediction_by_frame = _index_frames(prediction_frames)
    frame_keys = sorted(set(truth_by_frame) | set(prediction_by_frame))
    accumulators: dict[str, dict[str, Any]] = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ground_truth_objects": 0,
            "predictions": 0,
            "ignored_predictions": 0,
            "matches": [],
        }
        for label in label_order
    }

    for frame_key in frame_keys:
        truth_frame = truth_by_frame.get(frame_key)
        prediction_frame = prediction_by_frame.get(frame_key)
        truth_items = _frame_items(truth_frame, prediction=False)
        prediction_items = _frame_items(prediction_frame, prediction=True)
        _validate_item_labels(truth_items, label_order, "ground truth")
        _validate_item_labels(prediction_items, label_order, "prediction")

        for label in label_order:
            truth_for_label = [item for item in truth_items if item["label"] == label]
            active_truth = [item for item in truth_for_label if not item.get("ignore", False)]
            ignored_truth = [item for item in truth_for_label if item.get("ignore", False)]
            predictions = [item for item in prediction_items if item["label"] == label]
            pairs = _match_boxes(active_truth, predictions, iou_threshold)
            matched_truth = {truth_index for truth_index, _, _ in pairs}
            matched_predictions = {prediction_index for _, prediction_index, _ in pairs}
            ignored_prediction_indices = {
                prediction_index
                for prediction_index, prediction in enumerate(predictions)
                if prediction_index not in matched_predictions
                and any(
                    box_iou(prediction["bbox"], ignored["bbox"]) >= iou_threshold
                    for ignored in ignored_truth
                )
            }

            counts = accumulators[label]
            counts["tp"] += len(pairs)
            counts["fn"] += len(active_truth) - len(matched_truth)
            counts["fp"] += (
                len(predictions) - len(matched_predictions) - len(ignored_prediction_indices)
            )
            counts["ground_truth_objects"] += len(active_truth)
            counts["predictions"] += len(predictions) - len(ignored_prediction_indices)
            counts["ignored_predictions"] += len(ignored_prediction_indices)
            for truth_index, prediction_index, iou in pairs:
                counts["matches"].append(
                    {
                        "video_id": frame_key[0],
                        "frame_index": frame_key[1],
                        "ground_truth_index": truth_index,
                        "prediction_index": prediction_index,
                        "iou": iou,
                    }
                )

    per_class: dict[str, Any] = {}
    for label, counts in accumulators.items():
        metrics = classification_metrics(counts["tp"], counts["fp"], counts["fn"])
        per_class[label] = {
            **metrics,
            "ground_truth_objects": counts["ground_truth_objects"],
            "predictions": counts["predictions"],
            "ignored_predictions": counts["ignored_predictions"],
            "matches": counts["matches"],
        }

    total_tp = sum(values["tp"] for values in accumulators.values())
    total_fp = sum(values["fp"] for values in accumulators.values())
    total_fn = sum(values["fn"] for values in accumulators.values())
    return {
        "iou_threshold": iou_threshold,
        "annotated_frame_count": len(truth_by_frame),
        "prediction_frame_count": len(prediction_by_frame),
        "per_class": per_class,
        "micro": classification_metrics(total_tp, total_fp, total_fn),
    }


def evaluate_events(
    reference_events: Sequence[Mapping[str, Any] | Any] | Mapping[str, Any],
    predicted_events: Sequence[Mapping[str, Any] | Any] | Mapping[str, Any],
    *,
    event_types: Sequence[str] = EVENT_TYPES,
    tolerance_seconds: float = 0.5,
    require_coin_count_value_match: bool = False,
) -> dict[str, Any]:
    """Match same-type events one-to-one within a temporal tolerance.

    Among assignments with the maximum number of valid pairs, the result has
    minimum total absolute timing error.  Signed timing error is prediction
    timestamp minus reference timestamp.  Confirmation latency is predicted
    ``confirmed_at`` minus reference timestamp and is ``None`` when unavailable.

    With ``require_coin_count_value_match=True``, a ``coin_count_changed`` pair
    is eligible only when both old/new values are present and equal.  In either
    mode, value agreement is reported separately for matched coin-count events.
    """

    type_order = _validated_names(event_types, "event_types")
    if not isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    references = _event_items(reference_events)
    predictions = _event_items(predicted_events)
    _validate_event_records(references, type_order, "reference")
    _validate_event_records(predictions, type_order, "prediction")
    per_type: dict[str, Any] = {}
    all_matches: list[dict[str, Any]] = []

    for event_type in type_order:
        typed_references = [
            (index, event)
            for index, event in enumerate(references)
            if event["event"] == event_type and not event.get("ignore", False)
        ]
        typed_predictions = [
            (index, event)
            for index, event in enumerate(predictions)
            if event["event"] == event_type
        ]
        costs: list[list[float]] = []
        validity: list[list[bool]] = []
        for _, reference in typed_references:
            cost_row: list[float] = []
            valid_row: list[bool] = []
            for _, prediction in typed_predictions:
                distance = abs(prediction["timestamp"] - reference["timestamp"])
                value_compatible = (
                    event_type != "coin_count_changed"
                    or not require_coin_count_value_match
                    or _coin_count_values(reference) == _coin_count_values(prediction)
                    and _coin_count_values(reference) is not None
                )
                cost_row.append(distance)
                valid_row.append(distance <= tolerance_seconds and value_compatible)
            costs.append(cost_row)
            validity.append(valid_row)

        pairs = _lexicographic_assignment(costs, validity)
        match_details: list[dict[str, Any]] = []
        for reference_local_index, prediction_local_index, _ in pairs:
            reference_index, reference = typed_references[reference_local_index]
            prediction_index, prediction = typed_predictions[prediction_local_index]
            signed_error = prediction["timestamp"] - reference["timestamp"]
            confirmed_at = prediction.get("confirmed_at")
            value_status = _coin_value_status(reference, prediction, event_type)
            detail = {
                "reference_index": reference_index,
                "prediction_index": prediction_index,
                "signed_error_seconds": signed_error,
                "absolute_error_seconds": abs(signed_error),
                "confirmation_latency_seconds": (
                    confirmed_at - reference["timestamp"] if confirmed_at is not None else None
                ),
                "detector_confirmation_delay_seconds": (
                    confirmed_at - prediction["timestamp"] if confirmed_at is not None else None
                ),
                "coin_count_value_match": value_status,
            }
            match_details.append(detail)
            all_matches.append({"event": event_type, **detail})

        metric_counts = classification_metrics(
            len(pairs),
            len(typed_predictions) - len(pairs),
            len(typed_references) - len(pairs),
        )
        signed_errors = [item["signed_error_seconds"] for item in match_details]
        absolute_errors = [item["absolute_error_seconds"] for item in match_details]
        confirmation_latencies = [
            item["confirmation_latency_seconds"]
            for item in match_details
            if item["confirmation_latency_seconds"] is not None
        ]
        comparable_values = [
            item["coin_count_value_match"]
            for item in match_details
            if item["coin_count_value_match"] is not None
        ]
        per_type[event_type] = {
            **metric_counts,
            "reference_count": len(typed_references),
            "prediction_count": len(typed_predictions),
            "mean_signed_error_seconds": _mean_or_none(signed_errors),
            "mean_absolute_error_seconds": _mean_or_none(absolute_errors),
            "median_absolute_error_seconds": (
                median(absolute_errors) if absolute_errors else None
            ),
            "mean_confirmation_latency_seconds": _mean_or_none(confirmation_latencies),
            "coin_count_value_comparable": len(comparable_values),
            "coin_count_value_matches": sum(comparable_values),
            "coin_count_value_mismatches": len(comparable_values) - sum(comparable_values),
            "coin_count_value_accuracy": (
                sum(comparable_values) / len(comparable_values)
                if comparable_values
                else None
            ),
            "matches": match_details,
        }

    total_tp = sum(item["tp"] for item in per_type.values())
    total_fp = sum(item["fp"] for item in per_type.values())
    total_fn = sum(item["fn"] for item in per_type.values())
    all_signed_errors = [item["signed_error_seconds"] for item in all_matches]
    all_absolute_errors = [item["absolute_error_seconds"] for item in all_matches]
    all_confirmation_latencies = [
        item["confirmation_latency_seconds"]
        for item in all_matches
        if item["confirmation_latency_seconds"] is not None
    ]
    return {
        "tolerance_seconds": tolerance_seconds,
        "require_coin_count_value_match": require_coin_count_value_match,
        "per_event_type": per_type,
        "micro": {
            **classification_metrics(total_tp, total_fp, total_fn),
            "mean_signed_error_seconds": _mean_or_none(all_signed_errors),
            "mean_absolute_error_seconds": _mean_or_none(all_absolute_errors),
            "mean_confirmation_latency_seconds": _mean_or_none(
                all_confirmation_latencies
            ),
        },
    }


def evaluate_tracking(
    ground_truth_frames: Sequence[Mapping[str, Any]],
    prediction_frames: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] = DETECTION_LABELS,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Count identity switches and fragments on densely annotated segments.

    Frame associations use the same class-aware IoU assignment as detection
    evaluation.  For each ground-truth object, an identity switch is a change in
    predicted ``track_id`` between consecutive *matched* observations, including
    across an intervening miss.  A fragment is an additional matched run after a
    miss.  Identity metrics are ``None`` unless every scored truth observation
    has ``object_id`` and every matched prediction has ``track_id``.
    """

    label_order = _validated_names(labels, "labels")
    _validate_iou_threshold(iou_threshold)
    truth_by_frame = _index_frames(ground_truth_frames)
    prediction_by_frame = _index_frames(prediction_frames)
    observations: dict[tuple[str, str, str, str], list[tuple[int, Any]]] = {}
    total_truth_observations = 0
    matched_observations = 0
    missing_truth_ids = 0
    missing_prediction_ids = 0

    for frame_key, truth_frame in sorted(truth_by_frame.items()):
        prediction_frame = prediction_by_frame.get(frame_key)
        truth_items = [
            item
            for item in _frame_items(truth_frame, prediction=False)
            if not item.get("ignore", False)
        ]
        prediction_items = _frame_items(prediction_frame, prediction=True)
        _validate_item_labels(truth_items, label_order, "ground truth")
        _validate_item_labels(prediction_items, label_order, "prediction")
        segment_id = str(truth_frame.get("segment_id", ""))

        for label in label_order:
            typed_truth = [item for item in truth_items if item["label"] == label]
            typed_predictions = [
                item for item in prediction_items if item["label"] == label
            ]
            pairs = _match_boxes(typed_truth, typed_predictions, iou_threshold)
            prediction_for_truth = {
                truth_index: typed_predictions[prediction_index]
                for truth_index, prediction_index, _ in pairs
            }
            total_truth_observations += len(typed_truth)
            matched_observations += len(pairs)
            for truth_index, truth in enumerate(typed_truth):
                object_id = truth.get("object_id")
                if object_id is None:
                    missing_truth_ids += 1
                    object_key = f"__missing_{frame_key[1]}_{truth_index}"
                else:
                    object_key = str(object_id)
                prediction = prediction_for_truth.get(truth_index)
                track_id = prediction.get("track_id") if prediction is not None else None
                if prediction is not None and track_id is None:
                    missing_prediction_ids += 1
                key = (frame_key[0], segment_id, label, object_key)
                observations.setdefault(key, []).append((frame_key[1], track_id))

    identity_available = missing_truth_ids == 0 and missing_prediction_ids == 0
    switch_count: int | None = 0 if identity_available else None
    fragment_count: int | None = 0 if identity_available else None
    track_details: list[dict[str, Any]] = []
    if identity_available:
        for (video_id, segment_id, label, object_id), values in sorted(
            observations.items()
        ):
            sequence = [track_id for _, track_id in sorted(values)]
            matched_ids = [track_id for track_id in sequence if track_id is not None]
            switches = sum(
                current != previous
                for previous, current in zip(matched_ids, matched_ids[1:])
            )
            matched_runs = 0
            previous_matched = False
            for track_id in sequence:
                current_matched = track_id is not None
                if current_matched and not previous_matched:
                    matched_runs += 1
                previous_matched = current_matched
            fragments = max(0, matched_runs - 1)
            switch_count += switches
            fragment_count += fragments
            track_details.append(
                {
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "label": label,
                    "object_id": object_id,
                    "ground_truth_observations": len(sequence),
                    "matched_observations": len(matched_ids),
                    "identity_switches": switches,
                    "fragments": fragments,
                }
            )

    return {
        "iou_threshold": iou_threshold,
        "annotated_frame_count": len(truth_by_frame),
        "ground_truth_track_count": len(observations) if identity_available else None,
        "ground_truth_observations": total_truth_observations,
        "matched_observations": matched_observations,
        "observation_recall": (
            matched_observations / total_truth_observations
            if total_truth_observations
            else None
        ),
        "identity_metrics_available": identity_available,
        "identity_switches": switch_count,
        "fragments": fragment_count,
        "missing_ground_truth_object_ids": missing_truth_ids,
        "matched_predictions_missing_track_id": missing_prediction_ids,
        "tracks": track_details,
        "caveat": (
            "Identity counts describe only the densely annotated segments and "
            "visibility-episode IDs; they do not establish persistent game-world identity."
        ),
    }


def _validated_box(box: Sequence[float], context: str) -> tuple[float, float, float, float]:
    if isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError(f"{context} must contain four coordinates")
    coordinates = tuple(float(value) for value in box)
    if not all(isfinite(value) for value in coordinates):
        raise ValueError(f"{context} coordinates must be finite")
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError(f"{context} must have x2 > x1 and y2 > y1")
    return coordinates


def _validated_names(names: Sequence[str], context: str) -> tuple[str, ...]:
    values = tuple(names)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{context} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{context} must not contain duplicates")
    return values


def _validate_iou_threshold(threshold: float) -> None:
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")


def _index_frames(
    frames: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise TypeError("frame records must be mappings")
        if "frame_index" not in frame or isinstance(frame["frame_index"], bool):
            raise ValueError("each frame requires an integer frame_index")
        frame_index = frame["frame_index"]
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        key = (str(frame.get("video_id", "")), frame_index)
        if key in indexed:
            raise ValueError(f"duplicate frame key: {key}")
        indexed[key] = frame
    return indexed


def _frame_items(
    frame: Mapping[str, Any] | None, *, prediction: bool
) -> list[Mapping[str, Any]]:
    if frame is None:
        return []
    if prediction:
        if "detections" in frame:
            items = frame["detections"]
        elif "tracks" in frame:
            items = frame["tracks"]
        else:
            items = frame.get("objects", [])
    else:
        items = frame.get("objects", [])
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("frame objects/detections must be a sequence")
    normalized: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("frame objects/detections must be mappings")
        if "label" not in item or "bbox" not in item:
            raise ValueError("each object/detection requires label and bbox")
        _validated_box(item["bbox"], "object/detection bbox")
        normalized.append(item)
    return normalized


def _validate_item_labels(
    items: Sequence[Mapping[str, Any]], labels: Sequence[str], context: str
) -> None:
    allowed = set(labels)
    unknown = sorted({item["label"] for item in items if item["label"] not in allowed})
    if unknown:
        raise ValueError(f"unknown {context} labels: {unknown}")


def _match_boxes(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    ious = [
        [box_iou(reference["bbox"], prediction["bbox"]) for prediction in predictions]
        for reference in truth
    ]
    valid = [[value >= threshold for value in row] for row in ious]
    costs = [[1.0 - value for value in row] for row in ious]
    return [
        (truth_index, prediction_index, ious[truth_index][prediction_index])
        for truth_index, prediction_index, _ in _lexicographic_assignment(costs, valid)
    ]


def _lexicographic_assignment(
    costs: Sequence[Sequence[float]], validity: Sequence[Sequence[bool]]
) -> list[tuple[int, int, float]]:
    row_count = len(costs)
    if row_count == 0:
        return []
    column_count = len(costs[0])
    if column_count == 0:
        return []
    if len(validity) != row_count or any(len(row) != column_count for row in validity):
        raise ValueError("cost and validity matrices must have equal rectangular shape")
    if any(len(row) != column_count for row in costs):
        raise ValueError("cost matrix must be rectangular")
    valid_costs = [
        float(costs[row][column])
        for row in range(row_count)
        for column in range(column_count)
        if validity[row][column]
    ]
    if not valid_costs:
        return []
    if any(not isfinite(cost) or cost < 0 for cost in valid_costs):
        raise ValueError("valid assignment costs must be finite and non-negative")
    assignment_size = min(row_count, column_count)
    maximum_valid_cost = max(valid_costs)
    invalid_cost = (assignment_size + 1) * (maximum_valid_cost + 1.0)
    matrix = np.full((row_count, column_count), invalid_cost, dtype=float)
    for row in range(row_count):
        for column in range(column_count):
            if validity[row][column]:
                matrix[row, column] = costs[row][column]
    assigned_rows, assigned_columns = linear_sum_assignment(matrix)
    matches = [
        (int(row), int(column), float(costs[row][column]))
        for row, column in zip(assigned_rows, assigned_columns)
        if validity[row][column]
    ]
    return sorted(matches)


def _event_items(
    records: Sequence[Mapping[str, Any] | Any] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(records, Mapping):
        values = records.get("events")
        if values is None:
            raise ValueError("event document mapping requires an events array")
    else:
        values = records
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("events must be a sequence")
    normalized: list[dict[str, Any]] = []
    for value in values:
        if is_dataclass(value):
            record = asdict(value)
        elif isinstance(value, Mapping):
            record = dict(value)
        else:
            raise TypeError("events must be mappings or dataclass instances")
        normalized.append(record)
    return normalized


def _validate_event_records(
    records: Sequence[Mapping[str, Any]], event_types: Sequence[str], context: str
) -> None:
    allowed = set(event_types)
    for event in records:
        event_type = event.get("event")
        if event_type not in allowed:
            raise ValueError(f"unknown {context} event type: {event_type!r}")
        timestamp = event.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f"{context} event timestamp must be numeric")
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"{context} event timestamp must be finite and non-negative")
        confirmed_at = event.get("confirmed_at")
        if confirmed_at is not None and (
            isinstance(confirmed_at, bool)
            or not isinstance(confirmed_at, (int, float))
            or not isfinite(confirmed_at)
        ):
            raise ValueError(f"{context} event confirmed_at must be finite or null")


def _coin_count_values(event: Mapping[str, Any]) -> tuple[Any, Any] | None:
    details = event.get("details")
    evidence = event.get("evidence")
    sources = [event]
    if isinstance(details, Mapping):
        sources.insert(0, details)
    if isinstance(evidence, Mapping):
        sources.insert(0, evidence)
    for source in sources:
        if "old_count" in source and "new_count" in source:
            return source["old_count"], source["new_count"]
    return None


def _coin_value_status(
    reference: Mapping[str, Any], prediction: Mapping[str, Any], event_type: str
) -> bool | None:
    if event_type != "coin_count_changed":
        return None
    reference_values = _coin_count_values(reference)
    prediction_values = _coin_count_values(prediction)
    if reference_values is None or prediction_values is None:
        return None
    return reference_values == prediction_values


def _mean_or_none(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None
