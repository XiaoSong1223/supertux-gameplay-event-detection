# Ground-truth annotation and evaluation protocol

## Purpose and separation from development

This protocol defines the held-out evaluation before any predictions are viewed. The evaluation video is never used to choose templates, thresholds, supported appearances, temporal settings, or failure-specific fixes. Ground truth is drawn manually from the unmodified video; detector and tracker output must not be displayed during annotation. Ground-truth overlays may be rendered afterward only to check the annotations themselves.

The scored playable interval is 2.0 through 92.0 seconds. Menus, fades, loading screens, and material outside this interval are out of scope. Detector configuration is frozen before decoding the evaluation clip for inference. If held-out results lead to any code or parameter change, the original result, the change, and the rerun result must all be reported; the rerun is no longer an untouched held-out result.

All coordinates refer to the 854 by 480 processed image. Boxes use `[x1, y1, x2, y2]` with the right and bottom edges exclusive. They must have positive area and stay within the processed image. Timestamps are seconds from the start of the source video.

## Detection frame selection

The 40 detection frames are fixed independently of image contents and predictions. Target times are `2 + i * (92 - 2) / 39` seconds for `i = 0..39`; each target is represented by the nearest frame in the 60 fps source. This schedule is frozen even when a sampled frame has no eligible object. It spans the declared playable interval and avoids the transitions outside it.

| Sample | Target (s) | Source frame | Actual time (s) |
|---:|---:|---:|---:|
| 1 | 2.000000 | 120 | 2.000000 |
| 2 | 4.307692 | 258 | 4.300000 |
| 3 | 6.615385 | 397 | 6.616667 |
| 4 | 8.923077 | 535 | 8.916667 |
| 5 | 11.230769 | 674 | 11.233333 |
| 6 | 13.538462 | 812 | 13.533333 |
| 7 | 15.846154 | 951 | 15.850000 |
| 8 | 18.153846 | 1089 | 18.150000 |
| 9 | 20.461538 | 1228 | 20.466667 |
| 10 | 22.769231 | 1366 | 22.766667 |
| 11 | 25.076923 | 1505 | 25.083333 |
| 12 | 27.384615 | 1643 | 27.383333 |
| 13 | 29.692308 | 1782 | 29.700000 |
| 14 | 32.000000 | 1920 | 32.000000 |
| 15 | 34.307692 | 2058 | 34.300000 |
| 16 | 36.615385 | 2197 | 36.616667 |
| 17 | 38.923077 | 2335 | 38.916667 |
| 18 | 41.230769 | 2474 | 41.233333 |
| 19 | 43.538462 | 2612 | 43.533333 |
| 20 | 45.846154 | 2751 | 45.850000 |
| 21 | 48.153846 | 2889 | 48.150000 |
| 22 | 50.461538 | 3028 | 50.466667 |
| 23 | 52.769231 | 3166 | 52.766667 |
| 24 | 55.076923 | 3305 | 55.083333 |
| 25 | 57.384615 | 3443 | 57.383333 |
| 26 | 59.692308 | 3582 | 59.700000 |
| 27 | 62.000000 | 3720 | 62.000000 |
| 28 | 64.307692 | 3858 | 64.300000 |
| 29 | 66.615385 | 3997 | 66.616667 |
| 30 | 68.923077 | 4135 | 68.916667 |
| 31 | 71.230769 | 4274 | 71.233333 |
| 32 | 73.538462 | 4412 | 73.533333 |
| 33 | 75.846154 | 4551 | 75.850000 |
| 34 | 78.153846 | 4689 | 78.150000 |
| 35 | 80.461538 | 4828 | 80.466667 |
| 36 | 82.769231 | 4966 | 82.766667 |
| 37 | 85.076923 | 5105 | 85.083333 |
| 38 | 87.384615 | 5243 | 87.383333 |
| 39 | 89.692308 | 5382 | 89.700000 |
| 40 | 92.000000 | 5520 | 92.000000 |

Frame numbering is zero-based. If source metadata establishes a frame rate other than exactly 60 fps, preserve the target times, choose the nearest decoded source frame by presentation timestamp, and record the resulting indices and timestamps before running inference.

## What receives a box

Annotate every visible instance of these two semantic classes on each fixed frame:

- `enemy`: a recognizable gameplay enemy sprite. A stomped, squashed, frozen, or defeat-animation sprite remains an enemy while it is visually recognizable; record its state or subtype so this failure mode can be reported.
- `coin`: a recognizable non-HUD coin sprite in the gameplay world. This includes a short coin pop-out or reward animation because the evaluation is pixel-semantic rather than based on engine pickup state. The fixed HUD coin icon and digits are excluded.

Draw a tight box around visible sprite pixels, including visible animation parts and excluding shadows or effects that cannot be attributed to the sprite. For a sprite cut by the screen edge, box only the visible part and set `truncated: true`. For a partially occluded sprite that is still classifiable, box only its visible pixels and set `occluded: true`. Do not infer a box through terrain or another sprite.

Use `ignore: true` only when some visible pixels plausibly belong to an in-scope object but the class or extent cannot be decided from the source at full resolution. This is an adjudication outcome, not a way to omit difficult small, transformed, or partial objects. Ignored objects do not add false negatives, and an otherwise-unmatched prediction with IoU at least 0.5 against an ignored object does not add a false positive.

`object_id` identifies one visibility episode within a continuous tracking segment. It is not a persistent engine or game-world identity. `subtype` and `state` are descriptive fields and do not change the scored class label.

## Frame JSON

Ground truth uses one record per sampled or densely annotated frame:

```json
{
  "video_id": "evaluation",
  "frame_index": 397,
  "timestamp": 6.616667,
  "processed_size": [854, 480],
  "segment_id": "track-01",
  "objects": [
    {
      "object_id": "track-01-enemy-01",
      "label": "enemy",
      "bbox": [410, 302, 443, 340],
      "subtype": "snowball",
      "state": "active",
      "occluded": false,
      "truncated": false,
      "ignore": false
    }
  ]
}
```

`video_id`, `frame_index`, `timestamp`, `processed_size`, and `objects` are required. `segment_id` and `object_id` are required for tracking frames and omitted or optional for isolated detection frames. The remaining object fields are optional except `label` and `bbox`.

Prediction records use the same frame identity and a `detections` array:

```json
{
  "video_id": "evaluation",
  "frame_index": 397,
  "timestamp": 6.616667,
  "processed_size": [854, 480],
  "detections": [
    {
      "label": "enemy",
      "bbox": [409, 301, 444, 341],
      "confidence": 0.83,
      "track_id": 12,
      "subtype": "snowball"
    }
  ]
}
```

`track_id` is optional for detection scoring and required for identity scoring. Confidence is an evidence score and is not interpreted as a calibrated probability.

## Detection scoring

Match separately in every frame and class. A pair is eligible when its IoU is at least 0.5. Choose a one-to-one assignment that first maximizes the number of eligible pairs and then maximizes total IoU among those maximum-cardinality assignments. Each pair is a true positive; each remaining prediction is a false positive; each remaining non-ignored ground-truth object is a false negative.

Report ground-truth and prediction counts, TP, FP, FN, precision, recall, and F1 separately for `enemy` and `coin`, plus micro totals. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is `2TP/(2TP+FP+FN)`. A metric is JSON `null` when its own denominator is zero. For example, a class with references and no predictions has precision `null`, recall 0, and F1 0. No mean average precision is claimed from a single operating point.

## Visibility episodes and reference events

Reference events describe what becomes visible in the pixels. They do not claim an engine kill, pickup identity, or persistent world-object identity.

- `enemy_detected`: first processed frame of each enemy visibility episode.
- `enemy_disappeared`: first processed frame on which an enemy from an open episode is no longer visible.
- `item_detected`: first processed frame of each coin visibility episode.
- `coin_count_changed`: first processed frame on which the HUD number is clearly readable as a new value. Store both old and new values.

Annotate the event reference over the full 2.0 to 92.0 second interval at the 15 fps processing cadence, using frame stepping around each boundary. When an eligible object is already visible on the first scored frame, create its detected event at that frame and set `boundary: "left"`. This makes the starting state explicit and reproducible. When an object remains visible on the last scored frame, do not fabricate a disappearance; close the episode with `right_censored: true` in annotation notes and emit no `enemy_disappeared` event.

A fully hidden gap of 0.25 seconds or less is merged into the same episode only when the same sprite clearly resumes nearby with continuous motion. Emit no disappearance or re-detection for that gap. A longer gap closes the episode at its first absent processed frame; a later visible object starts a new visibility episode even when it may be the same game-world object. If continuity is genuinely ambiguous, flag it for adjudication before predictions are viewed.

The HUD coin icon is never an `item_detected` event. A non-HUD pop-out coin animation starts an item visibility episode. A stomped or defeated enemy stays in its current episode through its recognizable defeat animation, and its disappearance occurs only after recognizable enemy pixels vanish. A coin-count event comes only from a visually confirmed HUD digit transition; it must not be inferred from a coin vanishing, player contact, sound, or assumed engine behavior. Exclude unreadable transitions with an annotation note rather than guessing old or new values.

## Event JSON

Ground truth is stored as an event document:

```json
{
  "video_id": "evaluation",
  "interval": [2.0, 92.0],
  "events": [
    {
      "event_id": "event-0001",
      "event": "enemy_detected",
      "timestamp": 6.6,
      "episode_id": "enemy-episode-01"
    },
    {
      "event_id": "event-0002",
      "event": "coin_count_changed",
      "timestamp": 21.8,
      "details": {"old_count": 3, "new_count": 4}
    }
  ]
}
```

`event_id`, `event`, and `timestamp` are required. `episode_id` links visibility-boundary events when applicable. `details.old_count` and `details.new_count` are required for a retained `coin_count_changed` reference. Prediction events use the pipeline fields `event`, `timestamp`, `confirmed_at`, `confidence`, `track_id`, and `evidence`; coin counts may be stored as `evidence.old_count` and `evidence.new_count`.

## Event scoring

Match only events of the same type. A pair is eligible when absolute timestamp error is at most 0.5 seconds. Choose a one-to-one assignment that first maximizes eligible matches and then minimizes their total absolute time error. This prevents a locally closest but invalid global pairing from stealing another event's only valid match.

Report TP, FP, FN, precision, recall, F1, and matched timing summaries per event type and as micro totals. For every matched pair report:

- signed error: prediction `timestamp` minus reference `timestamp`;
- absolute error;
- confirmation latency: prediction `confirmed_at` minus reference `timestamp`;
- internal confirmation delay: prediction `confirmed_at` minus prediction `timestamp`.

Missing `confirmed_at` yields `null` latency and is excluded from latency means. Negative signed error is early and positive signed error is late. Confirmation latency may be larger than signed error because the reported event timestamp may be backdated to initial evidence.

For `coin_count_changed`, report old/new value agreement separately among temporally matched pairs. The primary optional bonus result should also be run with strict value matching, where a pair is eligible only when both old and new counts are present and equal. This prevents a timely but numerically wrong HUD read from receiving a true positive in the strict result.

## Dense tracking segments

Annotate three continuous segments at every 15 fps processed frame. Freeze their exact start/end timestamps before viewing any tracker predictions, spread them across the playable interval, and record the selection rule and ranges beside the annotations. Segments should collectively contain eligible objects; if a predetermined segment contains none, retain it and report that limitation rather than moving it after results are known.

Within a segment, assign one `object_id` to each continuous visibility episode. Use the tight visible-pixel box and partial-visibility rules above on every frame. IDs need not agree across segments. Pipeline `track_id` values are global within the video, but evaluation follows each ground-truth episode only inside its annotated segment.

Associate ground truth and predictions per frame with class-aware, one-to-one IoU at 0.5. For each ground-truth episode:

- an identity switch occurs when the assigned prediction `track_id` differs from the preceding matched observation's ID, including when a miss lies between them;
- a fragment is each additional run of matched observations after an intervening run of misses.

Report annotated observations, matched observations, observation recall, identity switches, and fragments, both as totals and by episode. Identity metrics are `null` when required IDs are missing. These numbers cover only three short, densely annotated segments, depend on the 0.5 IoU association, and describe visibility episodes rather than persistent game-world identities. They are diagnostic and must not be presented as full-video MOT benchmarks.

## Annotation workflow and audit trail

1. Record the source checksum, decoded dimensions/frame rate, processing cadence, evaluation interval, fixed 40-frame list, and tracking segment ranges.
2. Freeze detector and tracker configuration. Keep the evaluation clip unavailable to template creation and parameter tuning.
3. Annotate detection frames, dense tracking frames, and visual event boundaries without loading model output.
4. Perform a second manual pass in temporal order. Check every box with a ground-truth-only overlay and adjudicate boundary, occlusion, transformed-state, HUD, and short-gap flags.
5. Validate required fields, box geometry, unique frame/event IDs, timestamps, class names, and object-ID continuity.
6. Run inference once, preserve raw predictions, and calculate metrics from those saved predictions. Run smoothing-on and smoothing-off event scoring from the same raw detections.
7. Preserve annotation files, prediction files, metric JSON, configuration, and any post-evaluation changes. Never replace an original held-out result with a tuned rerun.

The report must state the small sample size, single-game/same-recorder domain, supported sprite appearances, transformed-state behavior, any ignored ambiguities, and observed failure cases. It must not invent sample counts or metric values before annotation and inference are complete.

### Execution separation record

On 2026-10-08 the isolated annotation agent was explicitly asked to begin the 40-frame raw-video annotation before the implementation owner declared detector configuration frozen. This reverses steps 2 and 3 above in wall-clock order, but preserves the intended information barrier: the annotator did not access detector output, source-level coordinates, or game-engine state, and did not reveal frame contents, labels, boxes, or aggregate label findings to the implementation owner. The implementation owner remained blind to the evaluation frames while tuning on development data. Held-out inference remains prohibited until configuration freeze. This exception must accompany the final evaluation provenance rather than being described as the nominal workflow.

### Executed reduced-scope tracking and event diagnostic

The completed tracking reference overrides the original 15 fps plan above. It uses the three preselected half-open raw-video segments `[8.0, 11.0)`, `[38.0, 41.0)`, and `[68.0, 71.0)` at 5 fps: 15 frames per segment and 45 frames total. Every retained frame was manually reviewed with ground-truth-only overlays. This is a short diagnostic rather than a dense 15 fps or full-video MOT benchmark. The reduced cadence can miss short visibility gaps and can undercount switches or fragments between sampled frames. Identity for overlapping coins in the third segment has greater uncertainty and follows visible screen-order continuity.

The executed event reference is also reduced in scope. It scores only the half-open interior intervals `[8.5, 10.5)`, `[38.5, 40.5)`, and `[68.5, 70.5)`, corresponding to a 0.5 second exclusion at each edge of the tracking segments. An event is retained only when its first visually observable boundary falls inside an interval. Objects already visible at the interval's left edge and episodes continuing past its right edge emit no event, avoiding claims based on unobserved prehistory or right-edge censoring. Prediction filtering must use these exact intervals.

Event timestamps are the first annotated 5 fps sample that shows the boundary and carry up to 0.2 seconds sampling uncertainty. The reference contains only `enemy_detected`, `enemy_disappeared`, and `item_detected`; it makes no engine-state, kill, pickup, or full-video coverage claim. A window with few or no events remains in the diagnostic rather than being moved after predictions are available. The annotation provenance is **AI-assisted visual annotation, not independently human audited**, performed from raw frames without prediction output.
