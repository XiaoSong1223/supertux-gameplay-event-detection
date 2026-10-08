# Frozen HUD evaluation

The frozen `CoinCounter.read` implementation was evaluated once on the same 40 fixed held-out frames used by the detection evaluation. The frames span 2.0–92.0 seconds using a linspace schedule mapped to the nearest 60 fps source frame.

Ground truth was read manually from enlarged upper-right crops of the raw evaluation video before any HUD prediction was viewed. The annotation is recorded as **AI-assisted visual, no human independent audit**. All 40 sampled counters were visually clear; there were no ambiguous/null labels. The saved annotation SHA-256 is `c0aaf113a0a7c14e3447a89c9e395922aa33e00b928fb888aede5daf3678f4de`.

### Post-evaluation annotation audit

After review identified a possible error at source frame 2889, all 40 raw HUD crops were re-read from the PNGs produced by the sequential frame extractor. One label was corrected: sample 21 at 48.15 seconds changed from 363 to 365. The last glyph is a 5; enlarged comparison against known 3/5 glyphs and the surrounding monotonic sequence confirmed the correction. The other 39 labels were unchanged.

This was an annotation error rather than a frame-selection error. Every saved raw sample PNG was pixel-identical to a fresh sequential decode, and a direct seek to frame 2889 was also pixel-identical. The initial annotation and result remain unchanged for provenance. The corrected annotation is `data/annotations/hud_evaluation_audited.json` (SHA-256 `b02eca1515a7a82872e1f28ee816f102c4a55f741daf44b9e3b35d111077e008`) and includes the machine-readable correction log. Its corresponding result is `outputs/metrics/hud_audited.json` (SHA-256 `3ae0bcf2a72fbe3fab0d63da88c12991e0168a7a320223ea865cd24b3b947283`). Metrics are unchanged because the frozen reader rejected all 40 frames.

## Result

| Measure | Result |
|---|---:|
| Evaluable labelled frames | 40 |
| Accepted reads | 0 |
| Exact accepted reads | 0 |
| Wrong accepted reads | 0 |
| Rejected/abstained reads | 40 |
| Coverage | 0/40 = 0.0% |
| Exact accuracy among accepted reads | Undefined (no accepted reads) |
| Overall exact rate including abstentions | 0/40 = 0.0% |
| Ambiguous ground-truth labels | 0 |

The zero wrong-read count does not indicate correctness: the frozen reader abstained on every held-out sample, so there is no held-out evidence of exact-count accuracy. The result exposes a complete coverage failure outside the development calibration clip. The model and threshold were not changed after this result.

## Post-evaluation failure diagnosis

A post-hoc inspection compared the frozen slot outputs at 5 and 15 seconds in each video. The raw crops and 8x12 slot boxes are saved in `outputs/diagnostics/hud_posthoc.png`; `scripts/diagnose_hud.py` reproduces the image after verifying the frozen model hashes.

| Clip/time | Visually read HUD | Frozen slot top labels/confidence at x=815, 823, 831, 839 | Read result |
|---|---:|---|---|
| Development 5 s | 108 | N/0.93, 1/0.94, 0/0.81, 8/0.87 | 108 accepted |
| Development 15 s | 115 | N/0.28, 3/0.17, 1/0.88, 5/0.88 | Rejected |
| Evaluation 5 s | 347 | N/0.49, 1/0.30, N/0.40, 7/0.50 | Rejected |
| Evaluation 15 s | 354 | N/0.18, 1/0.14, 5/0.20, 4/0.08 | Rejected |

`N` is the icon/non-digit class. The icon and glyphs occupy the same frozen boxes in both videos, so these examples do not support a layout-shift explanation. Instead, held-out glyphs receive low class separation and sometimes the wrong class despite visual alignment. A development frame at 15 seconds also fails when the leading digit and icon boundary have low separation, showing that the classifier was already sensitive to local rendering/background conditions. The evaluation strings 347 and 354 were not among the labelled training strings; because classification is slot-wise, the unseen whole-string combination is not itself causal, though digits 7 and 4 each have sparse labelled contexts. The evidence is most consistent with a development-pixel/background and compression generalization failure, but four diagnostic frames cannot isolate one cause. This diagnosis was performed after the metrics were fixed and was not used to change the model or threshold.

This fixed-frame test evaluates raw `read` observations. It does not estimate `coin_count_changed` event precision, recall, or confirmation latency because no accepted held-out observations were available and a full transition timeline was not independently annotated.

The audited per-frame ground truth is in `data/annotations/hud_evaluation_audited.json`; every audited prediction status is preserved in `outputs/metrics/hud_audited.json`. The original `data/annotations/hud_evaluation.json` and `outputs/metrics/hud.json` are retained unchanged as the first-pass record. Both metrics files verify the frozen hashes for `src/gameplay_events/hud.py` and `assets/hud/coin_digits_v1.npz`.

Reproduce the frozen scoring run with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/evaluate_hud.py
```
