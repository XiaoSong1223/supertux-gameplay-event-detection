# Coin-counter OCR notes

`gameplay_events.hud.CoinCounter` reads the upper-right SuperTux v0.6.3 coin counter from video pixels. It normalizes frames to the development transcode's 854x480 geometry, classifies the four right-aligned 8-pixel digit slots, rejects weak/partial reads, and confirms a new stable value before emitting `coin_count_changed`. The first confirmed value is only a baseline.

The checked-in `assets/hud/coin_digits_v1.npz` model is produced by `scripts/build_hud_templates.py`. Its labels and timestamps are written explicitly in that script and were manually read from `data/videos/development.webm`; the held-out evaluation clip is not read by the builder. The official SuperTux v0.6.3 bitmap font at `assets/upstream/data/images/engine/fonts/white.png` (commit `c1ddb4f28c54de77f07aa9965231a91c0ed06708`) was used to verify glyph geometry, while the trained pixels come only from the development HUD. Rebuild with:

```bash
.venv/bin/python scripts/build_hud_templates.py
```

Confidence is classifier separation scaled by lower-tail development margins. It is an evidence heuristic, not a calibrated probability. The reader is intentionally conservative around compression blur and animated counter transitions. Its main limitations are the fixed v0.6.3 HUD layout, the 854x480 calibration domain, tiny 8-pixel glyphs, and possible rejection when clouds or overlays pass through the counter. Model-training patches are not test accuracy. Independent fixed-frame held-out results, including the 0% coverage failure and post-hoc diagnosis, are reported in `docs/HUD_EVALUATION.md`; per-frame outputs remain in `outputs/metrics/hud.json`.
