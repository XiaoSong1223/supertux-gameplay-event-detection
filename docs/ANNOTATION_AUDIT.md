# Post-evaluation detection annotation audit

Audit date: 2026-10-09 (UTC+08:00)

This was a post-evaluation **AI visual audit**, not a human audit. It was
performed after the first evaluation result had been produced. The original
ground truth in `data/annotations/detection_evaluation.json` was preserved
unchanged.

## Isolation and material reviewed

The audit reviewed all 40 frozen raw PNG samples in
`data/annotation_work/evaluation_detection_samples` and GT-only overlays
rendered directly from the original annotation JSON. The ten raw contact sheets
were also checked. No detector predictions, prediction overlays,
`outputs/review`, detector metrics, or event metrics were opened or used.

The review applied the original scope consistently:

- include every recognizable non-HUD world coin, including round, edge-on,
  short pop-out, and reward poses;
- include recognizable enemies, including transformed, occluded, and defeated
  visible sprites;
- exclude the player, HUD coin icon/digits, sign illustrations, projectiles,
  bells and other non-coin power-ups, blocks, and debris;
- keep boxes tight around visible sprite pixels at 854x480.

## Full-frame review record

| Samples | Source frames | Audit result |
|---|---:|---|
| 01-04 | 120-535 | Complete. Sample 03's emerging round reward coin and both visible enemies are already covered. |
| 05-08 | 674-1089 | Complete. Round and edge-on coins, the player-occluded coin, bomb, iceblock, pink/gray snowballs are correctly represented. The small orange object in sample 08 is a projectile. |
| 09-12 | 1228-1643 | Complete. Visible enemies are covered; the orange object in sample 09 is a projectile. Sample 10 correctly contains no scoped object. |
| 13-16 | 1782-2197 | Complete. Enemy boxes and both six-coin arrays are complete. The orange object in sample 14 is a projectile; sign art in sample 16 remains excluded. |
| 17-20 | 2335-2751 | Complete. All spiky, snowball, and iceblock sprites are covered. The orange object in sample 17 is a projectile and the bell in sample 20 is a non-coin power-up. |
| 21-24 | 2889-3305 | Complete. Sample 21's short edge-on coin and defeated gray enemy are already included; its bell is excluded. Sample 24's six edge-on coins and two enemies are complete. |
| 25-28 | 3443-3858 | Complete. All four six-coin arrays and all visible ground, airborne, and transformed enemies are covered, including partially clipped coins at the left frame boundary. |
| 29-32 | 3997-4412 | Complete. The short pop-out/reward coin in sample 29 and the short coin above the block in sample 32 are already boxed. Sample 30 correctly contains no scoped object. The orange object in sample 31 is a projectile. |
| 33-36 | 4551-4966 | Complete. All gray snowballs and spiky enemies, including the player-overlapped visible spiky in sample 36, are covered. The orange object in sample 33 is a projectile. |
| 37-40 | 5105-5520 | Complete. These frames contain no scoped world coin or enemy; player, HUD, and end-of-level display elements remain excluded. |

## Result and change log

No corrections were required. The audited totals remain 114 objects: 57
enemies and 57 coins across 40 frames, with six empty frames. All 40 frame
indices match the frozen manifest, timestamps match `frame_index / 60`, source
images are 854x480, and all 114 boxes are valid and in bounds.

Because the audit found no annotation change, no
`data/annotations/detection_evaluation_audited.json` was created. The original
annotation file remains the authoritative GT for scoring. This zero-change
result does not turn the annotation into human-audited ground truth and does
not remove the need to disclose that the audit occurred after evaluation.
