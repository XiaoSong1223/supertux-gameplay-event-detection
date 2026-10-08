"""Build a deterministic sprite bank from the version-pinned upstream assets."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
specs = []
groups = [
    ("creatures/snowball", "snowball-*.png", "enemy", "snowball"),
    ("creatures/snowball", "smartball-*.png", "enemy", "smart_snowball"),
    ("creatures/mr_iceblock", "iceblock-*.png", "enemy", "iceblock"),
    ("creatures/mr_iceblock", "stomped-*.png", "enemy", "iceblock_stunned"),
    ("creatures/mr_iceblock", "waking-*.png", "enemy", "iceblock_stunned"),
    ("creatures/jumpy", "left-*.png", "enemy", "jumpy"),
    ("creatures/jumpy", "wooden-*.png", "enemy", "wooden_jumpy"),
    ("creatures/spiky", "spikywalk*.png", "enemy", "spiky"),
    ("creatures/mr_bomb", "bomb[1-8].png", "enemy", "bomb"),
    ("creatures/flying_snowball", "left-*.png", "enemy", "flying_snowball"),
    ("objects/coin", "coin-*.png", "coin", "coin"),
]
for folder, pattern, label, subtype in groups:
    for path in sorted((ROOT / "assets/upstream/data/images" / folder).glob(pattern)):
        specs.append({"path": "../upstream/data/images/" + str(path.relative_to(ROOT/"assets/upstream/data/images")), "label": label, "subtype": subtype, "vertical_flip": subtype == "iceblock"})
config = {"processed_width":854,"sprite_scales":[.625],"hud_exclusion_height":28,
          "proposal_threshold":.57,"thresholds":{"enemy":.76,"coin":.70},
          "max_color_mae":{"enemy":52,"coin":45},"max_proposals_per_template":120,"nms_iou":.30,"templates":specs}
(ROOT/"assets/templates/detector.json").write_text(json.dumps(config,indent=2)+"\n")
print('sprite files',len(specs))
