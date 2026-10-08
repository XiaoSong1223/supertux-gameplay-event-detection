"""Two-stage masked sprite matching, with no engine state or frame lookup table.

Coarse correlation proposes locations; full-resolution correlation and color
agreement verify them. The score is a similarity heuristic, not a probability.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np

from .types import Detection


@dataclass
class _Template:
    label: str
    subtype: str
    name: str
    bgr: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    coarse: np.ndarray
    coarse_mask: np.ndarray


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _finite_match(image, template, mask):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(result, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)


class SpriteDetector:
    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        self.config = json.loads(config_path.read_text())
        self.templates = []
        for spec in self.config["templates"]:
            path = config_path.parent / spec["path"]
            source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if source is None or source.ndim != 3 or source.shape[2] != 4:
                raise ValueError(f"Expected RGBA sprite: {path}")
            for scale in self.config["sprite_scales"]:
                rgba = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                alpha = np.uint8(rgba[:, :, 3] > 180) * 255
                x, y, w, h = cv2.boundingRect(alpha)
                if min(w, h) < (2 if spec["label"] == "coin" else 4):
                    continue
                rgba = rgba[y:y+h, x:x+w]
                alpha = alpha[y:y+h, x:x+w]
                transforms = [None, 1] if spec["label"] == "enemy" else [None]
                if spec.get("vertical_flip", False):
                    transforms += [0, -1]
                for transform in transforms:
                    bgr = np.ascontiguousarray(rgba[:, :, :3] if transform is None else cv2.flip(rgba[:, :, :3], transform))
                    mask = np.ascontiguousarray(alpha if transform is None else cv2.flip(alpha, transform))
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    coarse = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
                    coarse_mask = cv2.resize(mask, (coarse.shape[1], coarse.shape[0]), interpolation=cv2.INTER_NEAREST)
                    self.templates.append(_Template(spec["label"], spec["subtype"], path.stem, bgr, gray, mask, coarse, coarse_mask))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        expected = self.config["processed_width"]
        if frame.shape[1] != expected:
            raise ValueError(f"Detector expects width {expected}, got {frame.shape[1]}; resize before detection")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        height, width = gray.shape
        candidates = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, np.array([14, 100, 140], np.uint8), np.array([40, 255, 255], np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(gold)
        coin_proposals = [(int(x+w/2), int(y+h/2)) for x,y,w,h,area in stats[1:]
                          if 11 <= h <= 23 and 1 <= w <= 22 and area >= 10 and y >= self.config["hud_exclusion_height"]]
        for tpl in self.templates:
            h, w = tpl.gray.shape
            if tpl.label == "coin":
                positions = [(int(cx-w/2), int(cy-h/2)) for cx,cy in coin_proposals]
            else:
                positions = []
            # Keep correlation proposals too: gold connected components can
            # merge across vertically adjacent coins. Thin edge-on templates
            # use color proposals only to avoid ambiguous narrow-strip scans.
            if w >= 4:
                response = _finite_match(small, tpl.coarse, tpl.coarse_mask)
                response[: self.config["hud_exclusion_height"] // 2] = -1
                peaks = (response >= self.config["proposal_threshold"]) & (response == cv2.dilate(response, np.ones((5, 5), np.uint8)))
                ys, xs = np.where(peaks)
                if len(xs) > self.config["max_proposals_per_template"]:
                    order = np.argsort(response[ys, xs])[-self.config["max_proposals_per_template"]:]
                    xs, ys = xs[order], ys[order]
                positions += [(int(x)*2, int(y)*2) for x,y in zip(xs,ys)]
            for px, py in positions:
                x0, y0 = max(0, px-3), max(0, py-3)
                x1, y1 = min(width, px+w+4), min(height, py+h+4)
                if x1-x0 < w or y1-y0 < h:
                    continue
                local = _finite_match(gray[y0:y1, x0:x1], tpl.gray, tpl.mask)
                _, corr, _, loc = cv2.minMaxLoc(local)
                if corr < self.config["thresholds"][tpl.label]:
                    continue
                bx, by = x0+loc[0], y0+loc[1]
                patch = frame[by:by+h, bx:bx+w]
                valid = tpl.mask > 0
                mae = float(np.mean(np.abs(patch[valid].astype(np.float32)-tpl.bgr[valid].astype(np.float32))))
                if mae > self.config["max_color_mae"][tpl.label]:
                    continue
                score = .8*float(corr) + .2*max(0.0, 1-mae/100)
                candidates.append(Detection(tpl.label, (float(bx), float(by), float(bx+w), float(by+h)), float(np.clip(score,0,1)), tpl.subtype))
        kept = []
        for detection in sorted(candidates, key=lambda d:d.confidence, reverse=True):
            if not any(d.label == detection.label and (
                box_iou(d.bbox, detection.bbox) > self.config["nms_iou"] or
                (d.label == "coin" and abs((d.bbox[0]+d.bbox[2]-detection.bbox[0]-detection.bbox[2])/2)<7
                 and abs((d.bbox[1]+d.bbox[3]-detection.bbox[1]-detection.bbox[3])/2)<6)) for d in kept):
                kept.append(detection)
        return kept
