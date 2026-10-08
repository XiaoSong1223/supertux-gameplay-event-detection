from pathlib import Path
import cv2
import numpy as np
import pytest
from gameplay_events.detection import SpriteDetector, box_iou

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def detector():
    cv2.setNumThreads(1)
    return SpriteDetector(ROOT/"assets/templates/detector.json")


def add_coin(frame, x, y, pose=0):
    source=cv2.imread(str(ROOT/f"assets/upstream/data/images/objects/coin/coin-{pose}.png"),-1)
    source=cv2.resize(source,None,fx=.625,fy=.625,interpolation=cv2.INTER_AREA)
    h,w=source.shape[:2];a=source[:,:,3:]/255
    frame[y:y+h,x:x+w]=(source[:,:,:3]*a+frame[y:y+h,x:x+w]*(1-a)).astype(np.uint8)


def test_hud_icon_is_excluded_but_same_world_sprite_is_detected(detector):
    frame=np.full((180,854,3),(210,155,95),np.uint8)
    add_coin(frame,805,3);add_coin(frame,150,80)
    coins=[d for d in detector.detect(frame) if d.label=="coin"]
    assert len(coins)==1
    assert box_iou(coins[0].bbox,(150,80,170,100))>.5


def test_adjacent_coins_remain_distinct_and_constant_image_is_empty(detector):
    blank=np.full((180,854,3),(210,155,95),np.uint8)
    assert detector.detect(blank)==[]
    add_coin(blank,200,70);add_coin(blank,220,70)
    coins=[d for d in detector.detect(blank) if d.label=="coin"]
    assert len(coins)==2
    assert all(np.isfinite(d.confidence) and 0<=d.confidence<=1 for d in coins)


def test_rejects_uncalibrated_processing_width(detector):
    with pytest.raises(ValueError,match="expects width"):
        detector.detect(np.zeros((480,640,3),np.uint8))
