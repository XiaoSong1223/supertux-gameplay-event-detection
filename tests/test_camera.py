import cv2
import numpy as np
import pytest

from gameplay_events.camera import CameraMotion


def textured_frame(seed: int, height: int = 220, width: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def test_estimates_translation_and_ignores_changed_hud() -> None:
    first = textured_frame(1)
    second = translate(first, 7.0, -4.0)
    # A discontinuous HUD should not participate in feature tracking or cut QA.
    first[:28] = textured_frame(2, height=28)
    second[:28] = textured_frame(3, height=28)
    motion = CameraMotion()

    assert motion.update(first) == (0.0, 0.0)
    dx, dy = motion.update(second)

    assert dx == pytest.approx(7.0, abs=0.35)
    assert dy == pytest.approx(-4.0, abs=0.35)
    assert motion.scene_cut is False


def test_robust_consensus_rejects_a_moving_foreground_patch() -> None:
    first = textured_frame(4)
    second = translate(first, 5.0, 3.0)
    foreground = textured_frame(5, height=80, width=90)
    first[90:170, 40:130] = foreground
    second[90:170, 70:160] = foreground
    motion = CameraMotion(max_corners=700)

    motion.update(first)
    dx, dy = motion.update(second)

    assert dx == pytest.approx(5.0, abs=0.6)
    assert dy == pytest.approx(3.0, abs=0.6)
    assert motion.scene_cut is False


def test_textured_scene_cut_returns_zero_and_sets_flag() -> None:
    motion = CameraMotion()
    motion.update(textured_frame(10))

    estimate = motion.update(textured_frame(11))

    assert estimate == (0.0, 0.0)
    assert motion.scene_cut is True


def test_low_texture_pair_is_unobservable_but_not_a_cut() -> None:
    motion = CameraMotion()
    flat = np.full((180, 260, 3), 80, dtype=np.uint8)
    motion.update(flat)

    assert motion.update(flat.copy()) == (0.0, 0.0)
    assert motion.scene_cut is False


def test_large_low_texture_appearance_jump_is_a_cut() -> None:
    motion = CameraMotion()
    dark = np.full((180, 260, 3), 20, dtype=np.uint8)
    bright = np.full((180, 260, 3), 230, dtype=np.uint8)
    motion.update(dark)

    assert motion.update(bright) == (0.0, 0.0)
    assert motion.scene_cut is True


def test_reset_forgets_prior_frame_without_reporting_a_cut() -> None:
    motion = CameraMotion()
    motion.update(textured_frame(20))
    motion.reset()

    assert motion.update(textured_frame(21)) == (0.0, 0.0)
    assert motion.scene_cut is False


def test_resolution_change_and_invalid_input_are_handled() -> None:
    motion = CameraMotion()
    motion.update(textured_frame(30))
    assert motion.update(textured_frame(30, height=240)) == (0.0, 0.0)
    assert motion.scene_cut is True

    with pytest.raises(ValueError, match="uint8"):
        motion.update(np.zeros((100, 100, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="greater than hud_height"):
        motion.update(np.zeros((28, 100, 3), dtype=np.uint8))
