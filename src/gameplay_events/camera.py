"""Lightweight image-translation estimation for scrolling gameplay footage.

The estimate is the robust consensus of sparse background feature tracks.  It
works well when textured terrain occupies most of the usable frame.  Like any
single-layer translation model, it can be biased when a moving foreground owns
most trackable corners or when strong parallax creates several equally large
motion layers.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class CameraMotion:
    """Estimate image-coordinate camera translation since the previous frame."""

    def __init__(
        self,
        hud_height: int = 28,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        min_points: int = 12,
        fb_max_error: float = 1.5,
        min_inlier_ratio: float = 0.45,
        max_translation: float = 96.0,
        cut_correlation_threshold: float = 0.30,
        cut_difference_threshold: float = 0.08,
    ) -> None:
        if isinstance(hud_height, bool) or not isinstance(hud_height, int) or hud_height < 0:
            raise ValueError("hud_height must be a non-negative integer")
        if isinstance(max_corners, bool) or not isinstance(max_corners, int) or max_corners < 1:
            raise ValueError("max_corners must be a positive integer")
        if not _finite_between(quality_level, 0.0, 1.0, lower_open=True):
            raise ValueError("quality_level must be in (0, 1]")
        if not _finite_positive(min_distance):
            raise ValueError("min_distance must be finite and positive")
        if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 3:
            raise ValueError("min_points must be an integer of at least 3")
        if not _finite_positive(fb_max_error):
            raise ValueError("fb_max_error must be finite and positive")
        if not _finite_between(min_inlier_ratio, 0.0, 1.0, lower_open=True):
            raise ValueError("min_inlier_ratio must be in (0, 1]")
        if not _finite_positive(max_translation):
            raise ValueError("max_translation must be finite and positive")
        if not _finite_between(cut_correlation_threshold, -1.0, 1.0):
            raise ValueError("cut_correlation_threshold must be between -1 and 1")
        if not _finite_between(cut_difference_threshold, 0.0, 1.0):
            raise ValueError("cut_difference_threshold must be between 0 and 1")

        self.hud_height = hud_height
        self.max_corners = max_corners
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.min_points = min_points
        self.fb_max_error = float(fb_max_error)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_translation = float(max_translation)
        self.cut_correlation_threshold = float(cut_correlation_threshold)
        self.cut_difference_threshold = float(cut_difference_threshold)

        self.scene_cut = False
        self._previous_gray: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float]:
        """Return ``(dx, dy)`` translation from the preceding image to this one.

        Positive values point right and down in image coordinates.  The first
        frame, low-texture pairs, and rejected discontinuities return ``(0, 0)``.
        ``scene_cut`` describes only the current update.
        """

        gray = _to_gray(frame_bgr)
        if gray.shape[0] <= self.hud_height:
            raise ValueError("frame height must be greater than hud_height")
        gray = np.ascontiguousarray(gray[self.hud_height :])
        self.scene_cut = False

        previous = self._previous_gray
        self._previous_gray = gray.copy()
        if previous is None:
            return 0.0, 0.0
        if previous.shape != gray.shape:
            self.scene_cut = True
            return 0.0, 0.0

        raw_difference = _mean_absolute_difference(previous, gray)
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )
        if points is None or len(points) < self.min_points:
            # Two similarly flat frames are simply unobservable.  A large
            # appearance jump with no persistent texture is treated as a cut.
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or forward_status is None:
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous,
            current_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward_points is None or backward_status is None:
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        source = points.reshape(-1, 2)
        target = current_points.reshape(-1, 2)
        returned = backward_points.reshape(-1, 2)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(target).all(axis=1)
            & np.isfinite(returned).all(axis=1)
        )
        forward_backward_error = np.linalg.norm(returned - source, axis=1)
        valid &= forward_backward_error <= self.fb_max_error
        displacements = target[valid] - source[valid]
        if len(displacements) < self.min_points:
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        initial_median = np.median(displacements, axis=0)
        residuals = np.linalg.norm(displacements - initial_median, axis=1)
        residual_median = float(np.median(residuals))
        residual_mad = float(np.median(np.abs(residuals - residual_median)))
        residual_limit = max(1.25, residual_median + 3.0 * 1.4826 * residual_mad)
        inliers = residuals <= residual_limit
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(displacements)
        if inlier_count < self.min_points or inlier_ratio < self.min_inlier_ratio:
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        dx, dy = np.median(displacements[inliers], axis=0).astype(float).tolist()
        if math.hypot(dx, dy) > self.max_translation:
            self.scene_cut = raw_difference > self.cut_difference_threshold
            return 0.0, 0.0

        correlation, aligned_difference = _aligned_quality(previous, gray, dx, dy)
        if (
            correlation < self.cut_correlation_threshold
            and aligned_difference > self.cut_difference_threshold
        ):
            self.scene_cut = True
            return 0.0, 0.0

        return float(dx), float(dy)

    def reset(self) -> None:
        """Forget the prior frame, for example at a known scene boundary."""

        self._previous_gray = None
        self.scene_cut = False


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame_bgr must be a numpy array")
    if frame.dtype != np.uint8:
        raise ValueError("frame_bgr must have uint8 dtype")
    if frame.ndim == 2:
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("frame_bgr must not be empty")
        return frame.copy()
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("frame_bgr must be HxW BGR or BGRA uint8 data")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame_bgr must not be empty")
    conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame, conversion)


def _mean_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(left, right))) / 255.0


def _aligned_quality(
    previous: np.ndarray,
    current: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    height, width = previous.shape
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        previous,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpAffine(
        np.full_like(previous, 255),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if np.count_nonzero(valid) < max(64, int(previous.size * 0.25)):
        return -1.0, 1.0

    aligned_values = aligned[valid].astype(np.float32)
    current_values = current[valid].astype(np.float32)
    difference = float(np.mean(np.abs(aligned_values - current_values))) / 255.0
    aligned_std = float(aligned_values.std())
    current_std = float(current_values.std())
    if aligned_std < 1.0 or current_std < 1.0:
        correlation = 1.0 if difference <= 0.02 else 0.0
    else:
        correlation = float(np.corrcoef(aligned_values, current_values)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
    return correlation, difference


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_between(
    value: object,
    lower: float,
    upper: float,
    *,
    lower_open: bool = False,
) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return False
    lower_ok = float(value) > lower if lower_open else float(value) >= lower
    return lower_ok and float(value) <= upper
