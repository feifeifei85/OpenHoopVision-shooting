"""Shared geometry, trajectory smoothing, and serialization utilities."""

from typing import TypeAlias

import numpy as np

from core.config import CONFIG

JSONValue: TypeAlias = float | list["JSONValue"] | None


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the intersection over union of two xyxy boxes."""
    left, top = np.maximum(first[:2], second[:2])
    right, bottom = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(right - left)) * max(0.0, float(bottom - top))
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def apply_affine_transform(points_xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map image points through a 2D affine transform."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    if points_xy.shape[-1] != 2 or transform.shape[-2:] != (2, 3):
        raise ValueError("Expected points ending in 2 coordinates and a 2x3 affine transform")
    return np.einsum("...ij,...j->...i", transform[..., :2], points_xy) + transform[..., 2]


def skeleton_edges() -> list[tuple[int, int]]:
    """Return the configured connections between pose keypoints."""
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (15, 18),
        (15, 19),
        (16, 20),
        (16, 21),
        (16, 22),
        (9, CONFIG["HAND_BASES"]["left"]),
        (10, CONFIG["HAND_BASES"]["right"]),
    ]
    # COCO WholeBody face landmark contours: jaw, brows, nose, eyes, and mouth.
    face_start = CONFIG["FACE_START_INDEX"]
    face_contours = (
        (0, 16, False),
        (17, 21, False),
        (22, 26, False),
        (27, 30, False),
        (31, 35, False),
        (36, 41, True),
        (42, 47, True),
        (48, 59, True),
        (60, 67, True),
    )
    for start, end, closed in face_contours:
        edges.extend((face_start + i, face_start + i + 1) for i in range(start, end))
        if closed:
            edges.append((face_start + end, face_start + start))
    # Each hand has a wrist followed by four-joint finger chains.
    for base in CONFIG["HAND_BASES"].values():
        for first in CONFIG["HAND_FINGER_BASE_OFFSETS"].values():
            edges.append((base, base + first))
            edges.extend((base + i, base + i + 1) for i in range(first, first + 3))
    return edges


SKELETON_EDGES = skeleton_edges()


def smooth_trajectory(points: np.ndarray, radius: int = 3, max_gap: int = 2) -> np.ndarray:
    """Fill brief missed detections and apply a finite-aware triangular smoother."""
    smoothed = np.asarray(points, dtype=np.float32).copy()
    finite = np.isfinite(smoothed).all(axis=1)
    valid_frames = np.flatnonzero(finite)
    for left, right in zip(valid_frames[:-1], valid_frames[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            alpha = np.arange(1, gap + 1, dtype=np.float32)[:, None] / (gap + 1)
            smoothed[left + 1 : right] = smoothed[left] + alpha * (smoothed[right] - smoothed[left])
    finite = np.isfinite(smoothed).all(axis=1)
    source = smoothed.copy()
    for frame in np.flatnonzero(finite):
        indices = np.arange(max(0, frame - radius), min(len(source), frame + radius + 1))
        indices = indices[finite[indices]]
        weights = radius + 1 - np.abs(indices - frame)
        smoothed[frame] = np.average(source[indices], axis=0, weights=weights)
    return smoothed


def finite_median_filter(values: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a centered median filter while ignoring non-finite samples."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Median filtering expects a one-dimensional signal")
    if radius < 0:
        raise ValueError("Filter radius must be non-negative")
    if not len(values):
        return values.copy()

    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(values, radius, constant_values=np.nan), 2 * radius + 1
    )
    finite = np.isfinite(windows)
    counts = np.count_nonzero(finite, axis=1)
    ordered = np.sort(np.where(finite, windows, np.inf), axis=1)
    rows = np.arange(len(values))
    lower = ordered[rows, np.maximum(counts - 1, 0) // 2]
    upper = ordered[rows, counts // 2]
    result = (lower + upper) * 0.5
    result[counts == 0] = np.nan
    return result


def json_values(values: np.ndarray | np.generic) -> JSONValue:
    """Convert numeric arrays to JSON-compatible values, replacing nonfinite values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        value = float(array)
        return value if np.isfinite(value) else None
    return [json_values(value) for value in array]
