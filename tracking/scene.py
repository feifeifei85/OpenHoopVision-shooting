"""Estimate camera motion and select a stable basket."""

import cv2
import numpy as np


def estimate_background_camera_motion(
    previous_gray: np.ndarray,
    previous_mask: np.ndarray,
    current_gray: np.ndarray,
    current_mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Estimate background translation and zoom between adjacent frames."""
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        mask=previous_mask,
        blockSize=7,
    )
    if corners is None or len(corners) < 12:
        return None, None, 0.0
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None, None, 0.0
    valid = status.reshape(-1).astype(bool) & np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
    if errors is not None:
        valid &= errors.reshape(-1) <= 45.0
    previous_points = corners.reshape(-1, 2)[valid]
    next_points = next_points.reshape(-1, 2)[valid]
    if len(next_points) < 12:
        return None, None, 0.0

    height, width = current_mask.shape
    x = np.clip(np.rint(next_points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(next_points[:, 1]).astype(int), 0, height - 1)
    background = current_mask[y, x] > 0
    previous_points = previous_points[background]
    next_points = next_points[background]
    displacement = next_points - previous_points
    if len(displacement) < 12:
        return None, None, 0.0
    center = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - center, axis=1)
    residual_center = float(np.median(residual))
    residual_scale = 1.4826 * float(np.median(np.abs(residual - residual_center)))
    inliers = residual <= residual_center + max(2.0, 3.0 * residual_scale)
    if np.count_nonzero(inliers) < 12:
        return None, None, 0.0
    shift = np.median(displacement[inliers], axis=0) / max(scale, 1e-6)
    inlier_ratio = float(np.mean(inliers))
    support = min(float(np.count_nonzero(inliers)) / 40.0, 1.0)
    confidence = float(np.clip(inlier_ratio * support, 0.0, 1.0))

    affine, affine_inliers = cv2.estimateAffinePartial2D(
        previous_points,
        next_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=2000,
        confidence=0.99,
    )
    if affine is None or affine_inliers is None:
        return shift.astype(np.float32), None, confidence
    affine_count = int(np.count_nonzero(affine_inliers))
    if affine_count < 12:
        return shift.astype(np.float32), None, confidence

    zoom = float(np.hypot(affine[0, 0], affine[1, 0]))
    rotation = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    if not 0.85 <= zoom <= 1.15 or abs(rotation) > 20.0:
        return shift.astype(np.float32), None, confidence

    affine_confidence = min(affine_count / 40.0, 1.0) * affine_count / len(displacement)
    confidence = min(confidence, affine_confidence)
    transform = np.eye(3, dtype=np.float32)
    transform[:2, :2] = affine[:, :2]
    transform[:2, 2] = affine[:, 2] / max(scale, 1e-6)
    return shift.astype(np.float32), transform, float(np.clip(confidence, 0.0, 1.0))


def select_rim_box(
    rims_by_frame: list[np.ndarray],
    rim_scores_by_frame: list[np.ndarray],
    frame_count: int,
) -> np.ndarray:
    """Find a stable rim box by clustering per-frame detections."""
    rim_candidates = [boxes for boxes in rims_by_frame if len(boxes)]
    rim_scores = [scores for scores in rim_scores_by_frame if len(scores)]
    rim_box = np.full(4, np.nan, dtype=np.float32)
    if rim_candidates:
        all_rim_boxes = np.concatenate(rim_candidates, axis=0)
        all_rim_scores = np.concatenate(rim_scores, axis=0)
        centers = (all_rim_boxes[:, :2] + all_rim_boxes[:, 2:]) * 0.5
        sizes = (all_rim_boxes[:, 2:] - all_rim_boxes[:, :2]).clip(min=1.0)
        cluster_candidates = []
        for seed in all_rim_boxes:
            seed_width, seed_height = seed[2] - seed[0], seed[3] - seed[1]
            seed_scale = max(float(np.hypot(seed_width, seed_height)), 1.0)
            seed_center = (seed[:2] + seed[2:]) * 0.5
            distances = np.linalg.norm(centers - seed_center, axis=1) / seed_scale
            seed_size = np.maximum(np.asarray((seed_width, seed_height)), 1.0)
            ratios = np.maximum(sizes / seed_size, seed_size / sizes).max(axis=1)
            inliers = (distances <= 0.45) & (ratios <= 2.5)
            cluster_candidates.append((float(all_rim_scores[inliers].sum()), inliers))
        _, rim_inliers = max(cluster_candidates, key=lambda item: item[0])
        if np.count_nonzero(rim_inliers) >= max(3, round(frame_count * 0.01)):
            rim_box = np.median(all_rim_boxes[rim_inliers], axis=0).astype(np.float32)

    return rim_box


def track_rim_boxes(
    rims_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    camera_shift_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate the basket in stabilized coordinates, retaining per-frame boxes."""
    offsets = np.tile(camera_shift_xy, (1, 2))
    stabilized = [boxes - offsets[index] for index, boxes in enumerate(rims_by_frame)]
    reference = select_rim_box(stabilized, scores_by_frame, len(rims_by_frame))
    tracked = reference[None, :] + offsets
    if not np.isfinite(reference).all():
        return reference, tracked
    for index, (boxes, scores) in enumerate(zip(rims_by_frame, scores_by_frame)):
        if not len(boxes):
            continue
        predicted_center = (tracked[index, :2] + tracked[index, 2:]) * 0.5
        scale = max(float(np.linalg.norm(reference[2:] - reference[:2])), 1.0)
        distances = np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) * 0.5 - predicted_center, axis=1) / scale
        nearest = int(np.argmin(distances - 0.1 * scores))
        if distances[nearest] <= 0.75:
            tracked[index] = boxes[nearest]
    return reference, tracked
