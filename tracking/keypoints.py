"""Filter and validate pose trajectories over time."""

import numpy as np

from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata
from core.geometry import SKELETON_EDGES


def track_keypoint_sequence(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    boxes: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject low-confidence outliers using adaptive trajectories and hand-chain lengths.

    The raw model predictions are never changed. The returned mask identifies joints that
    remain usable for rendering after comparing each joint with its own sequence and
    hand/finger links with their clip-specific length distributions. Coordinates are
    measured relative to the tracked person box so the limits adapt to image resolution,
    player scale, and camera motion.
    """
    tracked = keypoints.copy()
    valid = np.isfinite(keypoints).all(axis=2) & np.isfinite(confidence) & (confidence > 0.0)
    rejection_reason = np.zeros(valid.shape, dtype=np.uint8)
    rejection_reason[~valid] = 4  # No usable raw prediction.
    box_height = boxes[:, 3] - boxes[:, 1]
    box_center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    valid_box = np.isfinite(boxes).all(axis=1) & (box_height > 0.0)
    valid &= valid_box[:, None]
    rejection_reason[~valid] = 4
    if not np.any(valid_box) or not np.any(valid):
        tracked[~valid] = np.nan
        return tracked, valid, rejection_reason

    relative_xy = np.full(keypoints.shape[:2] + (2,), np.nan, dtype=np.float32)
    relative_xy[valid] = (
        keypoints[:, :, :2][valid] - np.broadcast_to(box_center[:, None, :], keypoints.shape[:2] + (2,))[valid]
    ) / np.broadcast_to(box_height[:, None, None], keypoints.shape[:2] + (2,))[valid]
    relative_pose = np.concatenate([relative_xy, (keypoints[:, :, 2:3] / CONFIG["RTMW3D_CONFIG"]["z_range"])], axis=2)
    relative_pose[~valid] = np.nan
    present_frames = np.flatnonzero(valid_box)
    boundaries = np.flatnonzero(np.diff(present_frames) > 1) + 1
    segments = np.split(present_frames, boundaries)
    joint_count = keypoints.shape[1]
    confidence_median = np.full(keypoints.shape[:2], np.nan, dtype=np.float32)
    reliable = np.zeros_like(valid)

    def interpolate_observation(left: int, right: int, joint: int, alpha: float) -> tuple[np.ndarray, np.generic]:
        left_weight = 1.0 - alpha
        return (
            relative_pose[left, joint] * left_weight + relative_pose[right, joint] * alpha,
            confidence[left, joint] * left_weight + confidence[right, joint] * alpha,
        )

    # Compare every joint's normalized 3D trajectory with its reliable neighbors.
    # Per-joint baselines keep fast-moving limbs from setting thresholds for the body.
    for segment in segments:
        for joint in range(joint_count):
            samples = confidence[segment[valid[segment, joint]], joint]
            if not len(samples):
                continue
            median = float(np.median(samples))
            mad = float(np.median(np.abs(samples - median))) * 1.4826
            confidence_median[segment, joint] = median
            confidence_floor = max(median - 1.5 * mad, 0.8 * median)
            reliable[segment, joint] = valid[segment, joint] & (confidence[segment, joint] >= confidence_floor)

            anchors = segment[reliable[segment, joint]]
            pixel_floor = 1.0 / max(float(np.median(box_height[segment])), 1.0)
            quantization_floor = np.array(
                [pixel_floor, pixel_floor, 2.0 / CONFIG["RTMW3D_CONFIG"]["input_size_whd"][2]]
            )
            local_predictions = []
            local_residuals = []
            local_confidence_residuals = []
            for pos, frame in enumerate(segment):
                if not valid[frame, joint]:
                    continue
                nearby = segment[max(0, pos - 2) : pos + 3]
                neighbors = nearby[(nearby != frame) & valid[nearby, joint]]
                if len(neighbors) < 2:
                    continue
                before, after = neighbors[neighbors < frame], neighbors[neighbors > frame]
                if len(before) and len(after):
                    expected = np.median(relative_pose[neighbors, joint], axis=0)
                    expected_confidence = float(np.median(confidence[neighbors, joint]))
                else:
                    left, right = neighbors[:2] if not len(before) else neighbors[-2:]
                    span = float(timestamps[right] - timestamps[left])
                    if span <= 0:
                        continue
                    alpha = float(timestamps[frame] - timestamps[left]) / span
                    expected, expected_confidence = interpolate_observation(int(left), int(right), joint, alpha)
                local_predictions.append((int(frame), expected, expected_confidence))
                local_residuals.append(np.abs(relative_pose[frame, joint] - expected))
                local_confidence_residuals.append(abs(float(confidence[frame, joint]) - expected_confidence))

            local_prediction_by_frame = {frame: prediction for frame, prediction, _ in local_predictions}
            local_limit = local_confidence_limit = None
            if local_residuals:
                local_residuals = np.asarray(local_residuals)
                residual_center = np.median(local_residuals, axis=0)
                residual_scale = 1.4826 * np.median(np.abs(local_residuals - residual_center), axis=0)
                local_limit = residual_center + 6.0 * residual_scale + quantization_floor
                confidence_residuals = np.asarray(local_confidence_residuals)
                confidence_center = float(np.median(confidence_residuals))
                confidence_scale = 1.4826 * float(np.median(np.abs(confidence_residuals - confidence_center)))
                local_confidence_limit = confidence_center + 6.0 * confidence_scale

            # Reliable anchors estimate a per-joint baseline; local predictions
            # cover track ends or low-confidence spans without reliable anchors.
            if len(anchors) >= 3:
                anchor_residuals = []
                anchor_confidence_residuals = []
                for pos in range(1, len(anchors) - 1):
                    left, current, right = anchors[pos - 1], anchors[pos], anchors[pos + 1]
                    dt = float(timestamps[right] - timestamps[left])
                    if dt <= 0:
                        continue
                    alpha = float(timestamps[current] - timestamps[left]) / dt
                    expected, expected_confidence = interpolate_observation(left, right, joint, alpha)
                    anchor_residuals.append(np.abs(relative_pose[current, joint] - expected))
                    anchor_confidence_residuals.append(abs(float(confidence[current, joint] - expected_confidence)))
                if anchor_residuals:
                    residual_limit = np.quantile(anchor_residuals, 0.99, axis=0) + quantization_floor
                    confidence_drop_limit = float(np.quantile(anchor_confidence_residuals, 0.99))

                    for frame in segment[valid[segment, joint]]:
                        left_anchors = anchors[(anchors < frame) & valid[anchors, joint]]
                        right_anchors = anchors[(anchors > frame) & valid[anchors, joint]]
                        if not len(left_anchors) or not len(right_anchors):
                            continue
                        left, right = int(left_anchors[-1]), int(right_anchors[0])
                        span = float(timestamps[right] - timestamps[left])
                        if span <= 0:
                            continue
                        alpha = float(timestamps[frame] - timestamps[left]) / span
                        expected, expected_confidence = interpolate_observation(left, right, joint, alpha)
                        residual = np.abs(relative_pose[frame, joint] - expected)
                        confidence_drop = expected_confidence - float(confidence[frame, joint])
                        local_prediction = local_prediction_by_frame.get(int(frame))
                        local_outlier = (
                            local_prediction is not None
                            and local_limit is not None
                            and np.any(np.abs(relative_pose[frame, joint] - local_prediction) > 3.0 * local_limit)
                        )
                        if np.any(residual > residual_limit) and (
                            (confidence_drop > confidence_drop_limit and local_outlier)
                            or np.any(residual > 2.5 * residual_limit)
                        ):
                            valid[frame, joint] = False
                            rejection_reason[frame, joint] = 1

            if local_limit is not None:
                for frame, expected, expected_confidence in local_predictions:
                    left_anchors = anchors[(anchors < frame) & valid[anchors, joint]]
                    right_anchors = anchors[(anchors > frame) & valid[anchors, joint]]
                    if (len(left_anchors) and len(right_anchors)) or not valid[frame, joint]:
                        continue
                    residual = np.abs(relative_pose[frame, joint] - expected)
                    confidence_drop = expected_confidence - float(confidence[frame, joint])
                    if np.any(residual > local_limit) and (
                        confidence_drop > local_confidence_limit or np.any(residual > 3.0 * local_limit)
                    ):
                        valid[frame, joint] = False
                        rejection_reason[frame, joint] = 1

    # Hand/finger links are short relative to the person box. Their clip-specific
    # distribution catches isolated finger peaks without imposing pixel thresholds.
    for segment in segments:
        pixel_floor = 1.0 / max(float(np.median(box_height[segment])), 1.0)
        for start, end in SKELETON_EDGES:
            if start not in CONFIG["HAND_KEYPOINT_INDICES"] or end not in CONFIG["HAND_KEYPOINT_INDICES"]:
                continue
            pair = reliable[segment, start] & reliable[segment, end] & valid[segment, start] & valid[segment, end]
            lengths = np.linalg.norm(relative_xy[segment, start] - relative_xy[segment, end], axis=1)
            samples = lengths[pair & np.isfinite(lengths)]
            if len(samples) < 4:
                continue
            lower_quartile, upper_quartile = np.percentile(samples, [25.0, 75.0])
            upper = float(upper_quartile + 3.0 * (upper_quartile - lower_quartile)) + pixel_floor
            for local_index, frame in enumerate(segment):
                if not valid[frame, start] or not valid[frame, end]:
                    continue
                if not np.isfinite(lengths[local_index]) or lengths[local_index] <= upper:
                    continue
                low_confidence = [
                    joint for joint in (start, end) if confidence[frame, joint] < confidence_median[frame, joint]
                ]
                if low_confidence:
                    rejected_joint = min(low_confidence, key=lambda j: confidence[frame, j])
                    valid[frame, rejected_joint] = False
                    rejection_reason[frame, rejected_joint] = 2

    # Do not draw detached fingers: a missing wrist or finger joint invalidates the
    # downstream part of that hand chain for this rendered frame.
    for frame in range(len(valid)):
        for base in CONFIG["HAND_BASES"].values():
            if not valid[frame, base]:
                hand_slice = slice(base, base + CONFIG["HAND_JOINT_COUNT"])
                valid[frame, hand_slice] = False
                rejection_reason[frame, hand_slice] = np.where(
                    rejection_reason[frame, hand_slice] == 0,
                    3,
                    rejection_reason[frame, hand_slice],
                )
                continue
            for first in CONFIG["HAND_FINGER_BASE_OFFSETS"].values():
                parent = base
                for offset in range(first, first + CONFIG["HAND_JOINTS_PER_FINGER"]):
                    joint = base + offset
                    if not valid[frame, parent]:
                        valid[frame, joint] = False
                        if rejection_reason[frame, joint] == 0:
                            rejection_reason[frame, joint] = 3
                    parent = joint

    tracked[~valid] = np.nan
    return tracked, valid, rejection_reason


def filter_pose_observations(observations: PoseObservations, metadata: VideoMetadata) -> TrackedPoses:
    """Apply temporal filtering and prepare the pose arrays used downstream."""
    tracked_keypoints, track_valid, _ = track_keypoint_sequence(
        observations.visualization_keypoints,
        observations.confidence,
        observations.boxes,
        observations.timestamps,
    )
    visualization_keypoints = tracked_keypoints.copy()
    analysis_keypoints = tracked_keypoints.copy()
    analysis_keypoints[
        ~np.isfinite(observations.confidence) | (observations.confidence < CONFIG["KEYPOINT_THRESHOLD"])
    ] = np.nan
    outside_image = (
        (analysis_keypoints[:, :, 0] < -0.01 * metadata.width)
        | (analysis_keypoints[:, :, 0] > metadata.width * 1.01)
        | (analysis_keypoints[:, :, 1] < -0.01 * metadata.height)
        | (analysis_keypoints[:, :, 1] > metadata.height * 1.01)
    )
    analysis_keypoints[outside_image] = np.nan
    clipped_feet = observations.boxes[:, 3] >= metadata.height - 1
    analysis_keypoints[np.ix_(clipped_feet, CONFIG["FOOT_KEYPOINT_INDICES"])] = np.nan

    return TrackedPoses(
        keypoints=tracked_keypoints,
        valid=track_valid,
        visualization_keypoints=visualization_keypoints,
        analysis_keypoints=analysis_keypoints,
    )
