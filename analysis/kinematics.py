"""Geometric measurements and temporal landmark curves used by shot analysis."""

from dataclasses import dataclass

import numpy as np

from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata
from core.geometry import apply_affine_transform, finite_median_filter, json_values


_JOINT_KEYPOINTS = {
    "shoulder": ("hip", "shoulder", "elbow"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "wrist": ("elbow", "wrist", "hand_middle_1"),
    "hip": ("shoulder", "hip", "knee"),
    "knee": ("hip", "knee", "ankle"),
}
_MINIMUM_JOINT_ANGLES = {
    "shoulder": 0.0,
    "elbow": 30.0,
    "wrist": 0.0,
    "hip": 30.0,
    "knee": 20.0,
}
_JOINT_INDICES = {
    side: {
        name: np.asarray(
            [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in keypoints],
            dtype=np.int32,
        )
        for name, keypoints in _JOINT_KEYPOINTS.items()
    }
    for side in CONFIG["SIDE_NAMES"]
}
for _side_indices in _JOINT_INDICES.values():
    for _indices in _side_indices.values():
        _indices.setflags(write=False)


def contact_ball_center(
    observations: PoseObservations,
    tracked: TrackedPoses,
    frame: int,
    release_frame: int,
) -> tuple[np.ndarray | None, str | None]:
    """Recover a brief held-ball gap from observed ball-to-wrist offsets.

    Contact offsets require at least two nearby detections of the same hand.
    They are never extrapolated into the separated ball's flight.
    """
    center = observations.ball_centers_smoothed[frame]
    if np.isfinite(center).all():
        source = (
            "observed_ball_center"
            if np.isfinite(observations.ball_centers[frame]).all()
            else "interpolated_ball_center"
        )
        return center, source
    if frame > release_frame:
        return None, None
    nearby = np.flatnonzero(
        (
            np.abs(observations.timestamps - observations.timestamps[frame])
            <= CONFIG["BALL_HEIGHT_CONTACT_WINDOW_SECONDS"]
        )
        & (np.arange(len(observations.timestamps)) <= release_frame)
        & np.isfinite(observations.ball_centers).all(axis=1)
        & (observations.ball_scores >= CONFIG["BALL_DETECTION_THRESHOLD"])
    )
    candidates = []
    for side in CONFIG["SIDE_NAMES"]:
        wrist = CONFIG["KEYPOINT_INDEX"][f"{side}_wrist"]
        point = tracked.analysis_keypoints[frame, wrist, :2]
        if not np.isfinite(point).all() or not (observations.confidence[frame, wrist] >= CONFIG["KEYPOINT_THRESHOLD"]):
            continue
        offsets = []
        for sample in nearby:
            sample_point = tracked.analysis_keypoints[sample, wrist, :2]
            height = observations.boxes[sample, 3] - observations.boxes[sample, 1]
            if (
                not np.isfinite(sample_point).all() or not np.isfinite(height) or height <= 0
                or not (observations.confidence[sample, wrist] >= CONFIG["KEYPOINT_THRESHOLD"])
            ):
                continue
            offset = observations.ball_centers[sample] - sample_point
            if np.linalg.norm(offset) / height <= 0.2:
                offsets.append(offset)
        if len(offsets) >= 2:
            candidates.append((len(offsets), point + np.median(offsets, axis=0)))
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1], "hand_contact_offset"
    return None, None


def keypoint_velocities(keypoints: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-keypoint image-plane and model-depth velocities."""
    xy_velocity = np.full(keypoints.shape[:2] + (2,), np.nan, dtype=np.float32)
    z_velocity = np.full(keypoints.shape[:2], np.nan, dtype=np.float32)
    finite = np.isfinite(keypoints).all(axis=2)
    for frame in range(len(timestamps)):
        previous, following = frame - 1, frame + 1
        has_previous = np.zeros(keypoints.shape[1], dtype=bool)
        has_following = np.zeros_like(has_previous)
        if previous >= 0 and timestamps[frame] > timestamps[previous]:
            has_previous = finite[frame] & finite[previous]
        if following < len(timestamps) and timestamps[following] > timestamps[frame]:
            has_following = finite[frame] & finite[following]
        for mask, first, last in (
            (has_previous & has_following, previous, following),
            (has_previous & ~has_following, previous, frame),
            (has_following & ~has_previous, frame, following),
        ):
            if not np.any(mask):
                continue
            elapsed = float(timestamps[last] - timestamps[first])
            if elapsed <= 0:
                continue
            delta = (keypoints[last, mask] - keypoints[first, mask]) / elapsed
            xy_velocity[frame, mask] = delta[:, :2]
            z_velocity[frame, mask] = delta[:, 2]
    return xy_velocity, z_velocity


def _joint_angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float | None:
    """Return the interior angle in degrees for finite, nonzero vectors."""
    if not (np.isfinite(first).all() and np.isfinite(vertex).all() and np.isfinite(last).all()):
        return None
    left = first - vertex
    right = last - vertex
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def joint_angle_2d(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float | None:
    """Calculate an image-plane joint angle in degrees."""
    return _joint_angle(first[:2], vertex[:2], last[:2])


def joint_angle_3d(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float | None:
    """Calculate a three-dimensional joint angle in degrees."""
    return _joint_angle(first, vertex, last)


def pose_joint_angle_for_frame(
    observations: PoseObservations,
    tracked: TrackedPoses,
    frame_index: int,
    side: str,
    name: str,
) -> tuple[float | None, str]:
    """Measure a joint angle and identify the coordinate source used."""
    return pose_joint_angle_from_arrays(
        tracked.analysis_keypoints[frame_index],
        observations.boxes[frame_index],
        observations.confidence[frame_index],
        observations.raw_keypoints[frame_index],
        tracked.valid[frame_index],
        side,
        name,
    )


def pose_joint_angle_from_arrays(
    points: np.ndarray,
    bbox: np.ndarray,
    confidence: np.ndarray,
    raw_keypoints: np.ndarray,
    valid: np.ndarray,
    side: str,
    name: str,
) -> tuple[float | None, str]:
    """Measure one joint angle from a frame's filtered and raw pose arrays."""
    if name not in _JOINT_KEYPOINTS:
        return None, "unsupported_joint"
    indices = _JOINT_INDICES[side][name]
    angle_2d = joint_angle_2d(points[indices[0]], points[indices[1]], points[indices[2]])
    minimum = _MINIMUM_JOINT_ANGLES[name]
    if angle_2d is not None and angle_2d >= minimum:
        return angle_2d, "2d"

    confidence = confidence[indices]
    if (
        np.isfinite(bbox).all()
        and bbox[3] > bbox[1]
        and valid[indices].all()
        and np.isfinite(confidence).all()
        and np.all(confidence >= CONFIG["KEYPOINT_THRESHOLD"])
        and np.isfinite(points[indices]).all()
    ):
        box_height = float(bbox[3] - bbox[1])
        # Use unadjusted model depth; visualization continuity can select a
        # different depth bin from small backend score changes.
        raw_depth = raw_keypoints[indices, 2]
        if not np.isfinite(raw_depth).all():
            return angle_2d, "2d_invalid" if angle_2d is not None else "unavailable"
        # RTMW3D's depth axis is in its configured normalized range. Scaling
        # x/y by the person box height puts all three axes in comparable units.
        points_3d = np.column_stack(
            (
                points[indices, 0] / box_height,
                raw_depth / CONFIG["RTMW3D_CONFIG"]["z_range"],
                -points[indices, 1] / box_height,
            )
        )
        angle_3d = joint_angle_3d(points_3d[0], points_3d[1], points_3d[2])
        if angle_3d is not None and minimum <= angle_3d <= 180.0:
            return angle_3d, "3d_fallback"
    return angle_2d, "2d_invalid" if angle_2d is not None else "unavailable"


def body_height_reference_from_frames(
    observations: PoseObservations,
    tracked: TrackedPoses,
    metadata: VideoMetadata,
    camera_shift_y: np.ndarray,
    baseline_frames: np.ndarray,
    measurement_frames: list[int],
    camera_transform_to_reference_xy: np.ndarray | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Measure body height from stable foot and face landmarks before a shot phase."""
    baseline_frames = np.asarray(baseline_frames, dtype=np.int32)
    baseline_feet = []
    for frame in baseline_frames:
        box = observations.boxes[frame]
        if not np.isfinite(box).all() or box[3] >= metadata.height - 1:
            continue
        frame_points = tracked.analysis_keypoints[frame]
        foot_values = []
        for side in CONFIG["SIDE_NAMES"]:
            indices = CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"][side]
            points = frame_points[indices, :2]
            visible = (
                tracked.valid[frame, indices]
                & np.isfinite(points).all(axis=1)
                & (observations.confidence[frame, indices] >= CONFIG["KEYPOINT_THRESHOLD"])
                & (points[:, 0] >= 0)
                & (points[:, 0] < metadata.width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < metadata.height - 1)
            )
            if np.any(visible):
                visible_points = points[visible]
                if camera_transform_to_reference_xy is not None:
                    visible_points = apply_affine_transform(
                        visible_points,
                        camera_transform_to_reference_xy[frame],
                    )
                foot_values.append(float(np.max(visible_points[:, 1])))
        if len(foot_values) == 2:
            baseline_feet.append(
                float(np.mean(foot_values))
                if camera_transform_to_reference_xy is not None
                else float(np.mean(foot_values)) - camera_shift_y[frame]
            )

    if not baseline_feet:
        return None, None, None
    ground_y = float(np.median(baseline_feet))

    required_frames = np.unique(
        np.asarray(
            [*baseline_frames, *measurement_frames],
            dtype=np.int32,
        )
    )
    face_candidates = []
    for index in range(5):
        visible = (
            tracked.valid[:, index]
            & (observations.confidence[:, index] >= CONFIG["KEYPOINT_THRESHOLD"])
            & np.isfinite(tracked.analysis_keypoints[:, index, 1])
        )
        face_candidates.append((int(np.count_nonzero(visible[required_frames])), -index, visible))
    _, negative_index, visible = max(face_candidates, key=lambda candidate: candidate[:2])
    face_index = -negative_index
    head_y = []
    for frame in baseline_frames:
        if not visible[frame]:
            continue
        head_point = tracked.analysis_keypoints[frame, face_index, :2]
        if camera_transform_to_reference_xy is not None:
            head_point = apply_affine_transform(
                head_point,
                camera_transform_to_reference_xy[frame],
            )
        else:
            head_point = head_point.copy()
            head_point[1] -= camera_shift_y[frame]
        head_y.append(float(head_point[1]))
    if not head_y:
        return None, ground_y, ground_y
    body_height = ground_y - float(np.median(head_y))
    if body_height <= 0:
        return None, ground_y, ground_y
    return body_height, ground_y, ground_y


def elbow_angular_velocity_between_frames(
    observations: PoseObservations,
    tracked: TrackedPoses,
    side: str,
    start_frame: int,
    release_frame: int,
) -> tuple[float | None, int | None, int | None, float | None, float | None]:
    """Calculate mean elbow-angle speed from lift through release.

    Frame arguments use zero-based source indices. If the start-frame angle is
    unavailable within one frame, a bounded wider window supplies the first and
    last valid samples near the interval; their timestamps define elapsed time.
    """
    frame_count = len(observations.timestamps)
    if not (0 <= start_frame < release_frame < frame_count):
        return None, None, None, None, None

    fallback_radius = CONFIG["JOINT_ANGLE_METRIC_FALLBACK_FRAMES"]
    start_candidates = sorted(
        range(max(0, start_frame - 1), min(release_frame, start_frame + 2)),
        key=lambda frame: (abs(frame - start_frame), frame),
    )
    sampled_start = None
    start_angle = None
    for frame in start_candidates:
        angle, source = pose_joint_angle_for_frame(observations, tracked, frame, side, "elbow")
        if angle is not None and source != "2d_invalid":
            sampled_start, start_angle = frame, angle
            break

    used_extended_start_window = sampled_start is None
    if used_extended_start_window:
        for frame in range(
            max(0, start_frame - fallback_radius),
            min(release_frame, start_frame + fallback_radius + 1),
        ):
            angle, source = pose_joint_angle_for_frame(observations, tracked, frame, side, "elbow")
            if angle is not None and source != "2d_invalid":
                sampled_start, start_angle = frame, angle
                break

    release_angle, release_source = pose_joint_angle_for_frame(observations, tracked, release_frame, side, "elbow")
    sampled_end = release_frame if release_angle is not None and release_source != "2d_invalid" else None
    if used_extended_start_window and sampled_start is not None:
        end_candidates = range(
            max(sampled_start + 1, release_frame - fallback_radius),
            min(frame_count, release_frame + fallback_radius + 1),
        )
        for frame in end_candidates:
            angle, source = pose_joint_angle_for_frame(observations, tracked, frame, side, "elbow")
            if angle is not None and source != "2d_invalid":
                sampled_end, release_angle = frame, angle
    elif sampled_end is None and sampled_start is not None:
        end_candidates = sorted(
            range(max(sampled_start + 1, release_frame - 1), min(frame_count, release_frame + 2)),
            key=lambda frame: (abs(frame - release_frame), frame),
        )
        for frame in end_candidates:
            angle, source = pose_joint_angle_for_frame(observations, tracked, frame, side, "elbow")
            if angle is not None and source != "2d_invalid":
                sampled_end, release_angle = frame, angle
                break

    if sampled_start is None or start_angle is None or sampled_end is None or release_angle is None:
        return None, sampled_start, sampled_end, start_angle, release_angle

    elapsed = float(observations.timestamps[sampled_end] - observations.timestamps[sampled_start])
    if elapsed <= 0:
        return None, sampled_start, sampled_end, start_angle, release_angle
    return (
        float(abs(release_angle - start_angle) / elapsed),
        sampled_start,
        sampled_end,
        start_angle,
        release_angle,
    )


def joint_angles_2d(keypoints: np.ndarray) -> dict[str, float | None]:
    """Calculate supported two-dimensional joint angles for one pose."""
    angles = {}
    for side in CONFIG["SIDE_NAMES"]:
        shoulder = CONFIG["KEYPOINT_INDEX"][f"{side}_shoulder"]
        elbow = CONFIG["KEYPOINT_INDEX"][f"{side}_elbow"]
        wrist = CONFIG["KEYPOINT_INDEX"][f"{side}_wrist"]
        hip = CONFIG["KEYPOINT_INDEX"][f"{side}_hip"]
        knee = CONFIG["KEYPOINT_INDEX"][f"{side}_knee"]
        ankle = CONFIG["KEYPOINT_INDEX"][f"{side}_ankle"]
        toe = [CONFIG["KEYPOINT_INDEX"][f"{side}_{name}"] for name in CONFIG["TOE_NAMES"]]
        toe_points = keypoints[toe]
        visible_toes = np.isfinite(toe_points[:, :2]).all(axis=1)
        toe_center = np.full(3, np.nan)
        if visible_toes.any():
            toe_center[:2] = np.mean(toe_points[visible_toes, :2], axis=0)
        hand_base = CONFIG["HAND_BASES"][side]
        middle_finger_base = hand_base + CONFIG["HAND_FINGER_BASE_OFFSETS"]["middle"]
        body_triplets = {
            "shoulder": (hip, shoulder, elbow),
            "elbow": (shoulder, elbow, wrist),
            "wrist": (elbow, wrist, middle_finger_base),
            "hip": (shoulder, hip, knee),
            "knee": (hip, knee, ankle),
        }
        for name, (first, vertex, last) in body_triplets.items():
            angles[f"{side}_{name}"] = joint_angle_2d(keypoints[first], keypoints[vertex], keypoints[last])
        angles[f"{side}_ankle"] = joint_angle_2d(keypoints[knee], keypoints[ankle], toe_center)
        for finger, first_offset in CONFIG["HAND_FINGER_BASE_OFFSETS"].items():
            first = hand_base + first_offset
            for joint in range(CONFIG["HAND_JOINTS_PER_FINGER"] - 1):
                angles[f"{side}_{finger}_{joint + 1}"] = joint_angle_2d(
                    keypoints[hand_base] if joint == 0 else keypoints[first + joint - 1],
                    keypoints[first + joint],
                    keypoints[first + joint + 1],
                )
    return angles


def highest_head_keypoint(keypoints: np.ndarray) -> dict | None:
    """Return the highest visible head landmark in a pose."""
    head_indices = np.asarray([*range(5), *range(CONFIG["FACE_START_INDEX"], CONFIG["FACE_END_INDEX"])], dtype=np.int32)
    visible = np.isfinite(keypoints[head_indices, :2]).all(axis=1)
    if not visible.any():
        return None
    candidates = head_indices[visible]
    index = int(candidates[np.argmin(keypoints[candidates, 1])])
    return {
        "keypoint": CONFIG["KEYPOINT_NAMES"][index],
        "index": index,
        "xyz": json_values(keypoints[index]),
    }


def ground_metrics(
    bbox: np.ndarray, keypoints: np.ndarray, width: int, height: int
) -> tuple[dict | None, float | None, float | None, dict | None]:
    """Measure the visible ground point and body height for a frame."""
    head_top = highest_head_keypoint(keypoints)
    if not np.isfinite(bbox).all():
        return None, None, None, head_top
    ground_x = float(np.clip((bbox[0] + bbox[2]) * 0.5, 0, width - 1))
    ground_y = float(np.clip(bbox[3], 0, height - 1))
    ground = {
        "xy_px": [ground_x, ground_y],
        "y_normalized": ground_y / max(height - 1, 1),
        "source": "person_detection_box_bottom",
    }
    visible_feet = [index for index in CONFIG["FOOT_KEYPOINT_INDICES"] if np.isfinite(keypoints[index, :2]).all()]
    if visible_feet:
        lowest_index = max(visible_feet, key=lambda index: keypoints[index, 1])
        ground["lowest_visible_foot"] = {
            "keypoint": CONFIG["KEYPOINT_NAMES"][lowest_index],
            "xy_px": json_values(keypoints[lowest_index, :2]),
        }
        ground["bbox_bottom_minus_foot_px"] = max(ground_y - float(keypoints[lowest_index, 1]), 0.0)
    if head_top is None:
        return ground, None, None, None
    head_y = float(head_top["xyz"][1])
    height_px = max(ground_y - head_y, 0.0)
    return ground, height_px, height_px / max(height - 1, 1), head_top


def ball_trajectory_angle(
    centers: np.ndarray,
    timestamps: np.ndarray,
    frame_index: int,
    before: int,
    after: int,
    launch: bool,
    boxes: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[float | None, list[int]]:
    """Estimate the ball's direction from stable samples around an event."""
    first = max(0, frame_index - before)
    last = min(len(centers), frame_index + after + 1)
    indices = np.arange(first, last)
    valid = np.isfinite(centers[indices]).all(axis=1) & np.isfinite(timestamps[indices])
    weights = np.ones(len(indices), dtype=np.float32)
    if confidence is not None:
        scores = confidence[indices]
        valid &= np.isfinite(scores) & (scores >= CONFIG["BALL_DETECTION_THRESHOLD"])
        weights *= np.clip(scores, 0.0, 1.0) ** 2
    if boxes is not None:
        dimensions = boxes[indices, 2:] - boxes[indices, :2]
        valid &= np.isfinite(dimensions).all(axis=1) & np.all(dimensions > 0, axis=1)
        aspect_ratio = np.divide(
            np.minimum(dimensions[:, 0], dimensions[:, 1]),
            np.maximum(dimensions[:, 0], dimensions[:, 1]),
            out=np.zeros(len(indices), dtype=np.float32),
            where=np.maximum(dimensions[:, 0], dimensions[:, 1]) > 0,
        )
        clear = valid & (aspect_ratio >= CONFIG["RELEASE_ANGLE_MIN_BOX_ASPECT_RATIO"])
        if np.count_nonzero(clear) >= 2:
            valid = clear
        weights *= np.sqrt(np.clip(aspect_ratio, 0.0, 1.0))
    indices = indices[valid]
    if len(indices) < 2:
        return None, indices.tolist()
    times = timestamps[indices] - timestamps[frame_index]
    if np.ptp(times) <= 0:
        return None, indices.tolist()
    sample_centers = centers[indices].copy()
    if boxes is not None and image_width is not None and image_height is not None:
        for sample_index, detection_index in enumerate(indices):
            left, top, right, bottom = boxes[detection_index]
            box_width, box_height = right - left, bottom - top
            if box_width <= 0 or box_height <= 0:
                continue
            # A clipped ball box has a biased center. Recover the missing half
            # from the un-clipped dimension, using the ball's near-circular shape.
            if top <= 1 and bottom < image_height - 1 and box_width > box_height:
                sample_centers[sample_index, 1] = bottom - box_width * 0.5
            elif bottom >= image_height - 1 and top > 1 and box_width > box_height:
                sample_centers[sample_index, 1] = top + box_width * 0.5
            if left <= 1 and right < image_width - 1 and box_height > box_width:
                sample_centers[sample_index, 0] = right - box_height * 0.5
            elif right >= image_width - 1 and left > 1 and box_height > box_width:
                sample_centers[sample_index, 0] = left + box_height * 0.5

    base_weights = np.sqrt(weights[valid])
    fit_weights = base_weights.copy()
    for iteration in range(3):
        horizontal_fit = np.polyfit(times, sample_centers[:, 0], 1, w=fit_weights)
        vertical_fit = np.polyfit(times, sample_centers[:, 1], 1, w=fit_weights)
        if iteration == 2 or len(indices) < 3:
            break
        residual = np.hypot(
            sample_centers[:, 0] - np.polyval(horizontal_fit, times),
            sample_centers[:, 1] - np.polyval(vertical_fit, times),
        )
        residual_median = float(np.median(residual))
        residual_scale = 1.4826 * float(np.median(np.abs(residual - residual_median)))
        if not np.isfinite(residual_scale) or residual_scale < 1.0:
            break
        huber_weights = np.minimum(
            1.0,
            1.345 * residual_scale / np.maximum(residual, 1e-6),
        )
        fit_weights = base_weights * np.sqrt(huber_weights)
    horizontal_speed = float(horizontal_fit[0])
    vertical_speed = float(vertical_fit[0])
    if launch:
        vertical_speed = -vertical_speed
    return float(np.degrees(np.arctan2(vertical_speed, abs(horizontal_speed)))), indices.tolist()


def entry_trajectory_angle(
    observations: PoseObservations,
    frame_index: int,
    *,
    camera_compensation_enabled: bool = False,
) -> tuple[float | None, list[int]]:
    """Average tracked incoming directions at entry and the preceding two frames.

    Each direction uses a backward displacement, requiring four centers for
    three angles. Reject discontinuous speeds and angles before averaging;
    fewer than two reliable final directions do not establish an entry angle.
    """
    if not 0 <= frame_index < len(observations.timestamps):
        return None, []
    count = CONFIG["ENTRY_ANGLE_SAMPLE_FRAMES"]
    times = observations.timestamps
    first = max(0, min(
        int(np.searchsorted(times, times[frame_index] - CONFIG["ENTRY_ANGLE_TRACK_WINDOW_SECONDS"])),
        frame_index - 2 * count - 1,
    ))
    indices = np.arange(first, frame_index + 1)
    centers = observations.ball_centers[indices].astype(float, copy=True)
    if camera_compensation_enabled:
        centers -= observations.camera_shift_xy[indices]
    boxes = observations.ball_boxes[indices]
    sizes = boxes[:, 2:] - boxes[:, :2]
    scores = observations.ball_scores[indices]
    valid = (
        np.isfinite(centers).all(axis=1) & np.isfinite(scores)
        & (scores >= CONFIG["BALL_DETECTION_THRESHOLD"])
        & np.isfinite(sizes).all(axis=1) & (sizes > 0).all(axis=1)
    )
    radii = np.mean(sizes, axis=1) / 2
    ratio = CONFIG["ENTRY_ANGLE_MAX_SPEED_RATIO"]
    if np.any(valid):
        radius = np.median(radii[valid])
        valid &= (radii >= radius / ratio) & (radii <= radius * ratio)
    observed = np.flatnonzero(valid)
    previous, following = observed[:-1], observed[1:]
    gap = indices[following] - indices[previous]
    delta = centers[following] - centers[previous]
    elapsed = times[indices[following]] - times[indices[previous]]
    displacement = np.linalg.norm(delta, axis=1)
    valid_steps = (
        (gap <= CONFIG["ENTRY_ANGLE_MAX_GAP_FRAMES"]) & np.isfinite(elapsed) & (elapsed > 0)
        & (displacement >= CONFIG["ENTRY_ANGLE_MIN_DISPLACEMENT_PX"]) & (delta[:, 1] >= 0)
    )
    if not np.any(valid_steps):
        return None, []
    speeds = np.divide(displacement, elapsed, out=np.full_like(displacement, np.nan), where=elapsed > 0)
    speed = float(np.median(speeds[valid_steps]))
    valid_steps &= (speeds >= speed / ratio) & (speeds <= speed * ratio)
    if not np.any(valid_steps):
        return None, []
    angles = np.degrees(np.arctan2(delta[:, 1], np.abs(delta[:, 0])))
    initial_angle = float(np.median(angles[valid_steps]))
    history = []
    selected = []
    values = []
    for offset in np.flatnonzero(valid_steps):
        index = int(indices[following[offset]])
        recent = [angle for frame, angle in history if index - frame <= count]
        reference = float(np.median(recent)) if recent else initial_angle
        angle = float(angles[offset])
        if abs(angle - reference) > CONFIG["ENTRY_ANGLE_MAX_CHANGE_DEG"] * gap[offset]:
            continue
        history.append((index, angle))
        # A short gap uses the observed secant direction; no center is invented.
        for sample in range(max(int(indices[previous[offset]]) + 1, frame_index - count + 1), index + 1):
            selected.append(sample)
            values.append(angle)
    return (float(np.mean(values)) if len(values) >= 2 else None), selected


@dataclass(slots=True)
class JumpCurveAnalysis:
    """Foot-derived jump curve and release-height references."""

    ground_y_px: float
    values: np.ndarray
    release_lift_ratio: float | None
    minimum_lift_ratio: float
    confirmed_lift_ratio: float


@dataclass(slots=True)
class TorsoJumpAnalysis:
    """Fallback torso-based jump curve and its estimated peak."""

    curve: np.ndarray
    peak_frame: int | None = None
    peak_height_ratio: float | None = None


def build_jump_curve(
    foot_positions_y: np.ndarray,
    grounded_foot_positions_y: np.ndarray,
    body_height_px: float,
    fps: float,
    release_frame: int,
) -> JumpCurveAnalysis:
    """Build a finite-aware normalized jump curve from foot positions."""
    ground_y = float(np.percentile(grounded_foot_positions_y, 95))
    release_lift_ratio = None
    initial_feet = finite_median_filter(foot_positions_y, radius=max(1, round(fps * 0.1)))
    release_feet = initial_feet[release_frame : release_frame + 2]
    release_feet = release_feet[np.isfinite(release_feet)]
    if len(release_feet):
        release_foot_y = float(np.mean(release_feet))
        release_lift_ratio = float(max(ground_y - release_foot_y, 0.0) / body_height_px)

    minimum_lift_ratio = max(4.0 / body_height_px, CONFIG["MOTION_MIN_LIFT_RATIO"])
    confirmed_lift_ratio = max(4.0 / body_height_px, CONFIG["JUMP_CONFIRM_HEIGHT_RATIO"])
    smoothed_feet = finite_median_filter(foot_positions_y, radius=max(1, round(fps * 0.08)))
    values = np.full(len(foot_positions_y), np.nan, dtype=np.float32)
    if len(grounded_foot_positions_y) >= 3:
        values = np.maximum((ground_y - smoothed_feet) / body_height_px, 0.0)
        values[~np.isfinite(smoothed_feet)] = np.nan
        values = finite_median_filter(values)

    release_samples = values[release_frame : release_frame + 2]
    release_samples = release_samples[np.isfinite(release_samples)]
    if len(release_samples):
        release_lift_ratio = float(np.mean(release_samples))

    return JumpCurveAnalysis(
        ground_y_px=ground_y,
        values=values,
        release_lift_ratio=release_lift_ratio,
        minimum_lift_ratio=minimum_lift_ratio,
        confirmed_lift_ratio=confirmed_lift_ratio,
    )


def estimate_torso_jump(
    observations: PoseObservations,
    camera_shift_y: np.ndarray,
    anchor_frames: list[int],
    knee_frame: int | None,
    start_frame: int,
    end_frame: int,
    body_height_px: float,
    filtered_torso_y: np.ndarray,
    camera_transform_to_reference_xy: np.ndarray | None = None,
) -> TorsoJumpAnalysis:
    """Estimate a jump peak from torso height when foot landmarks are unavailable."""
    curve = filtered_torso_y
    anchor_values = curve[anchor_frames] if anchor_frames else np.empty(0)
    anchor_values = anchor_values[np.isfinite(anchor_values)]
    candidates = np.arange(start_frame, end_frame)
    candidates = candidates[np.isfinite(curve[candidates])]
    if len(anchor_values) < 3 or len(candidates) < 3:
        hip_indices = np.asarray(
            [CONFIG["KEYPOINT_INDEX"][f"{side}_hip"] for side in CONFIG["SIDE_NAMES"]]
        )
        shoulder_indices = np.asarray(
            [CONFIG["KEYPOINT_INDEX"][f"{side}_shoulder"] for side in CONFIG["SIDE_NAMES"]]
        )
        observed_torso_y = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
        for index, frame_points in enumerate(observations.visualization_keypoints):
            torso_samples = []
            for joint_indices in (hip_indices, shoulder_indices):
                visible = (
                    np.isfinite(frame_points[joint_indices, :2]).all(axis=1)
                    & (observations.confidence[index, joint_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
                )
                if np.any(visible):
                    joint_points = frame_points[joint_indices[visible], :2]
                    if camera_transform_to_reference_xy is not None:
                        joint_points = apply_affine_transform(
                            joint_points,
                            camera_transform_to_reference_xy[index],
                        )
                    else:
                        joint_points = joint_points.copy()
                        joint_points[:, 1] -= camera_shift_y[index]
                    torso_samples.append(
                        float(np.median(joint_points[:, 1]))
                    )
            if torso_samples:
                observed_torso_y[index] = float(np.mean(torso_samples))
        curve = finite_median_filter(observed_torso_y)
        anchor_values = curve[anchor_frames] if anchor_frames else np.empty(0)
        anchor_values = anchor_values[np.isfinite(anchor_values)]
        candidates = np.arange(start_frame, end_frame)
        candidates = candidates[np.isfinite(curve[candidates])]

    reference_y = (
        float(np.median(anchor_values))
        if len(anchor_values)
        else curve[knee_frame]
        if knee_frame is not None
        else np.nan
    )
    result = TorsoJumpAnalysis(curve=curve)
    if not np.isfinite(reference_y) or not len(candidates):
        return result

    jump_curve = np.maximum((reference_y - curve) / body_height_px, 0.0)
    peak_frame = int(candidates[np.argmax(jump_curve[candidates])])
    peak_height_ratio = float(jump_curve[peak_frame])
    minimum_lift_ratio = max(4.0 / body_height_px, CONFIG["MOTION_MIN_LIFT_RATIO"])
    if peak_height_ratio >= minimum_lift_ratio:
        result.peak_frame = peak_frame
        result.peak_height_ratio = peak_height_ratio
    return result
