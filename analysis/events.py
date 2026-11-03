"""Detect ball, pose, and shooting-hand events from frame observations."""

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

from analysis.kinematics import entry_trajectory_angle, joint_angle_2d, pose_joint_angle_for_frame
from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata
from core.geometry import SKELETON_EDGES
from models.inference import estimate_pose
from models.rtmw3d import RTMW3DModel


@dataclass(slots=True)
class KneeEventAnalysis:
    """Detected loading and knee-extension events with their supporting samples."""

    samples: list[tuple[int, float, float | None]]
    frame: int | None = None
    angle: float | None = None
    motion_start_frame: int | None = None
    status: str | None = None
    straight_frame: int | None = None


def detect_knee_event(
    observations: PoseObservations,
    tracked: TrackedPoses,
    candidate_frames: np.ndarray,
    shooting_side: str,
    release_frame: int,
    load_ball_low_frame: int | None,
    fps: float,
) -> KneeEventAnalysis:
    """Detect the deepest loading frame and a sustained knee-extension onset."""
    samples = []
    for index in candidate_frames:
        frame_points = tracked.analysis_keypoints[index]
        side_angles = {}
        for side in CONFIG["SIDE_NAMES"]:
            joint_indices = np.asarray(
                [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in ("hip", "knee", "ankle")]
            )
            visible = (
                tracked.valid[index, joint_indices].all()
                and np.isfinite(frame_points[joint_indices, :2]).all()
                and (observations.confidence[index, joint_indices] >= CONFIG["KEYPOINT_THRESHOLD"]).all()
            )
            if not visible:
                continue
            angle, _ = pose_joint_angle_for_frame(observations, tracked, int(index), side, "knee")
            if angle is not None and np.isfinite(angle):
                side_angles[side] = float(angle)
        if side_angles:
            samples.append((int(index), max(side_angles.values()), side_angles.get(shooting_side)))

    result = KneeEventAnalysis(samples=samples)
    if not samples:
        return result

    loading_onset = None
    if len(samples) >= 5:
        for sample_index, sample_angle, _ in samples:
            before = [
                angle
                for frame, angle, _ in samples
                if 0 < observations.timestamps[sample_index] - observations.timestamps[frame] <= 0.15
            ]
            after = [
                (frame, angle)
                for frame, angle, _ in samples
                if 0 <= observations.timestamps[frame] - observations.timestamps[sample_index] <= 0.15
            ]
            if len(before) < 3 or len(after) < 3:
                continue
            baseline_angle = float(np.median(before))
            if baseline_angle - float(np.median([angle for _, angle in after])) < 8.0:
                continue
            crossing = [frame for frame, angle in after if angle <= baseline_angle - 6.0]
            if crossing:
                loading_onset = min(crossing)
                break

    sample_frames = np.asarray([sample[0] for sample in samples], dtype=np.int32)
    flexion_angles = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    smoothed_angles = flexion_angles.copy()
    for sample_index, frame_index in enumerate(sample_frames):
        start = max(0, sample_index - 1)
        end = min(len(sample_frames), sample_index + 2)
        contiguous = np.abs(sample_frames[start:end] - frame_index) <= 1
        smoothed_angles[sample_index] = float(np.median(flexion_angles[start:end][contiguous]))

    search_start = 0
    search_end = len(sample_frames)
    if load_ball_low_frame is not None:
        window_start = max(0, load_ball_low_frame - round(fps * CONFIG["LOAD_SEARCH_BEFORE_SECONDS"]))
        window_end = min(
            release_frame + 1,
            load_ball_low_frame + round(fps * CONFIG["LOAD_SEARCH_AFTER_SECONDS"]) + 1,
        )
        in_window = np.flatnonzero((sample_frames >= window_start) & (sample_frames < window_end))
        if len(in_window) >= 3:
            search_start, search_end = int(in_window[0]), int(in_window[-1]) + 1
    if search_start >= search_end:
        search_start, search_end = 0, len(sample_frames)

    minimum_angle = float(np.min(smoothed_angles[search_start:search_end]))
    near_minimum = np.flatnonzero(
        smoothed_angles[search_start:search_end]
        <= minimum_angle + CONFIG["LOAD_ANGLE_PLATEAU_TOLERANCE_DEG"]
    )
    representative_index = min(search_start + int(near_minimum[-1]) + 1, search_end - 1)
    result.frame, event_angle, shooting_angle = samples[representative_index]
    result.angle = shooting_angle if shooting_angle is not None else event_angle
    result.motion_start_frame = (
        loading_onset if loading_onset is not None and loading_onset <= result.frame else result.frame
    )

    knee_time = float(observations.timestamps[result.frame])
    before_angles = [
        angle
        for index, angle, _ in samples
        if 0.0 < knee_time - observations.timestamps[index] <= 0.2
    ]
    after_angles = [
        angle
        for index, angle, _ in samples
        if 0.0 < observations.timestamps[index] - knee_time <= 0.2
    ]
    loading = len(before_angles) >= 2 and float(np.median(before_angles)) - event_angle >= 8.0
    recovery = len(after_angles) >= 2 and float(np.median(after_angles)) - event_angle >= 8.0
    result.status = "confirmed" if loading and recovery else "estimated"

    straight_streak = 0
    previous_index = None
    for index, angle, _ in samples:
        if index < result.frame:
            continue
        if previous_index is not None and index != previous_index + 1:
            straight_streak = 0
        straight_streak = straight_streak + 1 if angle >= CONFIG["KNEE_STRAIGHT_ANGLE_DEG"] else 0
        if straight_streak >= CONFIG["EXTENSION_CONFIRM_FRAMES"]:
            candidate = index - CONFIG["EXTENSION_CONFIRM_FRAMES"] + 1
            if candidate > result.frame:
                result.straight_frame = candidate
                break
        previous_index = index
    return result


def detect_arm_extension_frame(
    observations: PoseObservations,
    tracked: TrackedPoses,
    side: str,
    candidate_frames: list[int],
    timestamps: np.ndarray,
    fps: float,
    minimum_opening_deg: float = 20.0,
    *,
    angle_cache: dict[int, float | None] | None = None,
) -> int | None:
    """Find a sustained elbow-opening event when knee extension is not observable."""
    if len(candidate_frames) < 3:
        return None
    radius = max(1, round(fps * 0.08))
    cached_angles = angle_cache if angle_cache is not None else {}
    samples = []
    for frame_index in candidate_frames:
        start = max(0, frame_index - radius)
        end = min(len(timestamps), frame_index + radius + 1)
        values = []
        for index in range(start, end):
            if index not in cached_angles:
                cached_angles[index] = pose_joint_angle_for_frame(
                    observations, tracked, index, side, "elbow"
                )[0]
            angle = cached_angles[index]
            if angle is not None and np.isfinite(angle) and 30.0 <= angle <= 170.0:
                values.append(angle)
        if values:
            samples.append((frame_index, float(median(values))))

    if not samples:
        return None

    sample_frames = np.asarray([frame for frame, _ in samples], dtype=np.int32)
    sample_times = timestamps[sample_frames]
    sample_angles = np.asarray([angle for _, angle in samples], dtype=np.float64)
    sorted_times = bool(np.all(np.diff(sample_times) >= 0.0))
    window_seconds = CONFIG["MOTION_REFERENCE_WINDOW_SECONDS"]

    for sample_index, (frame_index, angle) in enumerate(samples):
        if sorted_times:
            current_time = sample_times[sample_index]
            start = np.searchsorted(sample_times, current_time, side="right")
            end_time = np.nextafter(current_time + window_seconds, np.inf)
            end = np.searchsorted(sample_times, end_time, side="right")
            future_times = sample_times[start:end]
            elapsed = future_times - current_time
            within_window = (elapsed > 0.0) & (elapsed <= window_seconds)
            future = sample_angles[start:end][within_window]
        else:
            future = np.asarray(
                [
                    value
                    for future_index, value in samples
                    if 0.0
                    < timestamps[future_index] - timestamps[frame_index]
                    <= window_seconds
                ],
                dtype=np.float64,
            )
        if len(future) >= 2 and float(median(future)) - angle >= minimum_opening_deg:
            return frame_index
    return None


def refine_lift_frame_from_arm_extension(
    observations: PoseObservations,
    tracked: TrackedPoses,
    side: str,
    candidate_frame: int,
    minimum_frame: int,
    release_frame: int,
) -> int:
    """Refine a lift frame when a clear elbow-extension onset is nearby.

    The search is limited to a small neighborhood of the detected lift frame.
    A candidate must show a fast opening rate that remains open over the
    configured confirmation interval.
    """
    radius = CONFIG["TAKEOFF_ARM_REFINE_RADIUS_FRAMES"]
    confirm_frames = CONFIG["TAKEOFF_ARM_REFINE_CONFIRM_FRAMES"]
    start = max(1, minimum_frame, candidate_frame - radius)
    end = min(release_frame, candidate_frame + radius + 1)

    for frame in range(start, end):
        previous_angle, _ = pose_joint_angle_for_frame(observations, tracked, frame - 1, side, "elbow")
        current_angle, _ = pose_joint_angle_for_frame(observations, tracked, frame, side, "elbow")
        confirm_frame = frame + confirm_frames
        if (
            previous_angle is None
            or current_angle is None
            or confirm_frame >= release_frame
            or observations.timestamps[frame] <= observations.timestamps[frame - 1]
        ):
            continue

        extension_rate = (current_angle - previous_angle) / (
            observations.timestamps[frame] - observations.timestamps[frame - 1]
        )
        if extension_rate < CONFIG["TAKEOFF_ARM_REFINE_MIN_RATE_DEG_S"]:
            continue

        confirmation_angles = [
            pose_joint_angle_for_frame(observations, tracked, sample_frame, side, "elbow")[0]
            for sample_frame in range(frame, confirm_frame + 1)
        ]
        if all(angle is not None for angle in confirmation_angles) and confirmation_angles[-1] >= current_angle:
            return frame

    return candidate_frame






def estimate_pose_shot_timing(
    observations: PoseObservations,
    tracked: TrackedPoses,
    fps: float,
    preferred_side: str | None = None,
) -> tuple[str, int, int] | None:
    """Estimate the shot-extension and release frames from arm pose when ball tracking fails."""
    candidates = []
    pose_series_by_side = {}
    for side in CONFIG["SIDE_NAMES"]:
        shoulder_index = CONFIG["KEYPOINT_INDEX"][f"{side}_shoulder"]
        elbow_index = CONFIG["KEYPOINT_INDEX"][f"{side}_elbow"]
        wrist_index = CONFIG["KEYPOINT_INDEX"][f"{side}_wrist"]
        joint_indices = np.asarray(
            [shoulder_index, elbow_index, wrist_index], dtype=np.int32
        )
        elbow_angles = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
        wrist_elevation = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
        angle_cache: dict[int, float | None] = {}
        for frame_index, points in enumerate(tracked.analysis_keypoints):
            visible = (
                tracked.valid[frame_index, joint_indices].all()
                and np.isfinite(points[joint_indices, :2]).all()
                and (observations.confidence[frame_index, joint_indices] >= CONFIG["KEYPOINT_THRESHOLD"]).all()
            )
            box_height = observations.boxes[frame_index, 3] - observations.boxes[frame_index, 1]
            if not visible or not np.isfinite(box_height) or box_height <= 0:
                continue
            angle, _ = pose_joint_angle_for_frame(observations, tracked, frame_index, side, "elbow")
            angle_cache[frame_index] = angle
            if angle is None or not np.isfinite(angle):
                continue
            elbow_angles[frame_index] = float(angle)
            wrist_elevation[frame_index] = float((points[shoulder_index, 1] - points[wrist_index, 1]) / box_height)
        pose_series_by_side[side] = (elbow_angles, wrist_elevation)

        valid_frames = np.flatnonzero(np.isfinite(elbow_angles) & np.isfinite(wrist_elevation))
        if len(valid_frames) < 3:
            continue
        extension_frame = detect_arm_extension_frame(
            observations,
            tracked,
            side,
            valid_frames.tolist(),
            observations.timestamps,
            fps,
            angle_cache=angle_cache,
        )
        if extension_frame is not None:
            end_time = observations.timestamps[extension_frame] + max(0.5, CONFIG["MOTION_POST_RELEASE_WINDOW_SECONDS"])
            release_candidates = valid_frames[
                (valid_frames > extension_frame) & (observations.timestamps[valid_frames] <= end_time)
            ]
        else:
            release_candidates = np.empty(0, dtype=np.int64)

        if len(release_candidates):
            peak_angle = float(np.percentile(elbow_angles[release_candidates], 90))
            extended = release_candidates[
                (elbow_angles[release_candidates] >= max(140.0, peak_angle - 8.0))
                & (wrist_elevation[release_candidates] >= 0.12)
            ]
            release_frame = (
                int(extended[0]) if len(extended)
                else int(release_candidates[np.argmax(wrist_elevation[release_candidates])])
            )
        else:
            release_frame = int(valid_frames[np.argmax(wrist_elevation[valid_frames])])
        if release_frame >= len(observations.timestamps) - 1:
            prior_release_candidates = valid_frames[valid_frames < release_frame]
            if not len(prior_release_candidates):
                continue
            release_frame = int(prior_release_candidates[-1])

        if extension_frame is None or extension_frame >= release_frame:
            before_release = valid_frames[valid_frames < release_frame]
            lower_elbow_frames = before_release[
                (observations.timestamps[release_frame] - observations.timestamps[before_release] <= 0.5)
                & (elbow_angles[before_release] <= elbow_angles[release_frame] - 12.0)
            ]
            extension_frame = (
                int(lower_elbow_frames[0])
                if len(lower_elbow_frames)
                else int(before_release[-1])
                if len(before_release)
                else release_frame
            )

        elapsed_start = max(0.0, observations.timestamps[release_frame] - 0.25)
        prior_frames = valid_frames[
            (observations.timestamps[valid_frames] >= elapsed_start) & (valid_frames <= extension_frame)
        ]
        prior_angle = (
            float(np.median(elbow_angles[prior_frames])) if len(prior_frames) else float(elbow_angles[extension_frame])
        )
        extension_amount = max(float(elbow_angles[release_frame]) - prior_angle, 0.0)
        score = max(float(wrist_elevation[release_frame]), 0.0) + extension_amount / 90.0
        if side == preferred_side:
            score += 0.05
        candidates.append((score, side, int(extension_frame), int(release_frame)))

    if not candidates:
        return None
    _, side, extension_frame, release_frame = max(candidates)
    elbow_angles, wrist_elevation = pose_series_by_side[side]
    confirm_frames = CONFIG["RELEASE_FALLBACK_CONFIRM_FRAMES"]
    search_end = min(release_frame, len(observations.timestamps) - confirm_frames)
    for frame in range(max(extension_frame + 1, 1), search_end):
        confirmation_frame = frame + confirm_frames
        interval = observations.timestamps[frame] - observations.timestamps[frame - 1]
        if (
            not np.isfinite(elbow_angles[frame - 1 : confirmation_frame + 1]).all()
            or not np.isfinite(wrist_elevation[frame])
            or wrist_elevation[frame] < CONFIG["RELEASE_POSE_ONSET_WRIST_ELEVATION_RATIO"]
            or interval <= 0
        ):
            continue
        opening_rate = (elbow_angles[frame] - elbow_angles[frame - 1]) / interval
        if (
            opening_rate >= CONFIG["RELEASE_POSE_ONSET_MIN_RATE_DEG_S"]
            and elbow_angles[confirmation_frame] >= elbow_angles[frame]
        ):
            release_frame = max(frame - 1, 0)
            break
    return side, extension_frame, release_frame


def detect_rim_contact(
    observations: PoseObservations,
    release_frame: int | None,
    release_timestamp: float | None,
    metric_validation_errors: list[str],
    camera_compensation_enabled: bool = False,
) -> tuple[int | None, float | None]:
    """Find ball entry below the rim's upper edge, falling back to rim overlap."""
    contact_frame = None
    contact_angle = None
    invalid_contact_seen = False
    overlap_fallback = None
    if release_frame is not None and np.isfinite(observations.rim_box).all():
        # Smoothing lags behind a fast descending ball and can move its apparent
        # rim crossing past an occlusion or bounce. Fit the actual detections.
        contact_search_end = min(
            len(observations.ball_boxes),
            int(
                np.searchsorted(
                    observations.timestamps,
                    release_timestamp + CONFIG["RIM_CONTACT_WINDOW_SECONDS"],
                    side="right",
                )
            ),
        )
        previous_ball_sample: tuple[int, np.ndarray, np.ndarray] | None = None
        for index in range(release_frame + 1, contact_search_end):
            rim = observations.rim_boxes[index] if observations.rim_boxes is not None else observations.rim_box
            if not np.isfinite(rim).all():
                continue
            rim_left, rim_top, rim_right, rim_bottom = rim
            ball_box = observations.ball_boxes[index]
            center = observations.ball_centers[index]
            confidence = observations.ball_scores[index]
            if (
                not np.isfinite(ball_box).all()
                or not np.isfinite(center).all()
                or not np.isfinite(confidence)
                or confidence < CONFIG["BALL_DETECTION_THRESHOLD"]
            ):
                continue
            closest = np.clip(center, (rim_left, rim_top), (rim_right, rim_bottom))
            distance = float(np.linalg.norm(center - closest))
            center_in_rim_x = rim_left <= center[0] <= rim_right
            if center_in_rim_x and rim_top <= center[1] <= rim_bottom:
                distance = float(
                    min(center[0] - rim_left, rim_right - center[0], center[1] - rim_top, rim_bottom - center[1])
                )
            ball_radius = float(((ball_box[2] - ball_box[0]) + (ball_box[3] - ball_box[1])) * 0.25)
            if center_in_rim_x and center[1] >= rim_top:
                candidate_angle, _ = entry_trajectory_angle(
                    observations, index, camera_compensation_enabled=camera_compensation_enabled,
                )
                contact_frame = index
                if candidate_angle is None and overlap_fallback is not None:
                    candidate_angle = overlap_fallback[1]
                contact_angle = candidate_angle
                invalid_contact_seen = candidate_angle is None
                break
            if distance <= ball_radius:
                candidate_angle, _ = entry_trajectory_angle(
                    observations, index, camera_compensation_enabled=camera_compensation_enabled,
                )
                if overlap_fallback is None:
                    overlap_fallback = (index, candidate_angle)

            if previous_ball_sample is not None:
                previous_index, previous_center, previous_rim = previous_ball_sample
                missing_frames = index - previous_index - 1
                previous_rim_offset = float(previous_center[1] - previous_rim[1])
                current_rim_offset = float(center[1] - rim_top)
                if (
                    0 < missing_frames <= CONFIG["ENTRY_ANGLE_MAX_GAP_FRAMES"]
                    and previous_rim_offset < 0.0 <= current_rim_offset
                    and current_rim_offset > previous_rim_offset
                ):
                    crossing_fraction = (
                        -previous_rim_offset / (current_rim_offset - previous_rim_offset)
                    )
                    crossing_x = float(
                        previous_center[0]
                        + crossing_fraction * (center[0] - previous_center[0])
                    )
                    crossing_left = float(
                        previous_rim[0]
                        + crossing_fraction * (rim_left - previous_rim[0])
                    )
                    crossing_right = float(
                        previous_rim[2]
                        + crossing_fraction * (rim_right - previous_rim[2])
                    )
                    if crossing_left <= crossing_x <= crossing_right:
                        candidate_angle, _ = entry_trajectory_angle(
                            observations,
                            index,
                            camera_compensation_enabled=camera_compensation_enabled,
                        )
                        if candidate_angle is None and overlap_fallback is not None:
                            candidate_angle = overlap_fallback[1]
                        contact_frame = index
                        contact_angle = candidate_angle
                        invalid_contact_seen = candidate_angle is None
                        break

            previous_ball_sample = (index, center.copy(), rim.copy())

        if contact_frame is None and overlap_fallback is not None:
            contact_frame, contact_angle = overlap_fallback
            invalid_contact_seen = contact_angle is None

    if contact_frame is not None:
        confirmed_entry_frame = contact_frame + CONFIG["RIM_ENTRY_CONFIRM_FRAMES"]
        if confirmed_entry_frame < len(observations.timestamps):
            contact_frame = confirmed_entry_frame
            contact_angle, _ = entry_trajectory_angle(
                observations,
                contact_frame,
                camera_compensation_enabled=camera_compensation_enabled,
            )
            invalid_contact_seen = contact_angle is None

    if invalid_contact_seen:
        metric_validation_errors.append("rim_contact_trajectory_invalid")
        logging.error("进球角度不可用：连续球路观测不足或存在速度、方向突变。")

    return contact_frame, contact_angle


def _hand_ball_clearance_measurements(
    raw_ball_centers: np.ndarray,
    ball_boxes: np.ndarray,
    boxes: np.ndarray,
    visualization_keypoints: np.ndarray,
    raw_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure hand-to-ball distance by side and ball radius by player height."""
    hand_distance_by_side = np.full((len(raw_ball_centers), len(CONFIG["SIDE_NAMES"])), np.nan, dtype=np.float32)
    ball_radius_ratio = np.full(len(raw_ball_centers), np.nan, dtype=np.float32)
    hand_segment_ids = {
        side: np.asarray(
            [
                edge
                for edge in SKELETON_EDGES
                if CONFIG["HAND_BASES"][side] <= edge[0] < CONFIG["HAND_BASES"][side] + CONFIG["HAND_JOINT_COUNT"]
                and CONFIG["HAND_BASES"][side] <= edge[1] < CONFIG["HAND_BASES"][side] + CONFIG["HAND_JOINT_COUNT"]
            ],
            dtype=np.int32,
        )
        for side in CONFIG["SIDE_NAMES"]
    }

    for index, center in enumerate(raw_ball_centers):
        box_height = boxes[index][3] - boxes[index][1]
        if not (np.isfinite(center).all() and np.isfinite(box_height) and box_height > 0):
            continue
        ball_box = np.asarray(ball_boxes[index], dtype=np.float32)
        if np.isfinite(ball_box).all():
            ball_width = ball_box[2] - ball_box[0]
            ball_height = ball_box[3] - ball_box[1]
            if ball_width > 0 and ball_height > 0:
                ball_radius_ratio[index] = (ball_width + ball_height) * 0.25 / box_height

        for side_index, side in enumerate(CONFIG["SIDE_NAMES"]):
            segments = hand_segment_ids[side]
            starts = visualization_keypoints[index, segments[:, 0], :2]
            ends = visualization_keypoints[index, segments[:, 1], :2]
            segment_confidence = np.minimum(
                raw_confidence[index, segments[:, 0]],
                raw_confidence[index, segments[:, 1]],
            )
            visible_segments = (
                np.isfinite(starts).all(axis=1)
                & np.isfinite(ends).all(axis=1)
                & np.isfinite(segment_confidence)
                & (segment_confidence >= CONFIG["KEYPOINT_THRESHOLD"])
            )
            if np.any(visible_segments):
                starts, ends = starts[visible_segments], ends[visible_segments]
                vectors = ends - starts
                offsets = center - starts
                fractions = np.clip(
                    np.sum(offsets * vectors, axis=1) / (np.sum(vectors * vectors, axis=1) + 1e-6),
                    0.0,
                    1.0,
                )
                nearest_points = starts + fractions[:, None] * vectors
                hand_distance_by_side[index, side_index] = (
                    np.min(np.linalg.norm(nearest_points - center, axis=1)) / box_height
                )
                continue

            hand_indices = np.arange(
                CONFIG["HAND_BASES"][side],
                CONFIG["HAND_BASES"][side] + CONFIG["HAND_JOINT_COUNT"],
            )
            hand_points = visualization_keypoints[index, hand_indices, :2]
            hand_confidence = raw_confidence[index, hand_indices]
            visible_hand = (
                np.isfinite(hand_points).all(axis=1)
                & np.isfinite(hand_confidence)
                & (hand_confidence >= CONFIG["KEYPOINT_THRESHOLD"])
            )
            if np.any(visible_hand):
                hand_center = np.median(hand_points[visible_hand], axis=0)
                hand_distance_by_side[index, side_index] = np.linalg.norm(hand_center - center) / box_height

    return hand_distance_by_side, ball_radius_ratio


def _hand_clearance_onset(
    clearance_ratio: np.ndarray,
    anchor_frame: int,
    timestamps: np.ndarray,
) -> int | None:
    """Find a sustained hand-to-ball separation near a provisional release."""
    confirm_frames = CONFIG["RELEASE_FALLBACK_CONFIRM_FRAMES"]
    search_start = max(
        1,
        int(
            np.searchsorted(
                timestamps,
                timestamps[anchor_frame] - CONFIG["RELEASE_FALLBACK_SEARCH_BEFORE_SECONDS"],
                side="left",
            )
        ),
    )
    search_end = min(
        len(timestamps) - confirm_frames + 1,
        int(
            np.searchsorted(
                timestamps,
                timestamps[anchor_frame] + CONFIG["RELEASE_FALLBACK_SEARCH_AFTER_SECONDS"],
                side="right",
            )
        ),
    )
    threshold = CONFIG["RELEASE_FALLBACK_HAND_CLEARANCE_RATIO"]
    for frame in range(search_start, search_end):
        if (
            not np.isfinite(clearance_ratio[frame])
            or clearance_ratio[frame] < threshold
            or (np.isfinite(clearance_ratio[frame - 1]) and clearance_ratio[frame - 1] >= threshold)
        ):
            continue
        confirmation = clearance_ratio[frame : frame + confirm_frames]
        if np.isfinite(confirmation).all() and np.all(confirmation >= threshold * 0.9):
            return frame
    return None


def detect_ball_pre_release(
    raw_ball_centers: np.ndarray,
    ball_boxes: np.ndarray,
    boxes: np.ndarray,
    visualization_keypoints: np.ndarray,
    raw_confidence: np.ndarray,
    timestamps: np.ndarray,
    metadata: VideoMetadata,
) -> np.ndarray:
    """Mark observed ball positions preceding confirmed separation from the hand."""
    hand_distance_by_side, ball_radius_ratio = _hand_ball_clearance_measurements(
        raw_ball_centers,
        ball_boxes,
        boxes,
        visualization_keypoints,
        raw_confidence,
    )
    hand_centers = np.full((len(raw_ball_centers), len(CONFIG["SIDE_NAMES"]), 2), np.nan, dtype=np.float32)
    for index, center in enumerate(raw_ball_centers):
        box_height = boxes[index][3] - boxes[index][1]
        if not (np.isfinite(center).all() and np.isfinite(box_height) and box_height > 0):
            continue

        for side_index, side in enumerate(CONFIG["SIDE_NAMES"]):
            base = CONFIG["HAND_BASES"][side]
            hand_indices = np.arange(base, base + CONFIG["HAND_JOINT_COUNT"])
            hand_points = visualization_keypoints[index, hand_indices, :2]
            hand_confidence = raw_confidence[index, hand_indices]
            visible_hand = (
                np.isfinite(hand_points).all(axis=1)
                & np.isfinite(hand_confidence)
                & (hand_confidence >= CONFIG["KEYPOINT_THRESHOLD"])
            )
            if np.any(visible_hand):
                hand_centers[index, side_index] = np.median(hand_points[visible_hand], axis=0)

    hand_distance_ratio = np.full(len(raw_ball_centers), np.nan, dtype=np.float32)
    for index, distances in enumerate(hand_distance_by_side):
        finite_distances = distances[np.isfinite(distances)]
        if len(finite_distances):
            hand_distance_ratio[index] = float(np.min(finite_distances))

    box_array = np.asarray(boxes, dtype=np.float32)
    ball_center_array = np.asarray(raw_ball_centers, dtype=np.float32)
    box_heights = box_array[:, 3] - box_array[:, 1]
    valid_box_heights = np.isfinite(box_heights) & (box_heights > 0)
    relative_ball_hand_offsets = np.full_like(hand_centers, np.nan)
    relative_ball_hand_offsets[valid_box_heights] = (
        ball_center_array[valid_box_heights, None, :] - hand_centers[valid_box_heights]
    ) / box_heights[valid_box_heights, None, None]

    release_window = max(3, round(metadata.fps * CONFIG["RELEASE_BASELINE_WINDOW_SECONDS"]))
    release_confirm_window = max(3, round(metadata.fps * CONFIG["RELEASE_FLIGHT_CONFIRM_SECONDS"]))
    candidate_frame = None
    candidate_side = None
    candidate_radius_px = None
    candidate_offset_baseline = None
    release_frame = len(raw_ball_centers)
    release_found = False
    used_border_fallback = False
    for index, distance_ratio in enumerate(hand_distance_ratio):
        if not np.isfinite(distance_ratio):
            candidate_frame = None
            candidate_side = None
            candidate_radius_px = None
            candidate_offset_baseline = None
            continue
        history_start = max(0, index - release_window)
        distance_history = hand_distance_ratio[history_start:index]
        distance_history = distance_history[np.isfinite(distance_history)]
        radius_history = ball_radius_ratio[history_start:index]
        radius_history = radius_history[np.isfinite(radius_history)]
        if len(distance_history) < 3 or not len(radius_history):
            continue
        baseline = float(np.median(distance_history))
        margin = float(np.median(radius_history)) * CONFIG["RELEASE_SEPARATION_RADIUS_RATIO"]
        absolute_margin = (
            ball_radius_ratio[index] * CONFIG["RELEASE_MIN_HAND_DISTANCE_BALL_RADII"]
            if np.isfinite(ball_radius_ratio[index])
            else np.inf
        )
        available_sides = np.flatnonzero(np.isfinite(hand_distance_by_side[index]))
        side_index = (
            int(available_sides[np.argmin(hand_distance_by_side[index, available_sides])])
            if len(available_sides)
            else None
        )
        relative_offset_baseline = None
        relative_offset_change = 0.0
        if side_index is not None:
            current_offset = relative_ball_hand_offsets[index, side_index]
            offset_history = relative_ball_hand_offsets[history_start:index, side_index]
            offset_history = offset_history[np.isfinite(offset_history).all(axis=1)]
            if len(offset_history) >= 2 and np.isfinite(current_offset).all():
                relative_offset_baseline = np.median(offset_history, axis=0)
                relative_offset_change = float(np.linalg.norm(current_offset - relative_offset_baseline))

        relative_separated = relative_offset_change >= CONFIG["RELEASE_RELATIVE_OFFSET_RATIO"]
        separated = (
            distance_ratio > baseline + margin
            or distance_ratio > absolute_margin
            or relative_separated
        )
        if candidate_frame is not None:
            relative_separation_persists = (
                candidate_offset_baseline is not None
                and side_index == candidate_side
                and relative_offset_change >= CONFIG["RELEASE_RELATIVE_OFFSET_RATIO"] * 0.5
            )
            if not separated and not relative_separation_persists:
                candidate_frame = None
                candidate_side = None
                candidate_radius_px = None
                candidate_offset_baseline = None
                continue
            candidate_age = timestamps[index] - timestamps[candidate_frame]
            if candidate_age > CONFIG["RELEASE_CANDIDATE_MAX_SECONDS"]:
                candidate_frame = None
                candidate_side = None
                candidate_radius_px = None
                candidate_offset_baseline = None
                continue
            available_confirm_frames = max(len(raw_ball_centers) - candidate_frame - 1, 1)
            required_confirm_frames = min(release_confirm_window, available_confirm_frames)
            if index - candidate_frame < required_confirm_frames:
                continue

            future_start = max(candidate_frame + 1, index - 1)
            future_end = min(len(raw_ball_centers), index + 2)
            future_centers = raw_ball_centers[future_start:future_end]
            future_hands = hand_centers[future_start:future_end, candidate_side]
            visible_ball = np.isfinite(future_centers).all(axis=1)
            initial_hand = hand_centers[candidate_frame, candidate_side]
            if np.any(visible_ball) and np.isfinite(initial_hand).all():
                future_ball = np.median(future_centers[visible_ball], axis=0)
                ball_travel = np.linalg.norm(future_ball - raw_ball_centers[candidate_frame])
                visible_future_hands = visible_ball & np.isfinite(future_hands).all(axis=1)
                if np.count_nonzero(visible_future_hands) >= 2:
                    initial_offset = raw_ball_centers[candidate_frame] - initial_hand
                    future_offset = future_ball - np.median(future_hands[visible_future_hands], axis=0)
                    relative_travel = np.linalg.norm(future_offset - initial_offset)
                else:
                    initial_separation = np.linalg.norm(raw_ball_centers[candidate_frame] - initial_hand)
                    future_separation = np.linalg.norm(future_ball - initial_hand)
                    relative_travel = max(float(future_separation - initial_separation), 0.0)
                if (
                    ball_travel >= candidate_radius_px * CONFIG["RELEASE_FLIGHT_TRAVEL_RADIUS_RATIO"]
                    and relative_travel >= candidate_radius_px * CONFIG["RELEASE_FLIGHT_TRAVEL_RADIUS_RATIO"]
                ):
                    release_frame = candidate_frame
                    release_found = True
                    break
            continue
        elif separated:
            shoulder_ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_shoulder"] for side in CONFIG["SIDE_NAMES"]]
            shoulders = visualization_keypoints[index, shoulder_ids, 1]
            visible_shoulders = np.isfinite(shoulders) & (
                raw_confidence[index, shoulder_ids] >= CONFIG["KEYPOINT_THRESHOLD"]
            )
            # A dribble or catch can also separate the ball from both hands.
            # A shot release must occur near or above the shoulder line.
            if np.any(visible_shoulders):
                shoulder_y = float(np.median(shoulders[visible_shoulders]))
                height = boxes[index][3] - boxes[index][1]
                if raw_ball_centers[index, 1] > shoulder_y + max(float(height) * 0.08, 0.0):
                    continue
            candidate_frame = index
            if side_index is not None:
                candidate_side = side_index
                candidate_radius_px = float(np.median(radius_history) * (boxes[index][3] - boxes[index][1]))
                candidate_offset_baseline = relative_offset_baseline
            else:
                candidate_frame = None
                candidate_side = None
                candidate_radius_px = None
                candidate_offset_baseline = None
    if not release_found:
        valid_ball_indices = np.flatnonzero(np.isfinite(raw_ball_centers).all(axis=1))
        if len(valid_ball_indices) >= 4:
            for index in valid_ball_indices:
                center = raw_ball_centers[index]
                ball_box = np.asarray(ball_boxes[index], dtype=np.float32)
                if not np.isfinite(ball_box).all():
                    continue
                ball_radius = float(((ball_box[2] - ball_box[0]) + (ball_box[3] - ball_box[1])) * 0.25)
                edge_margin = max(ball_radius * 1.5, min(metadata.width, metadata.height) * 0.035)
                near_top = center[1] <= edge_margin
                near_left = center[0] <= edge_margin
                near_right = center[0] >= metadata.width - edge_margin
                if not (near_top or near_left or near_right):
                    continue

                recent_indices = valid_ball_indices[
                    (valid_ball_indices < index)
                    & (timestamps[index] - np.asarray(timestamps)[valid_ball_indices] <= 0.12)
                ]
                if len(recent_indices) < 3:
                    continue
                recent_centers = raw_ball_centers[recent_indices[-3:]]
                displacement = center - recent_centers[0]
                box_height = float(boxes[index][3] - boxes[index][1])
                required_motion = max(ball_radius * 0.8, box_height * 0.04)
                moving_outward = (
                    (near_top and displacement[1] <= -required_motion)
                    or (near_left and displacement[0] <= -required_motion)
                    or (near_right and displacement[0] >= required_motion)
                )
                if (
                    not moving_outward
                    or not np.isfinite(boxes[index]).all()
                    or box_height <= 0
                    or not np.isfinite(hand_centers[index]).all(axis=1).any()
                ):
                    continue
                hand_distances = np.linalg.norm(hand_centers[index] - center, axis=1)
                if np.nanmin(hand_distances) > max(box_height * 0.2, ball_radius * 2.0):
                    continue

                next_indices = valid_ball_indices[
                    (valid_ball_indices > index)
                    & (np.asarray(timestamps)[valid_ball_indices] - timestamps[index] <= 0.08)
                ]
                if len(next_indices):
                    future_center = raw_ball_centers[next_indices[-1]]
                    continue_threshold = max(box_height * 0.01, ball_radius * 0.15)
                    if (
                        (near_top and future_center[1] > center[1] - continue_threshold)
                        or (near_left and future_center[0] > center[0] + continue_threshold)
                        or (near_right and future_center[0] < center[0] - continue_threshold)
                    ):
                        continue
                release_frame = min(int(index), len(raw_ball_centers))
                release_found = release_frame >= 0
                used_border_fallback = release_found
                if release_found:
                    break
        if not release_found:
            release_frame = -1

    if 0 <= release_frame < len(raw_ball_centers):
        hand_clearance_ratio = np.divide(
            hand_distance_ratio,
            ball_radius_ratio,
            out=np.full_like(hand_distance_ratio, np.nan),
            where=np.isfinite(ball_radius_ratio) & (ball_radius_ratio > 0),
        )
        if used_border_fallback:
            clearance_onset = _hand_clearance_onset(hand_clearance_ratio, release_frame, timestamps)
            if clearance_onset is not None:
                release_frame = clearance_onset
        elif (
            release_frame + 1 < len(raw_ball_centers)
            and np.isfinite(hand_clearance_ratio[release_frame])
            and hand_clearance_ratio[release_frame] < CONFIG["RELEASE_WEAK_HAND_CLEARANCE_RATIO"]
        ):
            release_frame += 1
        if (
            release_frame + 1 < len(raw_ball_centers)
            and np.isfinite(hand_clearance_ratio[release_frame])
            and hand_clearance_ratio[release_frame] < CONFIG["RELEASE_PERSISTENT_CONTACT_RATIO"]
        ):
            release_frame += 1

    ball_pre_release = np.isfinite(raw_ball_centers).all(axis=1) & (np.arange(len(raw_ball_centers)) < release_frame)

    return ball_pre_release


def estimate_shooting_hand(
    observations: PoseObservations,
    tracked: TrackedPoses,
    release_frame: int,
) -> tuple[str | None, float, dict[str, float]]:
    """Compare both arms on the same frames after the ball leaves the hands.

    The shooting arm usually remains elevated while the guide arm retracts.
    Hand-to-ball distance before release cannot distinguish these roles.
    Confidence describes the relative score margin, not a calibrated probability.
    """
    time = observations.timestamps[release_frame]
    frames = np.flatnonzero(
        (observations.timestamps >= time)
        & (observations.timestamps <= time + CONFIG["SHOOTING_HAND_FOLLOW_THROUGH_SECONDS"])
    )
    samples = {side: {} for side in CONFIG["SIDE_NAMES"]}
    shoulder_ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_shoulder"] for side in CONFIG["SIDE_NAMES"]]
    hip_ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_hip"] for side in CONFIG["SIDE_NAMES"]]
    inconsistent_frames = 0
    for frame in frames:
        height = observations.boxes[frame, 3] - observations.boxes[frame, 1]
        if not np.isfinite(height) or height <= 0:
            continue
        shoulders = tracked.analysis_keypoints[frame, shoulder_ids, 1]
        if not np.isfinite(shoulders).all():
            continue
        origin = np.mean(tracked.analysis_keypoints[frame, shoulder_ids, :2], axis=0)
        up = np.asarray([0.0, -1.0])
        hips = tracked.analysis_keypoints[frame, hip_ids, :2]
        if np.isfinite(hips).all():
            shoulder_span = (
                tracked.analysis_keypoints[frame, shoulder_ids[0], :2]
                - tracked.analysis_keypoints[frame, shoulder_ids[1], :2]
            )
            hip_span = hips[0] - hips[1]
            span_product = float(np.linalg.norm(shoulder_span) * np.linalg.norm(hip_span))
            if (
                tracked.valid[frame, hip_ids].all()
                and np.all(observations.confidence[frame, hip_ids] >= CONFIG["KEYPOINT_THRESHOLD"])
                and min(np.linalg.norm(shoulder_span), np.linalg.norm(hip_span)) > 1.0
                and np.dot(shoulder_span, hip_span) < -0.5 * span_product
            ):
                # Overlapping arms can make the model exchange upper-body
                # labels. Contradictory torso labels cannot establish a hand.
                inconsistent_frames += 1
                continue
            torso = origin - np.mean(hips, axis=0)
            length = float(np.linalg.norm(torso))
            if length > 0 and torso[1] < 0:
                up += torso / length
                up /= np.linalg.norm(up)
        for side in CONFIG["SIDE_NAMES"]:
            ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
            points = tracked.analysis_keypoints[frame, ids, :2]
            if (
                not tracked.valid[frame, ids].all()
                or not np.isfinite(points).all()
                or not np.all(observations.confidence[frame, ids] >= CONFIG["KEYPOINT_THRESHOLD"])
            ):
                continue
            _, elbow, wrist = points
            # Both arms need the same origin: the shooting shoulder itself
            # rises during extension and otherwise biases its arm's score down.
            wrist_lift = float(np.dot(wrist - origin, up) / height)
            elbow_lift = float(np.dot(elbow - origin, up) / height)
            if wrist_lift > 0.05:
                # Wrist flexion can lower the shooting hand below the guide
                # hand; the elevated elbow remains the stronger arm cue.
                samples[side][int(frame)] = wrist_lift + 3.0 * elbow_lift
    common = sorted(set(samples["left"]) & set(samples["right"]))
    evidence = {f"{side}_sample_count": float(len(values)) for side, values in samples.items()}
    evidence["inconsistent_torso_sample_count"] = float(inconsistent_frames)
    if len(common) >= 3:
        scores = {side: float(np.median([values[frame] for frame in common])) for side, values in samples.items()}
        evidence.update({f"{side}_score": score for side, score in scores.items()})
        side = max(scores, key=scores.get)
        margin = abs(scores["left"] - scores["right"])
        confidence = float(np.clip(margin / max(abs(scores["left"]), abs(scores["right"]), 1e-6), 0.0, 1.0))
        if margin >= 0.005:
            return side, confidence, evidence
    # Visibility of one arm alone is not evidence of handedness.
    return None, 0.0, evidence


def score_shooting_track(
    video_path: Path,
    boxes: np.ndarray,
    model: RTMW3DModel,
    device: torch.device,
) -> float:
    """Check a candidate's arm lift and extension before selecting the shooter."""
    available = np.flatnonzero(np.isfinite(boxes).all(axis=1))
    if len(available) < 3:
        return 0.0
    positions = np.linspace(0, len(available) - 1, min(len(available), CONFIG["SHOOTER_POSE_SAMPLES"])).astype(int)
    sample_frames = set(available[positions].tolist())
    samples = {side: [] for side in CONFIG["SIDE_NAMES"]}
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open candidate video: {video_path}")
        for index in range(max(sample_frames) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index not in sample_frames:
                continue
            points, confidence, _ = estimate_pose(model, frame, boxes[index], device)
            height = float(boxes[index, 3] - boxes[index, 1])
            if height <= 0:
                continue
            for side in CONFIG["SIDE_NAMES"]:
                ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
                if (
                    not np.isfinite(points[ids, :2]).all()
                    or not np.all(confidence[ids] >= CONFIG["KEYPOINT_THRESHOLD"])
                ):
                    continue
                shoulder, elbow, wrist = points[ids, :2]
                angle = joint_angle_2d(shoulder, elbow, wrist)
                if angle is not None:
                    samples[side].append(
                        (
                            index, angle,
                            float((shoulder[1] - wrist[1]) / height),
                            float((shoulder[1] - elbow[1]) / height),
                        )
                    )
    finally:
        capture.release()
    arm_scores = []
    for values in samples.values():
        scores = []
        for index, angle, wrist_lift, elbow_lift in values:
            before = [value[1] for value in values if value[0] < index]
            if len(before) < 2 or wrist_lift < 0.12 or elbow_lift < 0.0:
                continue
            opening = max(angle - float(np.percentile(before, 20)), 0.0)
            if opening >= 20.0:
                scores.append(wrist_lift + elbow_lift + opening / 180.0)
        if len(scores) >= 2:
            arm_scores.append(float(np.median(sorted(scores)[-3:])))
    return max(arm_scores, default=0.0)
