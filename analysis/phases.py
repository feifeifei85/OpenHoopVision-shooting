"""Inspect video conditions and detect the ordered phases of a shot."""

import numpy as np

from analysis.events import (
    detect_arm_extension_frame,
    detect_knee_event,
    estimate_pose_shot_timing,
    estimate_shooting_hand,
)
from analysis.kinematics import ball_trajectory_angle, ground_metrics, pose_joint_angle_for_frame
from analysis.measurements import ShotAnalysisContext, ShotAnalysisState
from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoProfile
from core.geometry import finite_median_filter


def analyze_shot_phases(
    context: ShotAnalysisContext,
    state: ShotAnalysisState,
    load_ball_low_frame: int | None,
) -> None:
    """Update the shared state with phase events and their supporting metrics."""
    observations = context.observations
    tracked = context.tracked
    metadata = context.metadata
    angle_measurements = context.angle_measurements

    pose_timing = None
    if state.release_frame is None or state.shooting_side is None:
        pose_timing = estimate_pose_shot_timing(
            observations,
            tracked,
            metadata.fps,
            preferred_side=state.shooting_side,
        )
    if pose_timing is not None:
        pose_side, pose_extension_frame, pose_release_frame = pose_timing
        if state.shooting_side is None:
            state.shooting_side = pose_side
        if state.extension_frame is None and (
            state.release_frame is None or pose_extension_frame < state.release_frame
        ):
            state.extension_frame = pose_extension_frame
            state.extension_trigger = "手肘伸展估计"
        if state.release_frame is None:
            if pose_extension_frame >= pose_release_frame:
                pose_extension_frame = max(pose_release_frame - 1, 0)
                state.extension_frame = pose_extension_frame
                state.extension_trigger = "手肘伸展估计"
            state.release_frame = pose_release_frame
            state.release_measurement_frame = (
                state.release_frame + 1
                if state.release_frame + 1 < len(observations.timestamps)
                else None
            )
            state.release_timestamp = (
                float(
                    (
                        observations.timestamps[state.release_frame]
                        + observations.timestamps[state.release_measurement_frame]
                    )
                    * 0.5
                )
                if state.release_measurement_frame is not None
                else float(observations.timestamps[state.release_frame])
            )
            state.phase_statuses["release"] = "estimated"
        if (
            state.extension_frame is not None
            and state.release_frame is not None
            and state.extension_frame < state.release_frame
        ):
            state.takeoff_frame = state.extension_frame
            state.takeoff_trigger = "第二阶段回退"
            state.phase_statuses["lift"] = "estimated"
            (
                state.takeoff_knee_angle,
                state.takeoff_elbow_angle,
                state.takeoff_upper_arm_angle,
                state.takeoff_hip_angle,
            ) = angle_measurements.at_frames(
                state.takeoff_frame,
                state.shooting_side,
                ("knee", "elbow", "shoulder", "hip"),
            )

    release_event_frame = state.release_event_frame
    if state.release_frame is None:
        return

    follow_side, state.shooting_side_confidence, state.shooting_side_evidence = estimate_shooting_hand(
        observations,
        tracked,
        state.release_frame,
    )
    if follow_side is not None:
        state.shooting_side = follow_side
        state.shooting_side_method = "paired_arm_follow_through"
    elif state.shooting_side is not None:
        state.shooting_side_method = "pose_extension"

    context.preparation_frames = np.flatnonzero(
        (np.arange(len(observations.timestamps)) <= release_event_frame)
        & np.isfinite(observations.boxes).all(axis=1)
    )

    angle_frame = (
        state.release_measurement_frame
        if state.release_measurement_frame is not None
        else state.release_frame
    )
    if angle_frame is not None:
        camera_shift = (
            observations.camera_shift_xy
            if context.video_profile.camera_compensation_enabled
            else np.zeros_like(observations.camera_shift_xy)
        )
        state.release_angle, state.release_angle_sample_frames = ball_trajectory_angle(
            observations.ball_centers - camera_shift,
            observations.timestamps,
            angle_frame,
            0,
            6,
            True,
            confidence=observations.ball_scores,
        )
        if state.release_angle is not None:
            state.metric_sources["release_angle_deg"] = "post_release_ball_trajectory_2d"

    if state.shooting_side is not None:
        knee_analysis = detect_knee_event(
            observations,
            tracked,
            context.preparation_frames,
            state.shooting_side,
            state.release_frame,
            load_ball_low_frame,
            metadata.fps,
        )
        state.knee_frame = knee_analysis.frame
        state.knee_angle = knee_analysis.angle
        state.knee_samples = knee_analysis.samples
        state.motion_start_frame = knee_analysis.motion_start_frame
        state.straight_frame = knee_analysis.straight_frame
        if knee_analysis.frame is not None:
            state.phase_statuses["load"] = knee_analysis.status
            (
                state.knee_elbow,
                state.knee_upper_arm,
                state.knee_hip,
            ) = angle_measurements.at_frames(
                state.knee_frame,
                state.shooting_side,
                ("elbow", "shoulder", "hip"),
            )
            _refine_takeoff_events(context, state)
            if state.unconfirmed_takeoff:
                state.unavailable_metric_codes.append("takeoff_unconfirmed")

    if state.knee_frame is None and state.shooting_side is not None:
        # A visible knee can establish loading even when the ankle is cropped.
        knee_index = CONFIG["KEYPOINT_INDEX"][f"{state.shooting_side}_knee"]
        before_extension = (
            state.extension_frame if state.extension_frame is not None else state.release_frame
        )
        candidates = context.preparation_frames[context.preparation_frames < before_extension]
        candidates = candidates[
            np.isfinite(tracked.analysis_keypoints[candidates, knee_index, 1])
            & (observations.confidence[candidates, knee_index] >= CONFIG["KEYPOINT_THRESHOLD"])
        ]
        if len(candidates):
            knee_points = context.to_reference_xy(
                candidates,
                tracked.analysis_keypoints[candidates, knee_index, :2],
            )
            knee_y = knee_points[:, 1]
            state.knee_frame = int(candidates[np.argmax(knee_y)])
            state.phase_statuses["load"] = "estimated"
            (
                state.knee_elbow,
                state.knee_upper_arm,
                state.knee_hip,
            ) = angle_measurements.at_frames(
                state.knee_frame,
                state.shooting_side,
                ("elbow", "shoulder", "hip"),
            )
            state.anchor_frames = candidates[
                (observations.timestamps[candidates] >= observations.timestamps[state.knee_frame] - 0.25)
                & (candidates <= state.knee_frame)
            ].tolist()
            if state.extension_frame is not None:
                state.elbow_angle = angle_measurements.at_frame(
                    state.extension_frame,
                    state.shooting_side,
                    "elbow",
                )
                state.upper_arm_angle = angle_measurements.at_frame(
                    state.extension_frame,
                    state.shooting_side,
                    "shoulder",
                )

    if state.motion_start_frame is None:
        state.motion_start_frame = (
            state.knee_frame if state.knee_frame is not None else state.extension_frame
        )


def _refine_takeoff_events(
    context: ShotAnalysisContext,
    state: ShotAnalysisState,
) -> None:
    """Refine lift and extension events using toe, knee, torso, and arm cues."""
    observations = context.observations
    tracked = context.tracked
    metadata = context.metadata
    preparation_frames = context.preparation_frames
    angle_measurements = context.angle_measurements
    release_event_frame = state.release_event_frame
    shooting_side = state.shooting_side
    knee_frame = state.knee_frame
    if knee_frame is None:
        return
    knee_samples = state.knee_samples
    motion_start_frame = state.motion_start_frame
    straight_frame = state.straight_frame
    takeoff_frame = state.takeoff_frame
    takeoff_trigger = state.takeoff_trigger
    takeoff_knee_angle = state.takeoff_knee_angle
    takeoff_elbow_angle = state.takeoff_elbow_angle
    takeoff_upper_arm_angle = state.takeoff_upper_arm_angle
    takeoff_hip_angle = state.takeoff_hip_angle
    extension_frame = state.extension_frame
    extension_trigger = state.extension_trigger
    elbow_angle = state.elbow_angle
    upper_arm_angle = state.upper_arm_angle
    extension_hip = state.extension_hip
    phase_statuses = {"lift": state.phase_statuses.get("lift", "unavailable")}
    unavailable_metric_codes = []
    toe_off_evidence_frame = state.toe_off_evidence_frame
    anchor_frames = []
    smoothed_torso_y = np.full(len(observations.timestamps), np.nan, dtype=np.float32)

    anchor_frame = motion_start_frame if motion_start_frame is not None else knee_frame
    anchor_start = max(
        0,
        anchor_frame - max(3, round(metadata.fps * 0.25)),
    )
    anchor_frames = [index for index in preparation_frames if anchor_start <= index <= anchor_frame]
    if len(anchor_frames) < 3:
        anchor_frames = [index for index in preparation_frames if index < anchor_frame][
            -max(3, round(metadata.fps * 0.25)) :
        ]
    toe_indices = {
        side: np.asarray([CONFIG["KEYPOINT_INDEX"][f"{side}_{name}"] for name in CONFIG["TOE_NAMES"]])
        for side in CONFIG["SIDE_NAMES"]
    }

    def toe_center_y(index: int, indices: np.ndarray) -> float | None:
        frame_points = tracked.analysis_keypoints[index]
        visible = (
            tracked.valid[index, indices]
            & np.isfinite(frame_points[indices, :2]).all(axis=1)
            & (observations.confidence[index, indices] >= CONFIG["KEYPOINT_THRESHOLD"])
        )
        if not np.any(visible):
            return None
        toe_points = context.to_reference_xy(index, frame_points[indices[visible], :2])
        return float(np.mean(toe_points[:, 1]))

    anchor_body_heights = [
        height_px
        for index in anchor_frames
        if (
            height_px := ground_metrics(
                observations.boxes[index],
                tracked.analysis_keypoints[index],
                metadata.width,
                metadata.height,
            )[1]
        )
        is not None
        and height_px > 0
    ]
    hip_indices = np.asarray([CONFIG["KEYPOINT_INDEX"]["left_hip"], CONFIG["KEYPOINT_INDEX"]["right_hip"]])
    shoulder_indices = np.asarray(
        [CONFIG["KEYPOINT_INDEX"]["left_shoulder"], CONFIG["KEYPOINT_INDEX"]["right_shoulder"]]
    )
    torso_center_y = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
    for index, frame_points in enumerate(tracked.analysis_keypoints):
        hip_visible = (
            tracked.valid[index, hip_indices]
            & np.isfinite(frame_points[hip_indices, 1])
            & (observations.confidence[index, hip_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
        )
        shoulder_visible = (
            tracked.valid[index, shoulder_indices]
            & np.isfinite(frame_points[shoulder_indices, 1])
            & (observations.confidence[index, shoulder_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
        )
        torso_samples = []
        if np.any(hip_visible):
            hip_points = context.to_reference_xy(index, frame_points[hip_indices[hip_visible], :2])
            torso_samples.append(
                float(np.median(hip_points[:, 1]))
            )
        if np.any(shoulder_visible):
            shoulder_points = context.to_reference_xy(
                index,
                frame_points[shoulder_indices[shoulder_visible], :2],
            )
            torso_samples.append(
                float(np.median(shoulder_points[:, 1]))
            )
        if torso_samples:
            torso_center_y[index] = float(np.mean(torso_samples))
    smoothed_torso_y = finite_median_filter(torso_center_y)

    toe_baselines = {}
    for side, indices in toe_indices.items():
        samples = [toe_center_y(index, indices) for index in anchor_frames]
        samples = [sample for sample in samples if sample is not None]
        if samples:
            toe_baselines[side] = float(np.median(samples))

    toe_off_frame = None
    toe_off_evidence_frame = None
    takeoff_knee_indices = np.asarray(
        [CONFIG["KEYPOINT_INDEX"][f"{shooting_side}_{joint}"] for joint in ("hip", "knee", "ankle")]
    )
    if len(anchor_frames) >= 3 and len(anchor_body_heights) >= 3 and toe_baselines:
        lift_threshold = max(
            float(np.median(anchor_body_heights)) * CONFIG["TOE_LIFT_RATIO"],
            4.0,
        )
        lift_streak = 0
        previous_index = None
        # Look two frames beyond ball separation so a lift that
        # starts at takeoff can still be confirmed after release.
        takeoff_search_end = min(
            len(observations.timestamps),
            release_event_frame
            + max(CONFIG["EXTENSION_CONFIRM_FRAMES"], round(metadata.fps * 0.15))
            + 1,
        )
        takeoff_search_start = max(
            knee_frame,
            motion_start_frame if motion_start_frame is not None else knee_frame,
        )
        for index in range(takeoff_search_start, takeoff_search_end):
            if previous_index is not None and index != previous_index + 1:
                lift_streak = 0
            lifted = False
            for side, baseline in toe_baselines.items():
                toe_y = toe_center_y(index, toe_indices[side])
                if toe_y is not None and baseline - toe_y >= lift_threshold:
                    lifted = True
                    break
            lift_streak = lift_streak + 1 if lifted else 0
            if lift_streak >= CONFIG["EXTENSION_CONFIRM_FRAMES"]:
                candidate = index - CONFIG["EXTENSION_CONFIRM_FRAMES"] + 1
                if candidate <= knee_frame:
                    previous_index = index
                    continue
                toe_off_evidence_frame = candidate
                toe_off_frame = min(candidate, release_event_frame)
                break
            previous_index = index

    lockout_frame = None
    lockout_streak = 0
    previous_index = None
    takeoff_search_end = min(
        len(observations.timestamps), release_event_frame + CONFIG["EXTENSION_CONFIRM_FRAMES"]
    )
    for index in range(knee_frame, takeoff_search_end):
        if previous_index is not None and index != previous_index + 1:
            lockout_streak = 0
        points = tracked.analysis_keypoints[index]
        visible = (
            tracked.valid[index, takeoff_knee_indices].all()
            and (observations.confidence[index, takeoff_knee_indices] >= CONFIG["KEYPOINT_THRESHOLD"]).all()
        )
        angle = (
            pose_joint_angle_for_frame(observations, tracked, index, shooting_side, "knee")[0]
            if visible
            else None
        )
        lockout_streak = (
            lockout_streak + 1 if angle is not None and angle >= CONFIG["KNEE_LOCKOUT_ANGLE_DEG"] else 0
        )
        if lockout_streak >= CONFIG["EXTENSION_CONFIRM_FRAMES"]:
            candidate = index - CONFIG["EXTENSION_CONFIRM_FRAMES"] + 1
            if candidate <= knee_frame:
                previous_index = index
                continue
            flexed_before = any(
                sample_index < candidate and sample_angle <= CONFIG["KNEE_STRAIGHT_ANGLE_DEG"] - 15.0
                for sample_index, sample_angle, _ in knee_samples
            )
            anchor_torso = smoothed_torso_y[anchor_frames] if anchor_frames else np.empty(0)
            anchor_torso = anchor_torso[np.isfinite(anchor_torso)]
            baseline_torso = float(np.median(anchor_torso)) if len(anchor_torso) else np.nan
            candidate_torso = smoothed_torso_y[knee_frame : candidate + 1]
            candidate_torso = candidate_torso[np.isfinite(candidate_torso)]
            torso_rise = (
                float(baseline_torso - np.min(candidate_torso))
                if np.isfinite(baseline_torso) and len(candidate_torso)
                else 0.0
            )
            required_torso_rise = (
                float(np.median(anchor_body_heights)) * 0.10 if anchor_body_heights else float("inf")
            )
            if candidate <= release_event_frame and flexed_before and torso_rise >= required_torso_rise:
                lockout_frame = candidate
                break
        previous_index = index

    arm_extension_frames = [
        sample_index
        for sample_index, _, _ in knee_samples
        if sample_index > knee_frame and sample_index < release_event_frame
    ]
    arm_angle_cache: dict[int, float | None] = {}
    arm_extension_frame = detect_arm_extension_frame(
        observations,
        tracked,
        shooting_side,
        arm_extension_frames,
        observations.timestamps,
        metadata.fps,
        minimum_opening_deg=20.0,
        angle_cache=arm_angle_cache,
    )
    if arm_extension_frame is None:
        arm_extension_frame = detect_arm_extension_frame(
            observations,
            tracked,
            shooting_side,
            arm_extension_frames,
            observations.timestamps,
            metadata.fps,
            minimum_opening_deg=12.0,
            angle_cache=arm_angle_cache,
        )
    arm_phase_frame = (
        arm_extension_frame + max(1, round(metadata.fps * 0.033))
        if arm_extension_frame is not None
        else None
    )
    if arm_phase_frame is not None and not knee_frame < arm_phase_frame < release_event_frame:
        arm_phase_frame = None

    if lockout_frame is not None:
        takeoff_frame, takeoff_trigger = lockout_frame, "膝盖伸直伴随躯干抬升"
        phase_statuses["lift"] = "confirmed"
    elif toe_off_frame is not None:
        takeoff_frame = toe_off_frame
        if toe_off_evidence_frame is not None and toe_off_evidence_frame > release_event_frame:
            takeoff_trigger = "小幅离地估计"
            phase_statuses["lift"] = "estimated"
        else:
            takeoff_trigger = "脚部持续抬升"
            phase_statuses["lift"] = "confirmed"
    elif arm_phase_frame is not None:
        takeoff_frame, takeoff_trigger = arm_phase_frame, "投篮臂伸展"
        phase_statuses["lift"] = "estimated"
    elif straight_frame is not None and straight_frame > knee_frame:
        takeoff_frame, takeoff_trigger = straight_frame, "膝盖伸展"
        phase_statuses["lift"] = "estimated"
    else:
        unavailable_metric_codes.append("takeoff_unconfirmed")

    if (
        takeoff_trigger == "脚部持续抬升"
        and takeoff_frame is not None
        and arm_phase_frame is not None
    ):
        corrected_frame = max(takeoff_frame - 1, knee_frame + 1)
        next_frame = min(corrected_frame + 1, len(observations.timestamps) - 1)
        frame_interval = (
            observations.timestamps[next_frame] - observations.timestamps[corrected_frame]
        )
        cue_time_delta = abs(
            observations.timestamps[corrected_frame] - observations.timestamps[arm_phase_frame]
        )
        if cue_time_delta <= CONFIG["TAKEOFF_CUE_AGREEMENT_SECONDS"] + frame_interval:
            fused_frame = (corrected_frame + arm_phase_frame + 1) // 2
            takeoff_frame = fused_frame + 1
            takeoff_trigger = "脚部与投篮臂动作融合"

    if straight_frame is not None and straight_frame > knee_frame:
        extension_frame, extension_trigger = straight_frame, "膝盖伸直"
    elif toe_off_frame is not None and toe_off_frame > knee_frame:
        # Toe lift is a fallback for poses that never reach the
        # knee-angle threshold; it must not preempt a supported
        # knee-extension event because toe articulation and court
        # perspective can create small apparent lifts while grounded.
        extension_frame, extension_trigger = toe_off_frame, "脚尖离地"
    if extension_frame is not None:
        extension_frame = int(extension_frame)
        elbow_angle = angle_measurements.at_frame(extension_frame, shooting_side, "elbow")
        upper_arm_angle = angle_measurements.at_frame(extension_frame, shooting_side, "shoulder")
        extension_hip = angle_measurements.at_frame(extension_frame, shooting_side, "hip")
    elif shooting_side is not None:
        extension_frame = detect_arm_extension_frame(
            observations,
            tracked,
            shooting_side,
            arm_extension_frames,
            observations.timestamps,
            metadata.fps,
            minimum_opening_deg=12.0,
            angle_cache=arm_angle_cache,
        )
        if extension_frame is not None:
            extension_trigger = "手肘持续伸展"
            elbow_angle = angle_measurements.at_frame(extension_frame, shooting_side, "elbow")
            upper_arm_angle = angle_measurements.at_frame(extension_frame, shooting_side, "shoulder")
            extension_hip = angle_measurements.at_frame(extension_frame, shooting_side, "hip")

    if takeoff_frame is not None and shooting_side is not None:
        (
            takeoff_knee_angle,
            takeoff_elbow_angle,
            takeoff_upper_arm_angle,
            takeoff_hip_angle,
        ) = angle_measurements.at_frames(
            takeoff_frame, shooting_side, ("knee", "elbow", "shoulder", "hip")
        )
    state.takeoff_frame = takeoff_frame
    state.takeoff_trigger = takeoff_trigger
    state.takeoff_knee_angle = takeoff_knee_angle
    state.takeoff_elbow_angle = takeoff_elbow_angle
    state.takeoff_upper_arm_angle = takeoff_upper_arm_angle
    state.takeoff_hip_angle = takeoff_hip_angle
    state.extension_frame = extension_frame
    state.extension_trigger = extension_trigger
    state.elbow_angle = elbow_angle
    state.upper_arm_angle = upper_arm_angle
    state.extension_hip = extension_hip
    state.phase_statuses["lift"] = phase_statuses["lift"]
    state.anchor_frames = anchor_frames
    state.smoothed_torso_y = smoothed_torso_y
    state.toe_off_evidence_frame = toe_off_evidence_frame
    state.unconfirmed_takeoff = "takeoff_unconfirmed" in unavailable_metric_codes


def inspect_video_profile(observations: PoseObservations, tracked: TrackedPoses) -> VideoProfile:
    """Summarize view geometry, observation coverage, and camera-motion evidence.

    View labels are conservative image-space diagnostics. Background-motion
    coverage gates vertical coordinate compensation; neither changes the
    user-defined release-angle sample points.
    """
    frame_count = len(observations.timestamps)
    player_coverage = float(np.isfinite(observations.boxes).all(axis=1).mean()) if frame_count else 0.0
    ball_coverage = float(np.isfinite(observations.ball_centers).all(axis=1).mean()) if frame_count else 0.0
    camera_motion_coverage = float(np.mean(observations.camera_motion_confidence >= 0.35)) if frame_count else 0.0
    camera_vertical_drift = (
        float(np.ptp(observations.camera_shift_xy[:, 1]))
        if frame_count and np.isfinite(observations.camera_shift_xy[:, 1]).any()
        else 0.0
    )
    camera_compensation_enabled = camera_motion_coverage >= 0.50
    held_frames = np.flatnonzero(observations.ball_pre_release)
    sample_frames = (
        held_frames if len(held_frames) >= 5 else np.flatnonzero(np.isfinite(observations.boxes).all(axis=1))
    )

    ratios = []
    for frame_index in sample_frames:
        points = tracked.analysis_keypoints[frame_index]
        shoulder_indices = [CONFIG["KEYPOINT_INDEX"]["left_shoulder"], CONFIG["KEYPOINT_INDEX"]["right_shoulder"]]
        hip_indices = [CONFIG["KEYPOINT_INDEX"]["left_hip"], CONFIG["KEYPOINT_INDEX"]["right_hip"]]
        nose_index = CONFIG["KEYPOINT_INDEX"]["nose"]
        required = shoulder_indices + hip_indices + [nose_index]
        if not np.all(tracked.valid[frame_index, required]):
            continue
        if not np.all(observations.confidence[frame_index, required] >= CONFIG["KEYPOINT_THRESHOLD"]):
            continue
        if not np.isfinite(points[required, :2]).all():
            continue
        left_shoulder, right_shoulder = points[shoulder_indices, :2]
        left_hip, right_hip = points[hip_indices, :2]
        shoulder_span = float(np.linalg.norm(left_shoulder - right_shoulder))
        torso_length = float(
            np.median(
                (
                    np.linalg.norm(left_shoulder - left_hip),
                    np.linalg.norm(right_shoulder - right_hip),
                )
            )
        )
        if shoulder_span < 8.0 or torso_length < 12.0:
            continue
        shoulder_midpoint = (left_shoulder + right_shoulder) * 0.5
        nose_offset = float(np.linalg.norm(points[nose_index, :2] - shoulder_midpoint))
        ratios.append((shoulder_span / torso_length, nose_offset / shoulder_span))

    if len(ratios) < 5:
        return VideoProfile(
            view_mode="unknown",
            view_confidence=0.0,
            view_sample_count=len(ratios),
            shoulder_span_to_torso_ratio=None,
            face_offset_to_shoulder_ratio=None,
            player_coverage=player_coverage,
            ball_coverage=ball_coverage,
            camera_motion_coverage=camera_motion_coverage,
            camera_vertical_drift_px=camera_vertical_drift,
            camera_compensation_enabled=camera_compensation_enabled,
        )

    features = np.asarray(ratios, dtype=np.float32)
    width_ratio = float(np.median(features[:, 0]))
    face_offset = float(np.median(features[:, 1]))
    side_votes = (features[:, 0] < 0.34) & (features[:, 1] > 1.1)
    front_votes = (features[:, 0] >= 0.52) & (features[:, 1] <= 0.85)
    side_support = float(np.mean(side_votes))
    front_support = float(np.mean(front_votes))
    if side_support >= 0.60:
        view_mode, support = "profile", side_support
    elif front_support >= 0.60:
        view_mode, support = "frontal", front_support
    else:
        view_mode, support = "oblique", max(1.0 - side_support - front_support, 0.0)
    sample_support = min(len(features) / 12.0, 1.0)
    feature_spread = float(
        np.median(np.abs(features[:, 0] - width_ratio)) / max(width_ratio, 1e-6)
        + np.median(np.abs(features[:, 1] - face_offset)) / max(face_offset, 1e-6)
    )
    stability = float(np.clip(1.0 - feature_spread, 0.0, 1.0))
    view_confidence = float(np.clip(support * sample_support * stability, 0.0, 1.0))
    return VideoProfile(
        view_mode=view_mode,
        view_confidence=view_confidence,
        view_sample_count=len(features),
        shoulder_span_to_torso_ratio=width_ratio,
        face_offset_to_shoulder_ratio=face_offset,
        player_coverage=player_coverage,
        ball_coverage=ball_coverage,
        camera_motion_coverage=camera_motion_coverage,
        camera_vertical_drift_px=camera_vertical_drift,
        camera_compensation_enabled=camera_compensation_enabled,
    )
