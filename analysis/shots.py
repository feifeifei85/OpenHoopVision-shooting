"""Detect shooting events and calculate motion metrics."""

import logging

import numpy as np

from analysis.events import (
    detect_rim_contact,
    estimate_pose_shot_timing,
    refine_lift_frame_from_arm_extension,
)
from analysis.kinematics import (
    elbow_angular_velocity_between_frames,
)
from analysis.report import validate_metric_values, validate_phase_event_order
from analysis.measurements import (
    JointAngleMetrics,
    ShotAnalysisContext,
    ShotAnalysisState,
    measure_shot_body,
)
from analysis.phases import analyze_shot_phases, inspect_video_profile
from core.config import CONFIG
from core.data import PoseObservations, ShotMetrics, TrackedPoses, VideoMetadata, VideoProfile
from core.geometry import apply_affine_transform


def motion_score_for_ratio(ratio: float) -> tuple[float, str]:
    """Map release-to-peak timing ratio to a score and motion class."""
    if ratio <= CONFIG["MOTION_RATIO_SINGLE_MAX"]:
        value = CONFIG["MOTION_VALUE_MIN"] + (
            ratio / CONFIG["MOTION_RATIO_SINGLE_MAX"] * (1.0 - CONFIG["MOTION_VALUE_MIN"])
        )
    else:
        value = 1.0 + (
            (ratio - CONFIG["MOTION_RATIO_SINGLE_MAX"])
            / (CONFIG["MOTION_RATIO_DOUBLE_MIN"] - CONFIG["MOTION_RATIO_SINGLE_MAX"])
        )
    motion_class = (
        "一段式"
        if ratio < CONFIG["MOTION_RATIO_SINGLE_MAX"]
        else "1.5段式"
        if ratio <= CONFIG["MOTION_RATIO_DOUBLE_MIN"]
        else "二段式"
    )
    return float(value), motion_class


def calculate_shot_metrics(
    observations: PoseObservations,
    tracked: TrackedPoses,
    metadata: VideoMetadata,
) -> ShotMetrics:
    """Calculate shot metrics using the final event frame as the height anchor."""
    video_profile = inspect_video_profile(observations, tracked)
    metrics = _calculate_shot_metrics_pass(
        observations, tracked, metadata, None, video_profile
    )

    def height_anchor(result: ShotMetrics) -> int | None:
        return next(
            (
                frame
                for frame in (
                    result.takeoff_frame,
                    result.extension_frame,
                    result.motion_start_frame,
                    result.knee_frame,
                )
                if frame is not None
            ),
            None,
        )

    anchor_frame = height_anchor(metrics)
    for _ in range(3):
        if anchor_frame is None:
            return metrics
        metrics = _calculate_shot_metrics_pass(
            observations, tracked, metadata, anchor_frame, video_profile
        )
        final_anchor = height_anchor(metrics)
        if final_anchor == anchor_frame:
            return metrics
        anchor_frame = final_anchor
    return metrics


def _calculate_shot_metrics_pass(
    observations: PoseObservations,
    tracked: TrackedPoses,
    metadata: VideoMetadata,
    body_height_anchor_frame: int | None,
    video_profile: VideoProfile,
) -> ShotMetrics:
    state = ShotAnalysisState()
    logging.info(
        "视频概况 "
        "view=%s view_confidence=%.3f view_samples=%d player_coverage=%.3f "
        "ball_coverage=%.3f camera_motion_coverage=%.3f camera_vertical_drift_px=%.1f",
        video_profile.view_mode,
        video_profile.view_confidence,
        video_profile.view_sample_count,
        video_profile.player_coverage,
        video_profile.ball_coverage,
        video_profile.camera_motion_coverage,
        video_profile.camera_vertical_drift_px,
    )
    camera_y = (
        observations.camera_shift_xy[:, 1]
        if video_profile.camera_compensation_enabled
        else np.zeros(len(observations.timestamps), dtype=np.float32)
    )
    camera_transform = (
        observations.camera_transform_to_reference_xy
        if video_profile.camera_compensation_enabled
        else None
    )
    held_frames = np.flatnonzero(observations.ball_pre_release)
    load_ball_low_frame = None
    if len(held_frames):
        held_ball_points = observations.ball_centers_smoothed[held_frames]
        if camera_transform is not None:
            held_ball_points = apply_affine_transform(
                held_ball_points,
                camera_transform[held_frames],
            )
        else:
            held_ball_points = held_ball_points.copy()
            held_ball_points[:, 1] -= camera_y[held_frames]
        held_ball_y = held_ball_points[:, 1]
        visible_held_ball = (
            np.isfinite(held_ball_y)
            & np.isfinite(observations.ball_scores[held_frames])
            & (observations.ball_scores[held_frames] >= CONFIG["BALL_DETECTION_THRESHOLD"])
        )
        if np.any(visible_held_ball):
            visible_frames = held_frames[visible_held_ball]
            load_ball_low_frame = int(visible_frames[np.argmax(held_ball_y[visible_held_ball])])
    state.release_frame = (
        int(held_frames[-1]) if len(held_frames) and held_frames[-1] + 1 < len(observations.timestamps) else None
    )
    # The dashboard marks the last frame with hand contact; measurements retain
    # the first separated frame used before the display frame was backdated.
    state.release_measurement_frame = state.release_frame + 1 if state.release_frame is not None else None
    state.release_timestamp = (
        float((observations.timestamps[state.release_frame] + observations.timestamps[state.release_frame + 1]) * 0.5)
        if state.release_frame is not None
        else None
    )
    metric_validation_errors = []
    angle_measurements = JointAngleMetrics(observations, tracked)
    angle_source_counts = angle_measurements.source_counts
    context = ShotAnalysisContext(
        observations=observations,
        tracked=tracked,
        metadata=metadata,
        video_profile=video_profile,
        camera_y=camera_y,
        camera_transform_to_reference_xy=camera_transform,
        angle_measurements=angle_measurements,
        body_height_anchor_frame=body_height_anchor_frame,
    )
    state.phase_statuses = {
        "load": "unavailable",
        "lift": "unavailable",
        "release": "confirmed" if state.release_frame is not None else "unavailable",
        "rim_contact": "unavailable",
    }
    state.smoothed_torso_y = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
    analyze_shot_phases(context, state, load_ball_low_frame)
    release_event_frame = state.release_event_frame
    preparation_frames = context.preparation_frames

    body_metrics = measure_shot_body(context, state, metric_validation_errors)
    body_height = body_metrics.body_height_px
    body_height_reference_partial = body_metrics.body_height_reference_partial
    ground_y = body_metrics.ground_y_px
    grounded_foot_baseline_y = body_metrics.grounded_foot_baseline_y_px
    knee_ball_height = body_metrics.knee_ball_height_ratio
    extension_ball_height = body_metrics.extension_ball_height_ratio
    takeoff_ball_height = body_metrics.takeoff_ball_height_ratio
    release_height = body_metrics.release_height_ratio
    release_foot_lift = body_metrics.release_foot_lift_ratio
    max_jump_height_ratio = body_metrics.max_jump_height_ratio
    release_elbow = body_metrics.release_elbow_angle_deg
    elbow_velocity = None
    velocity_start_sample = velocity_end_sample = None
    motion_ratio_raw = motion_value = motion_class = None
    if state.release_frame is not None and (state.extension_frame is None or state.shooting_side is None):
        pose_timing = estimate_pose_shot_timing(
            observations,
            tracked,
            metadata.fps,
            preferred_side=state.shooting_side,
        )
        if pose_timing is not None:
            pose_side, pose_extension_frame, _ = pose_timing
            if state.shooting_side is None:
                state.shooting_side = pose_side
            if state.extension_frame is None and pose_extension_frame < release_event_frame:
                state.extension_frame = pose_extension_frame
                state.extension_trigger = "手肘伸展估计"

    if (
        state.release_frame is not None
        and state.takeoff_frame is None
        and state.phase_statuses.get("lift") != "confirmed"
        and state.extension_frame is not None
        and state.extension_frame < release_event_frame
        and (state.knee_frame is None or state.extension_frame > state.knee_frame)
    ):
        state.takeoff_frame = state.extension_frame
        state.takeoff_trigger = "第二阶段回退"
        state.phase_statuses["lift"] = "estimated"
        if state.shooting_side is not None:
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
        state.unavailable_metric_codes = [
            code for code in state.unavailable_metric_codes if code != "takeoff_unconfirmed"
        ]

    if state.takeoff_frame is not None and state.takeoff_trigger != "出手前膝盖最大伸展":
        first_post_load_frame = state.knee_frame + 1 if state.knee_frame is not None else 0
        corrected_takeoff_frame = max(state.takeoff_frame - 1, first_post_load_frame)
        if corrected_takeoff_frame != state.takeoff_frame:
            state.takeoff_frame = corrected_takeoff_frame
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
            if ground_y is not None and body_height is not None and body_height > 0:
                takeoff_ball_height = body_metrics.ball_height_ratio(
                    context,
                    state,
                    state.takeoff_frame,
                    "takeoff_ball_height_ratio",
                )

    if state.takeoff_frame is not None and state.shooting_side is not None and release_event_frame is not None:
        refined_takeoff_frame = refine_lift_frame_from_arm_extension(
            observations,
            tracked,
            state.shooting_side,
            state.takeoff_frame,
            state.knee_frame + 1 if state.knee_frame is not None else 0,
            release_event_frame,
        )
        arm_refined_takeoff = refined_takeoff_frame != state.takeoff_frame
        toe_refined_takeoff = False
        if (
            state.toe_off_evidence_frame is not None
            and state.toe_off_evidence_frame < release_event_frame
            and abs(
                observations.timestamps[state.toe_off_evidence_frame]
                - observations.timestamps[refined_takeoff_frame]
            )
            <= CONFIG["TAKEOFF_CUE_AGREEMENT_SECONDS"]
        ):
            blended_frame = int(
                np.floor(
                    refined_takeoff_frame * (1.0 - CONFIG["TAKEOFF_TOE_EVIDENCE_WEIGHT"])
                    + state.toe_off_evidence_frame * CONFIG["TAKEOFF_TOE_EVIDENCE_WEIGHT"]
                    + 0.5
                )
            )
            toe_refined_takeoff = blended_frame != refined_takeoff_frame
            refined_takeoff_frame = blended_frame
        if state.shooting_side is not None and refined_takeoff_frame + 1 < release_event_frame:
            current_elbow = angle_measurements.at_frame(refined_takeoff_frame, state.shooting_side, "elbow")
            next_elbow = angle_measurements.at_frame(refined_takeoff_frame + 1, state.shooting_side, "elbow")
            if (
                current_elbow is not None
                and current_elbow < 30.0
                and next_elbow is not None
                and 30.0 <= next_elbow <= 180.0
                and next_elbow > current_elbow
            ):
                refined_takeoff_frame += 1
        if refined_takeoff_frame != state.takeoff_frame:
            state.takeoff_frame = refined_takeoff_frame
            if toe_refined_takeoff:
                state.takeoff_trigger = "多线索融合" if arm_refined_takeoff else "脚部线索融合"
            elif arm_refined_takeoff:
                state.takeoff_trigger = "手肘快速伸展"
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
            if ground_y is not None and body_height is not None and body_height > 0:
                takeoff_ball_height = body_metrics.ball_height_ratio(
                    context,
                    state,
                    state.takeoff_frame,
                    "takeoff_ball_height_ratio",
                )

    velocity_start_frame = state.takeoff_frame if state.takeoff_frame is not None else state.extension_frame
    velocity_end_frame = release_event_frame
    if (
        state.shooting_side is not None
        and velocity_start_frame is not None
        and velocity_end_frame is not None
    ):
        elbow_velocity, velocity_start_sample, velocity_end_sample, _, _ = elbow_angular_velocity_between_frames(
            observations, tracked, state.shooting_side, velocity_start_frame, velocity_end_frame
        )
        if elbow_velocity is not None:
            state.metric_sources["elbow_angular_velocity_deg_s"] = (
                "neighbor_pose_angle_event_window"
                if velocity_start_sample != velocity_start_frame or velocity_end_sample != velocity_end_frame
                else "source_video_time_not_physical_speed"
            )

    if release_event_frame is not None:
        if state.no_confirmed_airborne_phase and state.knee_frame is not None:
            state.motion_score_start_frame = state.knee_frame
        elif state.takeoff_frame is not None and state.takeoff_frame < release_event_frame:
            state.motion_score_start_frame = state.takeoff_frame
        elif state.extension_frame is not None and state.extension_frame <= release_event_frame:
            state.motion_score_start_frame = state.extension_frame
        elif state.motion_start_frame is not None and state.motion_start_frame < release_event_frame:
            state.motion_score_start_frame = state.motion_start_frame
        elif state.knee_frame is not None and state.knee_frame < release_event_frame:
            state.motion_score_start_frame = state.knee_frame
        else:
            state.motion_score_start_frame = max(release_event_frame - 1, 0)
            state.phase_statuses["motion"] = "estimated"

        if state.motion_peak_frame is None or state.motion_peak_frame <= state.motion_score_start_frame:
            state.motion_peak_frame = (
                release_event_frame + 1
                if release_event_frame + 1 < len(observations.timestamps)
                else release_event_frame
            )
            state.phase_statuses["motion"] = "estimated"

        if state.motion_score_start_frame is not None and state.motion_peak_frame is not None:
            numerator = release_event_frame - state.motion_score_start_frame
            denominator = state.motion_peak_frame - state.motion_score_start_frame
            if denominator <= 0 or numerator / denominator > 2.25:
                state.motion_peak_frame = (
                    release_event_frame + 1
                    if release_event_frame + 1 < len(observations.timestamps)
                    else release_event_frame
                )
                denominator = state.motion_peak_frame - state.motion_score_start_frame
                state.phase_statuses["motion"] = "estimated"
            if denominator > 0 and numerator >= 0:
                motion_ratio_raw = float(numerator / denominator)
                raw_score, motion_class = motion_score_for_ratio(motion_ratio_raw)
                motion_value = float(np.clip(raw_score, CONFIG["MOTION_VALUE_MIN"], CONFIG["MOTION_VALUE_MAX"]))
                if (
                    raw_score >= CONFIG["MOTION_VALUE_MAX"]
                    or state.max_jump_height_method == "foot_trace_below_lift_threshold"
                    or state.phase_statuses.get("lift") != "confirmed"
                    or state.phase_statuses.get("release") != "confirmed"
                ):
                    state.phase_statuses["motion"] = "estimated"
        if motion_value is None:
            state.unavailable_metric_codes.append("motion_timing_unavailable")

    state.contact_frame, contact_angle = detect_rim_contact(
        observations, state.release_frame, state.release_timestamp, metric_validation_errors,
        camera_compensation_enabled=video_profile.camera_compensation_enabled,
    )
    if state.contact_frame is not None:
        state.phase_statuses["rim_contact"] = "confirmed"
        state.metric_sources["entry_angle_deg"] = "tracked_three_frame_incoming_mean"

    metric_values = {
        "knee_ball_height_ratio": knee_ball_height,
        "extension_ball_height_ratio": extension_ball_height,
        "takeoff_ball_height_ratio": takeoff_ball_height,
        "release_height_ratio": release_height,
        "release_foot_lift_ratio": release_foot_lift,
        "max_jump_height_ratio": max_jump_height_ratio,
        "knee_elbow_angle_deg": state.knee_elbow,
        "knee_upper_arm_angle_deg": state.knee_upper_arm,
        "knee_hip_angle_deg": state.knee_hip,
        "takeoff_elbow_angle_deg": state.takeoff_elbow_angle,
        "takeoff_upper_arm_angle_deg": state.takeoff_upper_arm_angle,
        "takeoff_hip_angle_deg": state.takeoff_hip_angle,
        "extension_hip_angle_deg": state.extension_hip,
        "release_elbow_angle_deg": release_elbow,
        "elbow_angle_deg": state.elbow_angle,
        "upper_arm_angle_deg": state.upper_arm_angle,
        "knee_angle_deg": state.knee_angle,
        "takeoff_knee_angle_deg": state.takeoff_knee_angle,
        "release_angle_deg": state.release_angle,
        "contact_angle_deg": contact_angle,
        "motion_ratio_raw": motion_ratio_raw,
        "motion_value": motion_value,
        "elbow_angular_velocity_deg_s": elbow_velocity,
    }
    angle_metric_frames = {
        "knee_angle_deg": (state.knee_frame, "knee"),
        "knee_elbow_angle_deg": (state.knee_frame, "elbow"),
        "knee_upper_arm_angle_deg": (state.knee_frame, "shoulder"),
        "knee_hip_angle_deg": (state.knee_frame, "hip"),
        "extension_hip_angle_deg": (state.extension_frame, "hip"),
        "elbow_angle_deg": (state.extension_frame, "elbow"),
        "upper_arm_angle_deg": (state.extension_frame, "shoulder"),
        "takeoff_knee_angle_deg": (state.takeoff_frame, "knee"),
        "takeoff_elbow_angle_deg": (state.takeoff_frame, "elbow"),
        "takeoff_upper_arm_angle_deg": (state.takeoff_frame, "shoulder"),
        "takeoff_hip_angle_deg": (state.takeoff_frame, "hip"),
        "release_elbow_angle_deg": (state.release_frame, "elbow"),
    }
    for metric_id, (frame_index, joint_name) in angle_metric_frames.items():
        if metric_values[metric_id] is None:
            metric_values[metric_id] = angle_measurements.for_metric(
                frame_index, state.shooting_side, joint_name, metric_id, state.metric_sources
            )
        elif frame_index is not None and state.shooting_side is not None:
            metric_values[metric_id] = angle_measurements.smooth_metric(
                frame_index,
                state.shooting_side,
                joint_name,
                metric_id,
                metric_values[metric_id],
                state.metric_sources,
            )
    validate_metric_values(
        metric_values,
        metric_validation_errors,
        state.extension_frame,
        state.release_timestamp,
        observations.timestamps,
        velocity_start_frame=velocity_start_sample,
        velocity_end_frame=(
            velocity_end_sample if velocity_end_sample is not None else velocity_end_frame
        ),
    )

    knee_ball_height = metric_values["knee_ball_height_ratio"]
    extension_ball_height = metric_values["extension_ball_height_ratio"]
    takeoff_ball_height = metric_values["takeoff_ball_height_ratio"]
    release_height = metric_values["release_height_ratio"]
    release_foot_lift = metric_values["release_foot_lift_ratio"]
    max_jump_height_ratio = metric_values["max_jump_height_ratio"]
    minimum_reportable_lift = max(
        CONFIG["MOTION_MIN_LIFT_RATIO"],
        4.0 / body_height if body_height is not None and body_height > 0 else CONFIG["MOTION_MIN_LIFT_RATIO"],
    )
    if release_foot_lift is not None and release_foot_lift >= minimum_reportable_lift and max_jump_height_ratio is None:
        metric_validation_errors.append("max_jump_unavailable_with_release_lift")
    state.knee_elbow = metric_values["knee_elbow_angle_deg"]
    state.knee_upper_arm = metric_values["knee_upper_arm_angle_deg"]
    state.knee_hip = metric_values["knee_hip_angle_deg"]
    state.extension_hip = metric_values["extension_hip_angle_deg"]
    release_elbow = metric_values["release_elbow_angle_deg"]
    state.elbow_angle = metric_values["elbow_angle_deg"]
    state.upper_arm_angle = metric_values["upper_arm_angle_deg"]
    state.knee_angle = metric_values["knee_angle_deg"]
    state.takeoff_knee_angle = metric_values["takeoff_knee_angle_deg"]
    state.takeoff_elbow_angle = metric_values["takeoff_elbow_angle_deg"]
    state.takeoff_upper_arm_angle = metric_values["takeoff_upper_arm_angle_deg"]
    state.takeoff_hip_angle = metric_values["takeoff_hip_angle_deg"]
    state.release_angle = metric_values["release_angle_deg"]
    contact_angle = metric_values["contact_angle_deg"]
    motion_ratio_raw = metric_values["motion_ratio_raw"]
    motion_value = metric_values["motion_value"]
    elbow_velocity = metric_values["elbow_angular_velocity_deg_s"]
    if motion_value is not None and motion_ratio_raw is None:
        # The raw ratio may already have failed its own range check. Keep that
        # primary failure and don't report a second formula mismatch whose
        # only cause is the now-missing operand.
        motion_value = motion_class = None
    if motion_value is not None:
        score_start_frame = state.motion_score_start_frame
        score_peak_frame = state.motion_peak_frame
        if (
            score_start_frame is None
            or release_event_frame is None
            or release_event_frame < score_start_frame
            or score_peak_frame is None
            or score_peak_frame <= score_start_frame
        ):
            metric_validation_errors.append("motion_frame_dependencies_invalid")
            motion_ratio_raw = motion_value = motion_class = None
        else:
            expected_ratio = float(
                (release_event_frame - score_start_frame) / (score_peak_frame - score_start_frame)
            )
            expected_value, _ = motion_score_for_ratio(expected_ratio)
            expected_value = float(
                np.clip(
                    expected_value,
                    CONFIG["MOTION_VALUE_MIN"],
                    CONFIG["MOTION_VALUE_MAX"],
                )
            )
            if (
                motion_ratio_raw is None
                or not np.isclose(motion_ratio_raw, expected_ratio, rtol=1e-6, atol=1e-9)
                or not np.isclose(motion_value, expected_value, rtol=1e-6, atol=1e-9)
                or not CONFIG["MOTION_VALUE_MIN"] <= motion_value <= CONFIG["MOTION_VALUE_MAX"]
            ):
                metric_validation_errors.append("motion_formula_mismatch")
                motion_ratio_raw = motion_value = motion_class = None
    if max_jump_height_ratio is None:
        state.body_peak_frame = None
        state.max_jump_height_method = None
        if motion_value is not None:
            state.phase_statuses["motion"] = "estimated"
    if motion_value is None:
        motion_ratio_raw = None
        motion_class = None

    # Motion depends on takeoff, release, and peak frame numbers.
    for error in validate_phase_event_order(
        state.knee_frame,
        state.extension_frame,
        state.takeoff_frame,
        state.release_frame,
        state.release_measurement_frame,
        state.motion_peak_frame,
        state.contact_frame,
        len(observations.timestamps),
        state.motion_score_start_frame,
    ):
        if error not in metric_validation_errors:
            metric_validation_errors.append(error)
        if error in {
            "phase_order_takeoff_not_after_knee",
            "phase_order_takeoff_not_before_release",
            "phase_order_peak_before_takeoff",
            "phase_frame_out_of_range_takeoff",
        }:
            if velocity_start_frame == state.takeoff_frame:
                elbow_velocity = None
            state.takeoff_frame = None
            state.takeoff_trigger = None
            state.phase_statuses["lift"] = "unavailable"
            state.takeoff_knee_angle = None
            takeoff_ball_height = None
            state.takeoff_elbow_angle = None
            state.takeoff_upper_arm_angle = None
            state.takeoff_hip_angle = None
            motion_ratio_raw = motion_value = motion_class = None
        if error in {
            "phase_order_extension_not_after_knee",
            "phase_order_extension_not_before_release",
            "phase_frame_out_of_range_extension",
        }:
            if velocity_start_frame == state.extension_frame:
                elbow_velocity = None
            state.extension_frame = None
            state.extension_trigger = None
            extension_ball_height = state.extension_hip = None
            state.elbow_angle = state.upper_arm_angle = None
        if error in {
            "motion_peak_before_start",
            "phase_frame_out_of_range_peak",
        }:
            motion_ratio_raw = motion_value = motion_class = None
        if error in {"phase_order_contact_not_after_release", "phase_frame_out_of_range_contact"}:
            state.contact_frame = None
            contact_angle = None
            state.phase_statuses["rim_contact"] = "unavailable"

    if state.knee_frame is not None:
        prior_ball_frames = held_frames[held_frames < state.knee_frame]
        if len(prior_ball_frames):
            # Image y increases downward; use the lowest visible ball point before phase one.
            ball_points = observations.ball_centers_smoothed[prior_ball_frames]
            if camera_transform is not None:
                ball_points = apply_affine_transform(ball_points, camera_transform[prior_ball_frames])
            else:
                ball_points = ball_points.copy()
                ball_points[:, 1] -= camera_y[prior_ball_frames]
            ball_y = ball_points[:, 1]
            visible_ball = (
                np.isfinite(ball_y)
                & np.isfinite(observations.ball_scores[prior_ball_frames])
                & (observations.ball_scores[prior_ball_frames] >= CONFIG["BALL_DETECTION_THRESHOLD"])
            )
            if np.any(visible_ball):
                visible_frames = prior_ball_frames[visible_ball]
                state.ball_low_frame = int(visible_frames[np.argmax(ball_y[visible_ball])])

    # Preserve availability as machine-readable codes for every stage metric.
    # Missing values can be expected when the clip lacks the required event or
    # visibility; they should not make the audit claim full metric coverage.
    for metric_name, value in (
        ("knee_angle_deg", state.knee_angle),
        ("knee_ball_height_ratio", knee_ball_height),
        ("knee_elbow_angle_deg", state.knee_elbow),
        ("knee_upper_arm_angle_deg", state.knee_upper_arm),
        ("knee_hip_angle_deg", state.knee_hip),
        ("takeoff_ball_height_ratio", takeoff_ball_height),
        ("takeoff_elbow_angle_deg", state.takeoff_elbow_angle),
        ("takeoff_upper_arm_angle_deg", state.takeoff_upper_arm_angle),
        ("takeoff_hip_angle_deg", state.takeoff_hip_angle),
        ("release_height_ratio", release_height),
        ("release_foot_lift_ratio", release_foot_lift),
        ("max_jump_height_ratio", max_jump_height_ratio),
        ("release_angle_deg", state.release_angle),
        ("elbow_angular_velocity_deg_s", elbow_velocity),
        ("motion_value", motion_value),
    ):
        if value is None:
            code = f"metric_unavailable_{metric_name}"
            if code not in state.unavailable_metric_codes:
                state.unavailable_metric_codes.append(code)

    if max_jump_height_ratio is not None and state.max_jump_height_method == "foot_landmark_displacement":
        end_time = state.release_timestamp + CONFIG["MOTION_POST_RELEASE_WINDOW_SECONDS"]
        peak_window = (observations.timestamps >= observations.timestamps[state.release_frame]) & (
            observations.timestamps <= end_time
        )
        if np.any(observations.boxes[peak_window, 3] >= metadata.height - 1):
            state.metric_sources["max_jump_height_ratio"] = "visible_foot_trace_lower_bound"

    return ShotMetrics(
        rim_box=observations.rim_box,
        release_frame=state.release_frame,
        release_measurement_frame=state.release_measurement_frame,
        release_timestamp_sec=state.release_timestamp,
        knee_frame=state.knee_frame,
        ball_low_frame=state.ball_low_frame,
        extension_frame=state.extension_frame,
        extension_trigger=state.extension_trigger,
        takeoff_frame=state.takeoff_frame,
        takeoff_trigger=state.takeoff_trigger,
        takeoff_knee_angle_deg=state.takeoff_knee_angle,
        takeoff_ball_height_ratio=takeoff_ball_height,
        takeoff_elbow_angle_deg=state.takeoff_elbow_angle,
        takeoff_upper_arm_angle_deg=state.takeoff_upper_arm_angle,
        takeoff_hip_angle_deg=state.takeoff_hip_angle,
        body_peak_frame=state.body_peak_frame,
        max_jump_height_method=state.max_jump_height_method,
        contact_frame=state.contact_frame,
        metric_sources=state.metric_sources,
        shooting_side=state.shooting_side,
        shooting_side_confidence=state.shooting_side_confidence,
        shooting_side_method=state.shooting_side_method,
        shooting_side_evidence=state.shooting_side_evidence,
        knee_ball_height_ratio=knee_ball_height,
        knee_elbow_angle_deg=state.knee_elbow,
        knee_upper_arm_angle_deg=state.knee_upper_arm,
        knee_hip_angle_deg=state.knee_hip,
        extension_ball_height_ratio=extension_ball_height,
        extension_hip_angle_deg=state.extension_hip,
        release_height_ratio=release_height,
        release_angle_deg=state.release_angle,
        release_angle_sample_frames=state.release_angle_sample_frames,
        release_elbow_angle_deg=release_elbow,
        elbow_angular_velocity_deg_s=elbow_velocity,
        release_foot_lift_ratio=release_foot_lift,
        elbow_angle_deg=state.elbow_angle,
        upper_arm_angle_deg=state.upper_arm_angle,
        knee_angle_deg=state.knee_angle,
        max_jump_height_ratio=max_jump_height_ratio,
        motion_ratio_raw=motion_ratio_raw,
        motion_class=motion_class,
        motion_value=motion_value,
        contact_angle_deg=contact_angle,
        body_height_reference_px=body_height,
        body_height_reference_partial=body_height_reference_partial,
        ground_reference_y_px=ground_y,
        grounded_foot_baseline_y_px=grounded_foot_baseline_y,
        validation_errors=metric_validation_errors,
        unavailable_metric_codes=state.unavailable_metric_codes,
        video_profile=video_profile,
        phase_statuses=state.phase_statuses,
        motion_start_frame=state.motion_start_frame,
        motion_score_start_frame=state.motion_score_start_frame,
        motion_peak_frame=state.motion_peak_frame,
        angle_source_counts=angle_source_counts,
    )
