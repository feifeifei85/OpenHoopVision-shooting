"""Build shot reports, validate metrics, and apply reference annotations."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.kinematics import (
    ball_trajectory_angle,
    body_height_reference_from_frames,
    contact_ball_center,
    elbow_angular_velocity_between_frames,
    pose_joint_angle_for_frame,
)
from core.config import CONFIG
from core.data import PoseObservations, ShotMetrics, TrackedPoses, VideoMetadata
from core.geometry import apply_affine_transform


def calculate_shot_metrics(
    observations: PoseObservations,
    tracked: TrackedPoses,
    metadata: VideoMetadata,
) -> ShotMetrics:
    """Resolve the shooting analyzer lazily for reference recalculation."""
    from analysis.shots import calculate_shot_metrics as calculate

    return calculate(observations, tracked, metadata)


def motion_score_for_ratio(ratio: float) -> tuple[float, str]:
    """Resolve the motion scorer lazily for reference recalculation."""
    from analysis.shots import motion_score_for_ratio as score

    return score(ratio)


def shot_metric_phases(metrics: ShotMetrics, timestamps: np.ndarray) -> list[dict[str, object]]:
    """Convert shot measurements into ordered phase records."""
    phases = []

    def add_phase(
        phase_id: str,
        title: str,
        frame: int | None,
        trigger: str,
        measurements: list[tuple[str, str, float | None, str]],
        event_timestamp_sec: float | None = None,
        status: str | None = None,
    ) -> None:
        measurements_record = []
        for metric_id, label, value, unit in measurements:
            source = metrics.metric_sources.get(metric_id)
            estimated = source in {
                "hand_contact_offset", "interpolated_ball_center", "pre_release_hand_ball_line",
                "visible_foot_trace_lower_bound", "neighbor_pose_angle", "neighbor_pose_angle_interpolation",
                "neighbor_pose_angle_median", "neighbor_pose_angle_event_window",
                "torso_center_at_release_2d",
            }
            if unit == "倍身高" and metrics.body_height_reference_partial:
                unit = "倍可见身高"
                estimated = True
            if phase_id == "release" and label in {"出手时刻弹跳高度", "最大弹跳高度", "motion值"}:
                estimated = estimated or metrics.max_jump_height_method in {
                    "torso_center_peak_2d", "foot_trace_below_lift_threshold",
                }
                if label == "motion值":
                    estimated = estimated or any(
                        metrics.phase_statuses.get(phase) == "estimated" for phase in ("lift", "motion")
                    )
            measurements_record.append(
                {
                    "id": metric_id,
                    "label": label,
                    "source": source,
                    "value": float(value) if value is not None else None,
                    "unit": unit,
                    "status": "unavailable" if value is None else "estimated" if estimated else "measured",
                }
            )
            if source == "visible_foot_trace_lower_bound":
                measurements_record[-1]["bound"] = "lower"
        phases.append(
            {
                "id": phase_id,
                "title": title,
                "status": status or ("confirmed" if frame is not None else "unavailable"),
                "frame_index": int(frame) if frame is not None else None,
                "timestamp_sec": (
                    float(event_timestamp_sec if event_timestamp_sec is not None else timestamps[frame])
                    if frame is not None else None
                ),
                "trigger": trigger,
                "metrics": measurements_record,
            }
        )

    add_phase(
        "load",
        "蓄力与举球",
        metrics.knee_frame,
        "蓄力与举球",
        [
            ("knee_angle_deg", "膝盖弯曲度", metrics.knee_angle_deg, "°"),
            ("knee_ball_height_ratio", "球高度", metrics.knee_ball_height_ratio, "倍身高"),
            ("knee_elbow_angle_deg", "小臂夹角", metrics.knee_elbow_angle_deg, "°"),
            ("knee_upper_arm_angle_deg", "大臂夹角", metrics.knee_upper_arm_angle_deg, "°"),
            ("knee_hip_angle_deg", "髋部弯曲度", metrics.knee_hip_angle_deg, "°"),
        ],
        status=metrics.phase_statuses.get("load"),
    )
    add_phase(
        "lift",
        "起球",
        metrics.takeoff_frame,
        metrics.takeoff_trigger or "",
        [
            ("takeoff_ball_height_ratio", "球高度", metrics.takeoff_ball_height_ratio, "倍身高"),
            ("takeoff_elbow_angle_deg", "小臂夹角", metrics.takeoff_elbow_angle_deg, "°"),
            ("takeoff_upper_arm_angle_deg", "大臂夹角", metrics.takeoff_upper_arm_angle_deg, "°"),
            ("takeoff_hip_angle_deg", "髋部弯曲度", metrics.takeoff_hip_angle_deg, "°"),
        ],
        status=metrics.phase_statuses.get("lift"),
    )
    release_event_frame = (
        metrics.release_measurement_frame if metrics.release_measurement_frame is not None else metrics.release_frame
    )
    max_jump_label = "最大弹跳高度"
    add_phase(
        "release",
        "出手时刻",
        release_event_frame,
        "球离手",
        [
            ("release_height_ratio", "出手高度", metrics.release_height_ratio, "倍身高"),
            ("release_foot_lift_ratio", "出手时刻弹跳高度", metrics.release_foot_lift_ratio, "倍身高"),
            ("max_jump_height_ratio", max_jump_label, metrics.max_jump_height_ratio, "倍身高"),
            ("release_angle_deg", "出手角度", metrics.release_angle_deg, "°"),
            ("elbow_angular_velocity_deg_s", "甩臂角速度", metrics.elbow_angular_velocity_deg_s, "°/秒"),
            ("motion_value", "motion值", metrics.motion_value, ""),
        ],
        event_timestamp_sec=metrics.release_timestamp_sec,
        status=metrics.phase_statuses.get("release"),
    )
    add_phase(
        "rim_contact",
        "篮筐接触",
        metrics.contact_frame,
        "球碰篮筐",
        [
            ("entry_angle_deg", "进球角度", metrics.contact_angle_deg, "°"),
        ],
        status=metrics.phase_statuses.get("rim_contact"),
    )
    return phases


def shot_phase_time_intervals(
    metrics: ShotMetrics,
    timestamps: np.ndarray,
    phases: list[dict[str, object]],
) -> tuple[list[tuple[str, int]], dict[str, dict[str, float]]]:
    """Calculate elapsed-time intervals between detected shot phases."""
    timeline_events = [
        (str(phase["id"]), int(phase["frame_index"])) for phase in phases if phase["frame_index"] is not None
    ]
    if metrics.ball_low_frame is not None and 0 <= metrics.ball_low_frame < len(timestamps):
        timeline_events.append(("ball_low", metrics.ball_low_frame))
    timeline_events.sort(key=lambda event: event[1])

    intervals = {}
    for event_index, (event_id, end_frame) in enumerate(timeline_events):
        if event_id == "ball_low":
            continue
        start_frame = timeline_events[event_index - 1][1] if event_index else 0
        start_time_sec = float(timestamps[start_frame])
        end_time_sec = float(timestamps[end_frame])
        intervals[event_id] = {
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
            "duration_sec": max(end_time_sec - start_time_sec, 0.0),
        }
    return timeline_events, intervals


def shot_validation_codes(validation_errors: list[str]) -> list[str]:
    codes = []
    for error in validation_errors:
        if error.isascii() and error.replace("_", "").isalnum() and error.lower() == error:
            code = error
        elif error.startswith("motion 不可用"):
            code = "motion_unavailable"
        elif error.startswith("指标校验失败"):
            code = "metric_out_of_range"
        else:
            code = "metric_validation_failed"
        if code not in codes:
            codes.append(code)
    return codes


def shot_validation_error_codes(validation_errors: list[str]) -> list[str]:
    """Return validation codes that represent errors rather than notices."""
    notice_codes = {"takeoff_estimated_low_lift"}
    return [code for code in shot_validation_codes(validation_errors) if code not in notice_codes]


def build_shot_metrics_record(
    video_path: Path,
    metrics: ShotMetrics,
    observations: PoseObservations,
    metadata: VideoMetadata,
) -> dict[str, object]:
    """Build a per-video record with shot metrics and available video context."""
    phases = shot_metric_phases(metrics, observations.timestamps)
    _, phase_time_intervals = shot_phase_time_intervals(metrics, observations.timestamps, phases)
    for phase in phases:
        for measurement in phase["metrics"]:
            if measurement["value"] is not None:
                continue
            if phase["id"] == "rim_contact":
                reason = (
                    "entry_trajectory_not_reliable"
                    if metrics.contact_frame is not None
                    else "rim_not_observed"
                    if not np.isfinite(metrics.rim_box).all()
                    else "ball_rim_contact_not_observed"
                )
            elif phase["frame_index"] is None:
                reason = "phase_not_observed"
            elif measurement["label"] in {"球高度", "出手高度"}:
                reason = (
                    "body_height_reference_not_observed"
                    if metrics.body_height_reference_px is None
                    else "ball_not_observed_at_event"
                )
            elif measurement["label"] in {"最大弹跳高度", "出手时刻弹跳高度"}:
                reason = "grounded_feet_or_jump_peak_not_observed"
            elif measurement["label"] == "甩臂角速度":
                reason = "arm_extension_or_release_angle_not_observed"
            elif measurement["label"] == "出手角度":
                reason = "ball_trajectory_or_release_hand_not_observed"
            else:
                reason = "required_landmarks_not_observed"
            measurement["unavailable_reason"] = reason
    video_profile = metrics.video_profile
    camera_shift_x = observations.camera_shift_xy[:, 0]
    finite_camera_shift_x = camera_shift_x[np.isfinite(camera_shift_x)]
    horizontal_camera_drift = float(np.ptp(finite_camera_shift_x)) if len(finite_camera_shift_x) else 0.0
    relative_image_scale_range = None
    camera_transform = observations.camera_transform_to_reference_xy
    if video_profile.camera_compensation_enabled and camera_transform is not None:
        determinants = np.linalg.det(camera_transform[:, :, :2])
        valid_scales = np.isfinite(determinants) & (np.abs(determinants) > 1e-8)
        if np.any(valid_scales):
            relative_scales = 1.0 / np.sqrt(np.abs(determinants[valid_scales]))
            relative_image_scale_range = [
                float(np.min(relative_scales)),
                float(np.max(relative_scales)),
            ]
    duration_sec = metadata.frame_count / metadata.fps if metadata.fps > 0 else None
    rim_bbox = [float(value) for value in metrics.rim_box] if np.isfinite(metrics.rim_box).all() else None
    return {
        "schema_version": 2,
        "video_metadata": {
            "fps": float(metadata.fps),
            "frame_count": int(metadata.frame_count),
            "duration_sec": float(duration_sec) if duration_sec is not None else None,
            "width_px": int(metadata.width),
            "height_px": int(metadata.height),
        },
        "measurement_reference": {
            "body_height_px": metrics.body_height_reference_px,
            "ground_reference_y_px": metrics.ground_reference_y_px,
            "grounded_foot_baseline_y_px": metrics.grounded_foot_baseline_y_px,
            "body_height_visibility": (
                "unavailable" if metrics.body_height_reference_px is None
                else "partial" if metrics.body_height_reference_partial else "full"
            ),
            "height_ratio_reference": (
                None if metrics.body_height_reference_px is None
                else "visible_body_extent" if metrics.body_height_reference_partial else "body_height"
            ),
        },
        "scene_context": {
            "player_orientation_to_camera": {
                "mode": video_profile.view_mode,
                "status": "estimated" if video_profile.view_mode != "unknown" else "unavailable",
                "confidence": float(video_profile.view_confidence),
                "sample_count": int(video_profile.view_sample_count),
                "shoulder_span_to_torso_ratio": video_profile.shoulder_span_to_torso_ratio,
                "face_offset_to_shoulder_ratio": video_profile.face_offset_to_shoulder_ratio,
                "interpretation": "Player body view relative to camera; not an absolute left/right heading.",
            },
            "camera_motion": {
                "frame_coverage": float(video_profile.camera_motion_coverage),
                "horizontal_drift_px": horizontal_camera_drift,
                "vertical_drift_px": float(video_profile.camera_vertical_drift_px),
                "compensation_applied": bool(video_profile.camera_compensation_enabled),
                "zoom_compensation_applied": bool(
                    video_profile.camera_compensation_enabled and camera_transform is not None
                ),
                "relative_image_scale_range_from_first_frame": relative_image_scale_range,
            },
            "detection_coverage": {
                "player": float(video_profile.player_coverage),
                "ball": float(video_profile.ball_coverage),
            },
            "rim_bbox_xyxy_px": rim_bbox,
        },
        "shooter": {
            "shooting_hand": metrics.shooting_side,
            "shooting_hand_status": "estimated" if metrics.shooting_side is not None else "unavailable",
            "shooting_hand_confidence": metrics.shooting_side_confidence,
            "shooting_hand_confidence_type": "relative_score_margin",
            "shooting_hand_method": metrics.shooting_side_method,
            "shooting_hand_evidence": metrics.shooting_side_evidence,
            "association": {
                name: float(value) if np.isfinite(value) else None
                for name, value in observations.target_diagnostics.items()
            },
        },
        "video": str(video_path),
        "calculation_details": {
            "event_frames": {
                "release_last_contact": metrics.release_frame + 1 if metrics.release_frame is not None else None,
                "limb_extension": metrics.extension_frame + 1 if metrics.extension_frame is not None else None,
            },
            "event_measurements": {
                "limb_extension": {
                    "trigger": metrics.extension_trigger,
                    "forearm_angle_deg": metrics.elbow_angle_deg,
                    "upper_arm_angle_deg": metrics.upper_arm_angle_deg,
                    "hip_angle_deg": metrics.extension_hip_angle_deg,
                },
                "release": {
                    "forearm_angle_deg": metrics.release_elbow_angle_deg,
                },
            },
            "motion": {
                "class": metrics.motion_class,
                "peak_frame": metrics.motion_peak_frame + 1 if metrics.motion_peak_frame is not None else None,
            },
            "jump_height": {
                "method": metrics.max_jump_height_method,
                "maximum_frame": (
                    metrics.body_peak_frame + 1 if metrics.body_peak_frame is not None
                    else metrics.motion_peak_frame + 1 if metrics.motion_peak_frame is not None
                    else None
                ),
            },
        },
        "phases": [
            {
                "id": phase["id"],
                "name": phase["title"],
                "status": phase["status"],
                "frame": (int(phase["frame_index"]) + 1 if phase["frame_index"] is not None else None),
                "time_sec": phase["timestamp_sec"],
                **phase_time_intervals.get(
                    phase["id"],
                    {
                        "start_time_sec": None,
                        "end_time_sec": None,
                        "duration_sec": None,
                    },
                ),
                "metrics": phase["metrics"],
            }
            for phase in phases
        ],
        "validation_errors": shot_validation_error_codes(metrics.validation_errors),
        "validation_details": list(metrics.validation_errors),
    }


def validate_metric_values(
    metric_values: dict[str, float | None],
    metric_validation_errors: list[str],
    extension_frame: int | None,
    release_timestamp: float | None,
    timestamps: np.ndarray,
    *,
    velocity_start_frame: int | None = None,
    velocity_end_frame: int | None = None,
) -> None:
    """Replace out-of-range measurements with None and record their errors."""
    metric_ranges = {
        "knee_ball_height_ratio": (0.0, 2.0),
        "extension_ball_height_ratio": (0.0, 2.0),
        "takeoff_ball_height_ratio": (0.0, 2.0),
        "release_height_ratio": (0.0, 2.0),
        "release_foot_lift_ratio": (0.0, 0.75),
        "max_jump_height_ratio": (0.0, 0.75),
        "knee_elbow_angle_deg": (30.0, 180.0),
        "knee_upper_arm_angle_deg": (0.0, 180.0),
        "knee_hip_angle_deg": (30.0, 180.0),
        "takeoff_elbow_angle_deg": (30.0, 180.0),
        "takeoff_upper_arm_angle_deg": (0.0, 180.0),
        "takeoff_hip_angle_deg": (30.0, 180.0),
        "extension_hip_angle_deg": (30.0, 180.0),
        "release_elbow_angle_deg": (30.0, 180.0),
        "elbow_angle_deg": (30.0, 180.0),
        "upper_arm_angle_deg": (0.0, 180.0),
        "knee_angle_deg": (20.0, 180.0),
        "takeoff_knee_angle_deg": (20.0, 180.0),
        "release_angle_deg": (0.0, 90.0),
        "contact_angle_deg": (0.0, 90.0),
        "motion_ratio_raw": (0.0, 2.25),
        "motion_value": (CONFIG["MOTION_VALUE_MIN"], CONFIG["MOTION_VALUE_MAX"]),
    }
    for metric_name, (minimum, maximum) in metric_ranges.items():
        value = metric_values[metric_name]
        if value is not None and (not np.isfinite(value) or not minimum <= value <= maximum):
            message = f"指标校验失败：{metric_name}={value!r}，合理范围为 [{minimum}, {maximum}]。"
            metric_validation_errors.append(message)
            logging.error(message)
            metric_values[metric_name] = None

    elbow_velocity_value = metric_values["elbow_angular_velocity_deg_s"]
    if elbow_velocity_value is not None:
        if not np.isfinite(elbow_velocity_value) or elbow_velocity_value < 0:
            message = f"指标校验失败：甩臂角速度为 {elbow_velocity_value!r}，伸肘速度必须非负。"
            metric_validation_errors.append(message)
            logging.error(message)
            metric_values["elbow_angular_velocity_deg_s"] = None
        elif velocity_start_frame is not None and velocity_end_frame is not None:
            elapsed = float(timestamps[velocity_end_frame] - timestamps[velocity_start_frame])
            if elapsed > 0 and elbow_velocity_value > 180.0 / elapsed:
                message = f"指标校验失败：甩臂角速度 {elbow_velocity_value:.1f}°/秒超过角度区间与采样时间允许的上限。"
                metric_validation_errors.append(message)
                logging.error(message)
                metric_values["elbow_angular_velocity_deg_s"] = None
        elif extension_frame is not None and release_timestamp is not None:
            elapsed = release_timestamp - float(timestamps[extension_frame])
            if elapsed > 0 and elbow_velocity_value > 180.0 / elapsed:
                message = f"指标校验失败：甩臂角速度 {elbow_velocity_value:.1f}°/秒超过角度区间与采样时间允许的上限。"
                metric_validation_errors.append(message)
                logging.error(message)
                metric_values["elbow_angular_velocity_deg_s"] = None


def validate_phase_event_order(
    knee_frame: int | None,
    extension_frame: int | None,
    takeoff_frame: int | None,
    release_frame: int | None,
    release_event_frame: int | None,
    peak_frame: int | None,
    contact_frame: int | None,
    frame_count: int,
    motion_start_frame: int | None = None,
) -> list[str]:
    """Return stable codes for impossible event order without judging metric values."""
    events = {
        "knee": knee_frame,
        "extension": extension_frame,
        "takeoff": takeoff_frame,
        "release": release_frame,
        "release_event": release_event_frame,
        "peak": peak_frame,
        "contact": contact_frame,
        "motion_start": motion_start_frame,
    }
    errors = []
    for name, frame in events.items():
        if frame is not None and not 0 <= frame < frame_count:
            errors.append(f"phase_frame_out_of_range_{name}")
    if knee_frame is not None and takeoff_frame is not None and takeoff_frame <= knee_frame:
        errors.append("phase_order_takeoff_not_after_knee")
    if takeoff_frame is not None and release_event_frame is not None and takeoff_frame >= release_event_frame:
        errors.append("phase_order_takeoff_not_before_release")
    if knee_frame is not None and extension_frame is not None and extension_frame <= knee_frame:
        errors.append("phase_order_extension_not_after_knee")
    if extension_frame is not None and release_event_frame is not None and extension_frame >= release_event_frame:
        errors.append("phase_order_extension_not_before_release")
    if takeoff_frame is not None and peak_frame is not None and peak_frame < takeoff_frame:
        errors.append("phase_order_peak_before_takeoff")
    if motion_start_frame is not None and peak_frame is not None and peak_frame < motion_start_frame:
        errors.append("motion_peak_before_start")
    if release_frame is not None and contact_frame is not None and contact_frame <= release_frame:
        errors.append("phase_order_contact_not_after_release")
    return errors


def recalculate_reference_metrics(
    observations: PoseObservations,
    tracked: TrackedPoses,
    metadata: VideoMetadata,
    labels: dict,
    *,
    extension_frame_1based: int | None = None,
) -> ShotMetrics:
    """Use source-frame references for events and model observations for values.

    Unspecified events retain their current reference, including AI estimates.
    Frame intervals use their midpoint for measurement, not as an exact truth.
    Image displacements and video-time velocities remain estimates.
    """
    metrics = calculate_shot_metrics(observations, tracked, metadata)
    changes = {}
    events = {}
    for name in ("knee_low", "takeoff", "release", "jump_apex"):
        bounds = labels[name].get("frame_interval")
        frame = (sum(bounds) // 2 - 1) if bounds else None
        if frame is not None and not 0 <= frame < len(observations.timestamps):
            raise ValueError(f"Reference frame outside source: {name}")
        events[name] = frame
    knee, takeoff, release, peak = (events[k] for k in ("knee_low", "takeoff", "release", "jump_apex"))
    side = labels["shooting_hand"].get("value") or metrics.shooting_side
    extension = extension_frame_1based - 1 if extension_frame_1based is not None else None
    if extension is None and knee is not None and release is not None:
        candidates = []
        for index in range(knee + 1, release):
            angle, _ = pose_joint_angle_for_frame(observations, tracked, index, side, "elbow")
            if angle is not None:
                candidates.append((angle, index))
        extension = max(candidates)[1] if candidates else None
    changes.update(
        knee_frame=knee, takeoff_frame=takeoff, release_frame=release,
        release_measurement_frame=release,
        release_timestamp_sec=float(observations.timestamps[release]) if release is not None else None,
        body_peak_frame=peak, motion_peak_frame=peak, extension_frame=extension,
        extension_trigger="reference_knee_lockout" if extension_frame_1based else "maximum_pre_release_elbow_opening",
        takeoff_trigger="reference_event" if takeoff is not None else None,
        shooting_side=side, ball_low_frame=None, validation_errors=[], unavailable_metric_codes=[],
        motion_start_frame=knee, motion_score_start_frame=takeoff if takeoff is not None else extension,
    )
    sources = {}
    angle_counts = {}
    for prefix, frame in (("knee", knee), ("takeoff", takeoff), ("extension", extension), ("release", release)):
        for joint, suffix in (("knee", "knee_angle_deg"), ("elbow", "elbow_angle_deg"),
                              ("shoulder", "upper_arm_angle_deg"), ("hip", "hip_angle_deg")):
            value, source = (pose_joint_angle_for_frame(observations, tracked, frame, side, joint)
                             if frame is not None and side is not None else (None, "unavailable"))
            angle_counts[f"{prefix}_{joint}"] = {source: 1}
            field = f"{prefix}_{suffix}"
            if prefix == "knee" and joint == "knee":
                field = "knee_angle_deg"
            elif prefix == "extension" and joint in {"elbow", "shoulder"}:
                field = "elbow_angle_deg" if joint == "elbow" else "upper_arm_angle_deg"
            if hasattr(metrics, field):
                changes[field] = value
    changes["angle_source_counts"] = angle_counts
    camera = observations.camera_shift_xy if metrics.video_profile.camera_compensation_enabled else np.zeros(
        (len(observations.timestamps), 2)
    )
    camera_transform = (
        observations.camera_transform_to_reference_xy
        if metrics.video_profile.camera_compensation_enabled
        else None
    )
    # A fixed landmark is compared across frames; head and shoe series are not mixed.
    feet = np.full(len(observations.timestamps), np.nan)
    head = np.full_like(feet, np.nan)
    for index, points in enumerate(tracked.analysis_keypoints):
        foot_values = []
        for foot_side in CONFIG["SIDE_NAMES"]:
            indices = np.asarray(CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"][foot_side])
            good = (
                tracked.valid[index, indices] & np.isfinite(points[indices, :2]).all(axis=1)
                & (observations.confidence[index, indices] >= CONFIG["KEYPOINT_THRESHOLD"])
                & (points[indices, 1] >= 0) & (points[indices, 1] < metadata.height - 1)
                & (points[indices, 0] >= 0) & (points[indices, 0] < metadata.width - 1)
            )
            if np.any(good):
                visible_feet = points[indices[good], :2]
                if camera_transform is not None:
                    visible_feet = apply_affine_transform(visible_feet, camera_transform[index])
                    foot_values.append(float(np.max(visible_feet[:, 1])))
                else:
                    foot_values.append(float(np.max(visible_feet[:, 1] - camera[index, 1])))
        if len(foot_values) == 2 and observations.boxes[index, 3] < metadata.height - 1:
            feet[index] = np.mean(foot_values)
        nose = CONFIG["KEYPOINT_INDEX"]["nose"]
        if tracked.valid[index, nose] and observations.confidence[index, nose] >= CONFIG["KEYPOINT_THRESHOLD"]:
            if camera_transform is not None:
                head[index] = apply_affine_transform(points[nose, :2], camera_transform[index])[1]
            else:
                head[index] = points[nose, 1] - camera[index, 1]
    anchor = takeoff if takeoff is not None else extension
    baseline_frames = np.arange(max(0, (anchor or 0) - 3), anchor or 0)
    # Choose one facial landmark for the entire displacement measurement.
    # An occluded nose must not force frame-to-frame switching between face points.
    required_frames = list(baseline_frames) + [f for f in (release, peak) if f is not None]
    if required_frames:
        candidates = []
        for index in range(5):
            good = tracked.valid[:, index] & (
                observations.confidence[:, index] >= CONFIG["KEYPOINT_THRESHOLD"]
            ) & np.isfinite(tracked.analysis_keypoints[:, index, 1])
            candidates.append((int(np.count_nonzero(good[required_frames])), -index, good))
        _, negative_index, good = max(candidates, key=lambda candidate: candidate[:2])
        if camera_transform is not None:
            stabilized_head = apply_affine_transform(
                tracked.analysis_keypoints[:, -negative_index, :2],
                camera_transform,
            )
            head = np.where(good, stabilized_head[:, 1], np.nan)
        else:
            head = np.where(good, tracked.analysis_keypoints[:, -negative_index, 1] - camera[:, 1], np.nan)
    height, ground, grounded_foot_baseline_y = body_height_reference_from_frames(
        observations,
        tracked,
        metadata,
        camera[:, 1],
        baseline_frames,
        [frame for frame in (release, peak) if frame is not None],
        camera_transform,
    )
    if height is None or height <= 0:
        height = metrics.body_height_reference_px
        ground = metrics.ground_reference_y_px
    changes.update(body_height_reference_px=height, ground_reference_y_px=ground,
                   grounded_foot_baseline_y_px=grounded_foot_baseline_y)
    for name, frame in (("knee_ball_height_ratio", knee), ("takeoff_ball_height_ratio", takeoff),
                        ("extension_ball_height_ratio", extension), ("release_height_ratio", release)):
        center, source = (contact_ball_center(observations, tracked, frame, release)
                          if frame is not None and release is not None else (None, None))
        camera_offset = camera[frame, 1] if camera_transform is None else 0.0
        changes[name] = float(max(0, ground - center[1] + camera_offset) / height) if (
            center is not None and ground is not None and height is not None and height > 0
        ) else None
        if source:
            sources[name] = source
    jump = labels["jump_present"].get("value")
    proxy = "head" if labels["jump_apex"].get("method") == "human_head_position_estimate" else "feet"
    series = head if proxy == "head" else feet
    baseline = series[baseline_frames]
    baseline = baseline[np.isfinite(baseline)]
    if not len(baseline) or peak is None or not np.isfinite(series[peak]):
        proxy, series, baseline = "head", head, baseline_head
    for name, frame in (("max_jump_height_ratio", peak), ("release_foot_lift_ratio", release)):
        value = None
        if jump is False:
            value = 0.0
        elif (
            takeoff is not None and frame is not None and len(baseline)
            and height is not None and height > 0 and np.isfinite(series[frame])
        ):
            value = float(max(0, np.median(baseline) - series[frame]) / height)
        changes[name] = value
        sources[name] = "human_confirmed_no_airborne_phase" if jump is False else f"reference_{proxy}_displacement_2d"
    changes["max_jump_height_method"] = sources["max_jump_height_ratio"]
    if changes["max_jump_height_ratio"] is not None and changes["release_foot_lift_ratio"] is not None:
        # Foot posture can make the release sample higher than the body-apex sample.
        # Report the maximum sampled proxy without changing the annotated body-apex event.
        changes["max_jump_height_ratio"] = max(
            changes["max_jump_height_ratio"], changes["release_foot_lift_ratio"]
        )
        if jump is not False:
            sources["max_jump_height_ratio"] = f"reference_{proxy}_event_sample_maximum_2d"
            changes["max_jump_height_method"] = sources["max_jump_height_ratio"]
    centers = observations.ball_centers - camera
    for name, frame, launch in (("release_angle_deg", release, True),):
        angle, samples = (ball_trajectory_angle(centers, observations.timestamps, frame, 0, 6, launch,
                                              confidence=observations.ball_scores)
                          if frame is not None else (None, []))
        changes[name] = angle
        changes["release_angle_sample_frames"] = samples
        sources[name] = "post_release_ball_trajectory_2d"
    start = takeoff if takeoff is not None else extension
    motion_peak = peak
    if jump is False:
        start = knee
        motion_peak = extension if extension_frame_1based is not None else None
        motion_peak_is_valid = (
            knee is not None
            and release is not None
            and motion_peak is not None
            and knee < motion_peak <= release
        )
        if (
            knee is not None
            and release is not None
            and not motion_peak_is_valid
            and side is not None
        ):
            knee_extensions = []
            for frame in range(knee + 1, release + 1):
                angle, source = pose_joint_angle_for_frame(
                    observations, tracked, frame, side, "knee"
                )
                if angle is not None and source != "2d_invalid":
                    knee_extensions.append((angle, frame))
            motion_peak = (
                max(knee_extensions, key=lambda sample: (sample[0], -sample[1]))[1]
                if knee_extensions
                else None
            )
        changes.update(
            motion_score_start_frame=start,
            motion_peak_frame=motion_peak,
        )
    changes.update(elbow_angular_velocity_deg_s=None, motion_ratio_raw=None, motion_value=None, motion_class=None)
    velocity_start = takeoff if takeoff is not None else extension
    velocity_start_sample = None
    if release is not None and velocity_start is not None and side is not None:
        velocity, velocity_start_sample, _, start_angle, _ = elbow_angular_velocity_between_frames(
            observations, tracked, side, velocity_start, release
        )
        changes["elbow_angular_velocity_deg_s"] = velocity
        if takeoff is not None and start_angle is not None:
            changes["takeoff_elbow_angle_deg"] = start_angle
        if velocity is not None:
            sources["elbow_angular_velocity_deg_s"] = "source_video_time_not_physical_speed"
    if (
        start is not None
        and motion_peak is not None
        and release is not None
        and start < motion_peak
        and (jump is not False or motion_peak <= release)
        and release >= start
    ):
        ratio = float((release - start) / (motion_peak - start))
        score, motion_class = motion_score_for_ratio(ratio)
        changes.update(
            motion_ratio_raw=ratio,
            motion_value=float(np.clip(score, CONFIG["MOTION_VALUE_MIN"], CONFIG["MOTION_VALUE_MAX"])),
            motion_class=motion_class,
        )
        if jump is False:
            sources["motion_value"] = "no_jump_knee_extension_timing"
    # Entry intervals supplied by reviewers are not physical metal-contact evidence.
    changes.update(contact_frame=None, contact_angle_deg=None, metric_sources=sources)
    statuses = {name: "estimated" if frame is not None else "unavailable" for name, frame in (
        ("knee_low", knee), ("takeoff", takeoff), ("release", release)
    )}
    for name in statuses:
        if labels[name].get("provenance", {}).get("reviewer_kind") == "human" and labels[name]["status"] == "reviewed":
            statuses[name] = "confirmed"
    statuses.update(rim_contact="unavailable", jump="estimated", motion="estimated")
    changes["phase_statuses"] = {
        {"knee_low": "load", "takeoff": "lift"}.get(name, name): status for name, status in statuses.items()
    }
    result = replace(metrics, **changes)
    values = dict(vars(result))
    validate_metric_values(
        values,
        result.validation_errors,
        extension,
        result.release_timestamp_sec,
        observations.timestamps,
        velocity_start_frame=velocity_start_sample,
        velocity_end_frame=release,
    )
    return replace(result, **values)
