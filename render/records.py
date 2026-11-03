"""Prepare display data from shot-analysis inputs."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.kinematics import build_jump_curve, pose_joint_angle_from_arrays
from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata
from core.geometry import apply_affine_transform, finite_median_filter

_ANGLE_NAMES = ("knee", "hip", "shoulder", "elbow")
_FOREARM_ANGLE_TREND_TOLERANCE_DEG = 1.0
_BALL_SPEED_WINDOW_SECONDS = 0.25


@dataclass(frozen=True)
class ShotAnalysisView:
    """Analysis values arranged for a shot visualization."""

    metrics: dict[str, Any]
    frames: list[dict[str, Any]]
    side: str | None
    release_index: int
    has_release: bool
    timestamps: list[float]
    series: dict[str, list[float]]
    chart_start_index: int
    phase_markers: list[tuple[int, str]]
    keypoint_index: dict[str, int]


def _build_analysis_frames(
    metrics: dict[str, Any],
    observations: PoseObservations,
    tracked: TrackedPoses,
) -> tuple[list[dict[str, Any]], str | None]:
    camera_compensation_enabled = bool(
        metrics.get("scene_context", {}).get("camera_motion", {}).get("compensation_applied")
    )
    frames: list[dict[str, Any]] = []
    side = metrics.get("shooter", {}).get("shooting_hand")
    if side not in {"left", "right"}:
        side = None
    foot_indices = np.asarray(
        CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"][side]
        if side is not None
        else CONFIG["FOOT_KEYPOINT_INDICES"],
        dtype=np.int32,
    )
    for index, points in enumerate(tracked.analysis_keypoints):
        raw_points = observations.raw_keypoints[index]
        confidence = observations.confidence[index]
        valid = tracked.valid[index]
        bbox = observations.boxes[index]
        transform = (
            observations.camera_transform_to_reference_xy[index]
            if observations.camera_transform_to_reference_xy is not None
            else None
        )
        joint_angles = {name: None for name in _ANGLE_NAMES}
        if side is not None:
            for name in _ANGLE_NAMES:
                angle, source = pose_joint_angle_from_arrays(
                    points,
                    bbox,
                    confidence,
                    raw_points,
                    valid,
                    side,
                    name,
                )
                joint_angles[name] = (
                    float(angle)
                    if angle is not None and np.isfinite(angle) and source != "2d_invalid"
                    else None
                )
        frame = {
            "frame_index": index,
            "timestamp_sec": float(observations.timestamps[index]),
            "joint_angles_metric_deg": joint_angles,
            "ball_speed_body_heights_per_s": None,
            "keypoints_3d_filtered": points,
            "keypoint_confidence": confidence,
            "keypoint_track_valid": valid,
            "player_bbox_xyxy_px": bbox,
            "camera_shift_xy_px": observations.camera_shift_xy[index],
            "camera_transform_to_reference_xy": transform,
        }
        ball_center = observations.ball_centers_smoothed[index]
        if ball_center.shape == (2,) and np.isfinite(ball_center).all():
            if camera_compensation_enabled and transform is not None and transform.shape == (2, 3):
                ball_center = apply_affine_transform(ball_center, transform)
            frame["ball_center_xy_px"] = ball_center

        visible_feet = (
            valid[foot_indices]
            & np.isfinite(points[foot_indices, :2]).all(axis=1)
            & (confidence[foot_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
        )
        if np.any(visible_feet):
            foot_points = points[foot_indices[visible_feet], :2]
            if camera_compensation_enabled and transform is not None and transform.shape == (2, 3):
                foot_points = apply_affine_transform(foot_points, transform)
                foot_y = np.max(foot_points[:, 1])
            else:
                camera_shift = observations.camera_shift_xy[index]
                camera_y = float(camera_shift[1]) if camera_compensation_enabled else 0.0
                foot_y = np.max(foot_points[:, 1]) - camera_y
            frame["foot_y_compensated_px"] = float(foot_y)
        frames.append(frame)

    if not frames:
        raise ValueError("No frame observations are available for dashboard rendering")
    measurement_reference = metrics.get("measurement_reference", {})
    body_height = measurement_reference.get("body_height_px")
    body_height_px = float(body_height) if body_height is not None else None
    fps = float(metrics["video_metadata"].get("fps") or 30.0)
    speed_history_frames = max(round(fps * _BALL_SPEED_WINDOW_SECONDS), 2)
    speed_release_index = int(
        np.clip(phase_frame(metrics, "release", len(frames) - 1), 0, len(frames) - 1)
    )
    if body_height_px is not None and np.isfinite(body_height_px) and body_height_px > 0:
        for index, frame in enumerate(frames[: speed_release_index + 1]):
            if frame.get("ball_center_xy_px") is None:
                continue
            sample_indices = [
                sample_index
                for sample_index in range(max(0, index - speed_history_frames), index + 1)
                if frames[sample_index].get("ball_center_xy_px") is not None
            ]
            if len(sample_indices) < 2:
                continue
            sample_times = np.asarray(
                [float(frames[sample_index]["timestamp_sec"]) for sample_index in sample_indices],
                dtype=np.float64,
            )
            elapsed = sample_times - sample_times[-1]
            if np.ptp(sample_times) <= 1e-8:
                continue
            centers = np.asarray(
                [frames[sample_index]["ball_center_xy_px"] for sample_index in sample_indices],
                dtype=np.float64,
            )
            design = np.column_stack((elapsed, np.ones(len(elapsed))))
            velocity_xy = np.linalg.lstsq(design, centers, rcond=None)[0][0]
            frame["ball_speed_body_heights_per_s"] = float(
                np.linalg.norm(velocity_xy) / body_height_px
            )

    release_speed = frames[speed_release_index].get("ball_speed_body_heights_per_s")
    for frame in frames[speed_release_index + 1 :]:
        frame["ball_speed_body_heights_per_s"] = release_speed

    ground_feet_y = measurement_reference.get("grounded_foot_baseline_y_px")
    release_frame = metrics.get("calculation_details", {}).get("event_frames", {}).get(
        "release_last_contact"
    )
    if (
        body_height_px is not None
        and body_height_px > 0
        and ground_feet_y is not None
        and release_frame is not None
    ):
        foot_positions = np.asarray(
            [
                frame.get("foot_y_compensated_px", np.nan)
                for frame in frames
            ],
            dtype=np.float32,
        )
        release_frame = int(release_frame) - 1
        jump_curve = build_jump_curve(
            foot_positions,
            np.full(3, float(ground_feet_y), dtype=np.float32),
            body_height_px,
            fps,
            int(np.clip(release_frame, 0, len(frames) - 1)),
        ).values
        for frame, jump_height in zip(frames, jump_curve):
            frame["jump_height_ratio"] = float(jump_height) if np.isfinite(jump_height) else None
    return frames, side


def phase_frame(metrics: dict[str, Any], phase_id: str, default: int) -> int:
    """Return a phase frame as a zero-based index, or the supplied default."""
    phase = next((item for item in metrics.get("phases", []) if item.get("id") == phase_id), None)
    frame = phase.get("frame") if phase else None
    return int(frame) - 1 if frame is not None else default


def phase_metric(metrics: dict[str, Any], metric_id: str) -> float | None:
    """Return a metric value from the saved phase records."""
    for phase in metrics.get("phases", []):
        for metric in phase.get("metrics", []):
            if metric.get("id") == metric_id:
                value = metric.get("value")
                return float(value) if value is not None and np.isfinite(value) else None
    return None


def _angle_series(frames: list[dict[str, Any]]) -> dict[str, list[float]]:
    series = {name: [] for name in _ANGLE_NAMES}
    for frame in frames:
        angles = frame.get("joint_angles_metric_deg") or {}
        for name in series:
            value = angles.get(name)
            series[name].append(float(value) if value is not None and np.isfinite(value) else float("nan"))
    return series

def _smooth_angle_series(
    values: list[float],
    anchors: list[tuple[int, float]],
    radius: int,
) -> list[float]:
    raw = np.asarray(values, dtype=np.float32)
    raw[(raw < 0.0) | (raw > 180.0)] = np.nan
    filtered = finite_median_filter(raw, radius=radius)
    smoothed = np.full_like(filtered, np.nan)
    for index in range(len(filtered)):
        start, end = max(0, index - radius), min(len(filtered), index + radius + 1)
        sample_indices = np.arange(start, end)
        finite = np.isfinite(filtered[sample_indices])
        if not np.any(finite):
            continue
        sample_indices = sample_indices[finite]
        weights = radius + 1 - np.abs(sample_indices - index)
        smoothed[index] = np.average(filtered[sample_indices], weights=weights)

    valid_anchors = [
        (index, value)
        for index, value in anchors
        if 0 <= index < len(smoothed) and np.isfinite(value) and 0.0 <= value <= 180.0
    ]
    for index, value in valid_anchors:
        smoothed[index] = value
    return smoothed.astype(float).tolist()

def _smooth_shot_angles(
    series: dict[str, list[float]],
    metrics: dict[str, Any],
    selected_index: int,
) -> dict[str, list[float]]:
    details = metrics.get("calculation_details", {})
    event_frames = details.get("event_frames", {})
    event_measurements = details.get("event_measurements", {})
    metric_to_series = {
        "knee_angle_deg": "knee",
        "takeoff_knee_angle_deg": "knee",
        "knee_hip_angle_deg": "hip",
        "takeoff_hip_angle_deg": "hip",
        "extension_hip_angle_deg": "hip",
        "knee_upper_arm_angle_deg": "shoulder",
        "takeoff_upper_arm_angle_deg": "shoulder",
        "upper_arm_angle_deg": "shoulder",
        "knee_elbow_angle_deg": "elbow",
        "takeoff_elbow_angle_deg": "elbow",
        "elbow_angle_deg": "elbow",
        "release_elbow_angle_deg": "elbow",
    }
    anchors = {name: {} for name in series}
    for phase in metrics.get("phases", []):
        frame = phase.get("frame")
        if frame is None:
            continue
        for metric in phase.get("metrics", []):
            name = metric_to_series.get(metric.get("id"))
            value = metric.get("value")
            if name is not None and value is not None:
                anchors[name][int(frame) - 1] = float(value)

    extension_frame = event_frames.get("limb_extension")
    extension_measurements = event_measurements.get("limb_extension", {})
    if extension_frame is not None:
        for name, field in (
            ("hip", "hip_angle_deg"),
            ("shoulder", "upper_arm_angle_deg"),
            ("elbow", "forearm_angle_deg"),
        ):
            value = extension_measurements.get(field)
            if value is not None:
                anchors[name].setdefault(int(extension_frame) - 1, float(value))
    release_contact_frame = event_frames.get("release_last_contact")
    release_forearm_angle = event_measurements.get("release", {}).get("forearm_angle_deg")
    if release_contact_frame is not None and release_forearm_angle is not None:
        anchors["elbow"].setdefault(int(release_contact_frame) - 1, float(release_forearm_angle))

    fps = float(metrics["video_metadata"].get("fps") or 30.0)
    radius = max(2, min(4, round(fps * 0.1)))
    for name, values in anchors.items():
        if 0 <= selected_index < len(series[name]) and np.isfinite(series[name][selected_index]):
            values[selected_index] = series[name][selected_index]
        event_values = sorted(values.items())
        series[name] = _smooth_angle_series(series[name], event_values, radius)
    return series

def _preparation_event_marker(
    frames: list[dict[str, Any]],
    elbow_series: list[float],
    start_index: int,
    lift_index: int,
) -> tuple[int, str] | None:
    """Find a visible ball dip before lift, or a clear forearm-angle peak."""
    if lift_index - start_index < 3:
        return None

    ball_runs: list[list[tuple[int, float]]] = []
    current_run: list[tuple[int, float]] = []
    for index in range(start_index, lift_index):
        point_data = frames[index].get("ball_center_xy_px")
        if point_data is None:
            if current_run:
                ball_runs.append(current_run)
                current_run = []
            continue
        point = np.asarray(point_data, dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            if current_run:
                ball_runs.append(current_run)
                current_run = []
            continue
        current_run.append((index, float(point[1])))
    if current_run:
        ball_runs.append(current_run)

    dip_candidates = []
    for run in ball_runs:
        for position in range(1, len(run) - 1):
            _, y = run[position]
            if y >= run[position - 1][1] and y >= run[position + 1][1] and (
                y > run[position - 1][1] or y > run[position + 1][1]
            ):
                dip_candidates.append(run[position][0])
    for candidate_index in sorted(dip_candidates, reverse=True):
        angles = np.asarray(elbow_series[candidate_index : lift_index + 1], dtype=np.float32)
        if len(angles) < 2 or not np.isfinite(angles).all():
            continue
        angle_changes = np.diff(angles)
        if (
            np.all(angle_changes <= _FOREARM_ANGLE_TREND_TOLERANCE_DEG)
            and angles[0] - angles[-1] > _FOREARM_ANGLE_TREND_TOLERANCE_DEG
        ):
            return candidate_index, "dashboard.phase_dip"

    peak_index = max(
        range(start_index, min(lift_index, len(elbow_series))),
        key=lambda index: elbow_series[index] if np.isfinite(elbow_series[index]) else -np.inf,
        default=None,
    )
    if peak_index is None or peak_index <= start_index or peak_index >= lift_index - 1:
        return None
    peak_angle = elbow_series[peak_index]
    previous_angle = elbow_series[peak_index - 1]
    next_angle = elbow_series[peak_index + 1]
    if (
        np.isfinite(peak_angle)
        and np.isfinite(previous_angle)
        and np.isfinite(next_angle)
        and peak_angle > previous_angle
        and peak_angle > next_angle
    ):
        return peak_index, "dashboard.phase_forearm_peak"
    return None

def _chart_phase_markers(
    metrics: dict[str, Any],
    series: dict[str, list[float]],
    frames: list[dict[str, Any]],
    release_index: int,
) -> tuple[int, list[tuple[int, str]]]:
    """Return the curve origin and confirmed phase event markers."""
    phase_titles = {
        "load": "dashboard.phase_ball_lift",
        "lift": "dashboard.phase_rise",
        "release": "dashboard.phase_release",
    }
    phases = {phase.get("id"): phase for phase in metrics.get("phases", [])}
    markers = []
    load_index = None
    for phase_id, title_key in phase_titles.items():
        frame = phases.get(phase_id, {}).get("frame")
        if frame is None:
            continue
        frame_index = int(frame) - 1
        if not 0 <= frame_index <= release_index:
            continue
        markers.append((frame_index, title_key))
        if phase_id == "load":
            load_index = frame_index

    curve_start_index = 0
    if load_index is not None:
        preparation_marker = _preparation_event_marker(
            frames,
            series["elbow"],
            0,
            load_index,
        )
        if preparation_marker is not None:
            marker_index, title_key = preparation_marker
            markers.append((marker_index, title_key))
            if title_key == "dashboard.phase_dip":
                curve_start_index = marker_index
    return curve_start_index, markers


def prepare_shot_analysis_view(
    metrics: dict[str, Any],
    metadata: VideoMetadata,
    observations: PoseObservations,
    tracked: TrackedPoses,
) -> ShotAnalysisView:
    """Arrange the current analysis inputs for rendering."""
    frames, side = _build_analysis_frames(metrics, observations, tracked)
    frame_count = int(metrics["video_metadata"]["frame_count"])
    if len(frames) != frame_count or len(frames) != metadata.frame_count:
        raise ValueError("Analysis data and video metadata have different frame counts")

    release_event_index = phase_frame(metrics, "release", -1)
    has_release = 0 <= release_event_index < len(frames)
    release_index = release_event_index if has_release else len(frames) - 1
    timestamps = [float(frame["timestamp_sec"]) for frame in frames]
    series = _angle_series(frames)
    if side is not None:
        series = _smooth_shot_angles(series, metrics, release_index)
    chart_start_index, phase_markers = _chart_phase_markers(metrics, series, frames, release_index)
    keypoint_index = {name: index for index, name in enumerate(CONFIG["KEYPOINT_NAMES"])}
    return ShotAnalysisView(
        metrics=metrics,
        frames=frames,
        side=side,
        release_index=release_index,
        has_release=has_release,
        timestamps=timestamps,
        series=series,
        chart_start_index=chart_start_index,
        phase_markers=phase_markers,
        keypoint_index=keypoint_index,
    )
