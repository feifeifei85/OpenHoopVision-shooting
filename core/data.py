"""Shared data structures for video observations and shot metrics."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Frame rate, frame count, and dimensions of the source video."""

    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class PoseObservations:
    """Per-frame detections, pose estimates, and camera motion."""

    timestamps: np.ndarray
    boxes: np.ndarray
    detector_scores: np.ndarray
    ball_boxes: np.ndarray
    ball_centers: np.ndarray
    ball_centers_smoothed: np.ndarray
    ball_scores: np.ndarray
    ball_pre_release: np.ndarray
    rim_box: np.ndarray
    raw_keypoints: np.ndarray
    confidence: np.ndarray
    visualization_keypoints: np.ndarray
    target_diagnostics: dict[str, float]
    camera_shift_xy: np.ndarray
    camera_motion_confidence: np.ndarray
    rim_boxes: np.ndarray | None = None
    camera_transform_to_reference_xy: np.ndarray | None = None


@dataclass(frozen=True)
class VideoProfile:
    """Observed clip conditions used to route and audit later analysis stages."""

    view_mode: str
    view_confidence: float
    view_sample_count: int
    shoulder_span_to_torso_ratio: float | None
    face_offset_to_shoulder_ratio: float | None
    player_coverage: float
    ball_coverage: float
    camera_motion_coverage: float
    camera_vertical_drift_px: float
    camera_compensation_enabled: bool


@dataclass
class ShotMetrics:
    """Detected shooting events, measurements, and validation results."""

    rim_box: np.ndarray
    release_frame: int | None
    release_measurement_frame: int | None
    release_timestamp_sec: float | None
    knee_frame: int | None
    ball_low_frame: int | None
    extension_frame: int | None
    extension_trigger: str | None
    takeoff_frame: int | None
    takeoff_trigger: str | None
    takeoff_knee_angle_deg: float | None
    takeoff_ball_height_ratio: float | None
    takeoff_elbow_angle_deg: float | None
    takeoff_upper_arm_angle_deg: float | None
    takeoff_hip_angle_deg: float | None
    body_peak_frame: int | None
    max_jump_height_method: str | None
    contact_frame: int | None
    shooting_side: str | None
    knee_ball_height_ratio: float | None
    knee_elbow_angle_deg: float | None
    knee_upper_arm_angle_deg: float | None
    knee_hip_angle_deg: float | None
    extension_ball_height_ratio: float | None
    extension_hip_angle_deg: float | None
    release_height_ratio: float | None
    release_angle_deg: float | None
    release_angle_sample_frames: list[int]
    release_elbow_angle_deg: float | None
    elbow_angular_velocity_deg_s: float | None
    release_foot_lift_ratio: float | None
    elbow_angle_deg: float | None
    upper_arm_angle_deg: float | None
    knee_angle_deg: float | None
    max_jump_height_ratio: float | None
    motion_ratio_raw: float | None
    motion_class: str | None
    motion_value: float | None
    contact_angle_deg: float | None
    body_height_reference_px: float | None
    ground_reference_y_px: float | None
    grounded_foot_baseline_y_px: float | None
    validation_errors: list[str]
    unavailable_metric_codes: list[str]
    video_profile: VideoProfile
    phase_statuses: dict[str, str] = field(default_factory=dict)
    motion_start_frame: int | None = None
    motion_score_start_frame: int | None = None
    motion_peak_frame: int | None = None
    angle_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    shooting_side_confidence: float = 0.0
    shooting_side_method: str | None = None
    shooting_side_evidence: dict[str, float] = field(default_factory=dict)
    metric_sources: dict[str, str] = field(default_factory=dict)
    body_height_reference_partial: bool = False


@dataclass
class TrackedPoses:
    """Filtered pose trajectories and validity masks."""

    keypoints: np.ndarray
    valid: np.ndarray
    visualization_keypoints: np.ndarray
    analysis_keypoints: np.ndarray
