"""Shot analysis state, angle measurements, and body-relative metric calculations."""

import logging
from dataclasses import dataclass, field

import numpy as np

from analysis.kinematics import (
    body_height_reference_from_frames,
    build_jump_curve,
    contact_ball_center,
    estimate_torso_jump,
    highest_head_keypoint,
    pose_joint_angle_for_frame,
)
from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata, VideoProfile
from core.geometry import apply_affine_transform


class JointAngleMetrics:
    """Read, recover, and smooth joint angles while tracking their sources."""

    def __init__(self, observations: PoseObservations, tracked: TrackedPoses) -> None:
        self.observations = observations
        self.tracked = tracked
        self.source_counts: dict[str, dict[str, int]] = {}

    def at_frame(self, frame_index: int | None, side: str, name: str) -> float | None:
        """Return a valid pose angle and count the source that supplied it."""
        if frame_index is None:
            return None
        angle, source = pose_joint_angle_for_frame(
            self.observations, self.tracked, frame_index, side, name
        )
        if angle is not None and np.isfinite(angle):
            source_counts = self.source_counts.setdefault(f"{side}_{name}", {})
            source_counts[source] = source_counts.get(source, 0) + 1
        return angle

    def at_frames(
        self, frame_index: int | None, side: str | None, names: tuple[str, ...]
    ) -> tuple[float | None, ...]:
        """Return a group of joint angles at the same shot phase."""
        if side is None:
            return (None,) * len(names)
        return tuple(self.at_frame(frame_index, side, name) for name in names)

    def for_metric(
        self,
        frame_index: int | None,
        side: str | None,
        name: str,
        metric_id: str,
        metric_sources: dict[str, str],
    ) -> float | None:
        """Recover a missing metric angle from neighboring valid pose samples."""
        angle = self.at_frame(frame_index, side, name) if side is not None else None
        if angle is not None or frame_index is None or side is None:
            return angle

        radius = CONFIG["JOINT_ANGLE_METRIC_FALLBACK_FRAMES"]
        start = max(0, frame_index - radius)
        end = min(len(self.observations.timestamps), frame_index + radius + 1)
        samples = []
        for sample_frame in range(start, end):
            if sample_frame == frame_index:
                continue
            sample, source = pose_joint_angle_for_frame(
                self.observations, self.tracked, sample_frame, side, name
            )
            if sample is None or not np.isfinite(sample) or source == "2d_invalid":
                continue
            samples.append((sample_frame, float(sample)))
            source_counts = self.source_counts.setdefault(f"{side}_{name}", {})
            source_counts[source] = source_counts.get(source, 0) + 1

        if not samples:
            return None

        before = [sample for sample in samples if sample[0] < frame_index]
        after = [sample for sample in samples if sample[0] > frame_index]
        if before and after:
            before_frame, before_angle = max(before)
            after_frame, after_angle = min(after)
            timestamps = self.observations.timestamps
            before_time = timestamps[before_frame]
            after_time = timestamps[after_frame]
            weight = (timestamps[frame_index] - before_time) / (after_time - before_time)
            metric_sources[metric_id] = "neighbor_pose_angle_interpolation"
            return float(before_angle + weight * (after_angle - before_angle))

        sample_frame, value = min(samples, key=lambda sample: (abs(sample[0] - frame_index), sample[0]))
        metric_sources[metric_id] = "neighbor_pose_angle"
        return value

    def smooth_metric(
        self,
        frame_index: int,
        side: str,
        name: str,
        metric_id: str,
        current_value: float,
        metric_sources: dict[str, str],
    ) -> float:
        """Smooth low-confidence metric angles using nearby valid poses."""
        direct_angle, source = pose_joint_angle_for_frame(
            self.observations, self.tracked, frame_index, side, name
        )
        if (
            direct_angle is None
            or source == "2d_invalid"
            or not np.isclose(direct_angle, current_value, atol=1e-4)
        ):
            return current_value

        joint_names = {
            "elbow": ("shoulder", "elbow", "wrist"),
            "hip": ("shoulder", "hip", "knee"),
            "knee": ("hip", "knee", "ankle"),
            "shoulder": ("hip", "shoulder", "elbow"),
        }
        joint_indices = np.asarray(
            [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in joint_names[name]]
        )
        confidence = self.observations.confidence[frame_index, joint_indices]
        if (
            not np.isfinite(confidence).all()
            or float(np.min(confidence)) >= CONFIG["JOINT_ANGLE_SMOOTHING_CONFIDENCE_THRESHOLD"]
        ):
            return current_value

        radius = CONFIG["JOINT_ANGLE_METRIC_FALLBACK_FRAMES"]
        samples = []
        for sample_frame in range(
            max(0, frame_index - radius),
            min(len(self.observations.timestamps), frame_index + radius + 1),
        ):
            sample_confidence = self.observations.confidence[sample_frame, joint_indices]
            if (
                not self.tracked.valid[sample_frame, joint_indices].all()
                or not np.isfinite(sample_confidence).all()
                or np.min(sample_confidence) < CONFIG["KEYPOINT_THRESHOLD"]
            ):
                continue
            angle, sample_source = pose_joint_angle_for_frame(
                self.observations, self.tracked, sample_frame, side, name
            )
            if angle is not None and sample_source != "2d_invalid" and np.isfinite(angle):
                samples.append(float(angle))

        if len(samples) < 3:
            return current_value
        metric_sources[metric_id] = "neighbor_pose_angle_median"
        return float(np.median(samples))


@dataclass(slots=True)
class ShotAnalysisContext:
    """Video inputs and shared measurements for one analysis pass."""

    observations: PoseObservations
    tracked: TrackedPoses
    metadata: VideoMetadata
    video_profile: VideoProfile
    camera_y: np.ndarray
    camera_transform_to_reference_xy: np.ndarray | None
    angle_measurements: JointAngleMetrics
    body_height_anchor_frame: int | None
    preparation_frames: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def to_reference_xy(
        self,
        frame_indices: int | np.ndarray,
        points_xy: np.ndarray,
    ) -> np.ndarray:
        """Stabilize image points against the estimated camera zoom and pan."""
        transforms = self.camera_transform_to_reference_xy
        if transforms is not None:
            return apply_affine_transform(points_xy, transforms[frame_indices])
        points = np.asarray(points_xy, dtype=np.float32).copy()
        points[..., 1] -= self.camera_y[frame_indices]
        return points


@dataclass(slots=True)
class ShotAnalysisState:
    """Per-video event timeline and phase evidence accumulated during analysis."""

    release_frame: int | None = None
    release_measurement_frame: int | None = None
    release_timestamp: float | None = None
    shooting_side: str | None = None
    shooting_side_confidence: float = 0.0
    shooting_side_method: str | None = None
    shooting_side_evidence: dict[str, float] = field(default_factory=dict)
    release_angle: float | None = None
    release_angle_sample_frames: list[int] = field(default_factory=list)
    knee_frame: int | None = None
    knee_angle: float | None = None
    knee_samples: list[tuple[int, float, float | None]] = field(default_factory=list)
    knee_elbow: float | None = None
    knee_upper_arm: float | None = None
    knee_hip: float | None = None
    straight_frame: int | None = None
    ball_low_frame: int | None = None
    motion_start_frame: int | None = None
    extension_frame: int | None = None
    extension_trigger: str | None = None
    takeoff_frame: int | None = None
    takeoff_trigger: str | None = None
    toe_off_evidence_frame: int | None = None
    body_peak_frame: int | None = None
    max_jump_height_method: str | None = None
    contact_frame: int | None = None
    motion_score_start_frame: int | None = None
    motion_peak_frame: int | None = None
    no_confirmed_airborne_phase: bool = False
    phase_statuses: dict[str, str] = field(default_factory=dict)
    metric_sources: dict[str, str] = field(default_factory=dict)
    unavailable_metric_codes: list[str] = field(default_factory=list)
    anchor_frames: list[int] = field(default_factory=list)
    smoothed_torso_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    takeoff_knee_angle: float | None = None
    takeoff_elbow_angle: float | None = None
    takeoff_upper_arm_angle: float | None = None
    takeoff_hip_angle: float | None = None
    elbow_angle: float | None = None
    upper_arm_angle: float | None = None
    extension_hip: float | None = None
    unconfirmed_takeoff: bool = False

    @property
    def release_event_frame(self) -> int | None:
        """Return the first separated-ball frame used by event calculations."""
        if self.release_measurement_frame is not None:
            return self.release_measurement_frame
        return self.release_frame


@dataclass(slots=True)
class ShotBodyMetrics:
    """Body reference and height measurements produced by one analysis pass."""

    body_height_px: float | None = None
    body_height_reference_partial: bool = False
    ground_y_px: float | None = None
    grounded_foot_baseline_y_px: float | None = None
    knee_ball_height_ratio: float | None = None
    extension_ball_height_ratio: float | None = None
    takeoff_ball_height_ratio: float | None = None
    release_height_ratio: float | None = None
    release_foot_lift_ratio: float | None = None
    max_jump_height_ratio: float | None = None
    release_elbow_angle_deg: float | None = None
    torso_curve: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def ball_height_ratio(
        self,
        context: ShotAnalysisContext,
        state: ShotAnalysisState,
        index: int,
        metric_name: str,
    ) -> float | None:
        """Normalize a ball center by the measured body height."""
        if self.body_height_px is None or self.ground_y_px is None:
            return None
        center, source = contact_ball_center(
            context.observations,
            context.tracked,
            index,
            state.release_frame,
        )
        if center is None:
            return None
        if source != "observed_ball_center" or metric_name not in state.metric_sources:
            state.metric_sources[metric_name] = source
        reference_center = context.to_reference_xy(index, center)
        return float(
            max(self.ground_y_px - reference_center[1], 0.0)
            / self.body_height_px
        )


def measure_shot_body(
    context: ShotAnalysisContext,
    state: ShotAnalysisState,
    validation_errors: list[str],
) -> ShotBodyMetrics:
    """Measure grounded body reference, ball heights, and jump displacement."""
    observations = context.observations
    tracked = context.tracked
    metadata = context.metadata
    camera_y = context.camera_y
    angle_measurements = context.angle_measurements
    body_height_anchor_frame = context.body_height_anchor_frame
    preparation_frames = context.preparation_frames
    result = ShotBodyMetrics()
    if state.release_frame is not None:
        anchor_frame = (
            body_height_anchor_frame
            if body_height_anchor_frame is not None
            else state.takeoff_frame
            if state.takeoff_frame is not None
            else state.extension_frame
            if state.extension_frame is not None
            else state.motion_start_frame
            if state.motion_start_frame is not None
            else state.knee_frame
        )
        if anchor_frame is not None:
            prior_frames = preparation_frames[preparation_frames < anchor_frame]
            grounded_frames = prior_frames[-3:]
            if not len(grounded_frames):
                grounded_frames = np.asarray([anchor_frame])
            ground_boxes = observations.boxes[grounded_frames]
            valid_boxes = np.isfinite(ground_boxes).all(axis=1) & (ground_boxes[:, 3] > ground_boxes[:, 1])
            if np.any(valid_boxes):
                result.body_height_reference_partial = bool(np.all(ground_boxes[valid_boxes, 3] >= metadata.height - 1))
                (
                    result.body_height_px,
                    result.ground_y_px,
                    result.grounded_foot_baseline_y_px,
                ) = body_height_reference_from_frames(
                    observations,
                    tracked,
                    metadata,
                    camera_y,
                    grounded_frames,
                    [
                        frame
                        for frame in (state.release_measurement_frame, state.body_peak_frame)
                        if frame is not None
                    ],
                    context.camera_transform_to_reference_xy,
                )
                if result.ground_y_px is None:
                    valid_ground_boxes = ground_boxes[valid_boxes]
                    ground_points = np.column_stack(
                        (
                            (valid_ground_boxes[:, 0] + valid_ground_boxes[:, 2]) * 0.5,
                            valid_ground_boxes[:, 3],
                        )
                    )
                    ground_frames = grounded_frames[valid_boxes]
                    if context.camera_transform_to_reference_xy is not None:
                        ground_points = context.to_reference_xy(ground_frames, ground_points)
                    else:
                        ground_points[:, 1] -= camera_y[ground_frames]
                    result.ground_y_px = float(
                        np.median(ground_points[:, 1])
                    )
                head_tops = []
                for index in grounded_frames:
                    if not np.isfinite(observations.boxes[index]).all():
                        continue
                    head = highest_head_keypoint(tracked.analysis_keypoints[index])
                    if head is not None:
                        head_xy = context.to_reference_xy(index, head["xyz"][:2])
                        head_tops.append(float(head_xy[1]))
                if result.body_height_px is None and head_tops:
                    result.body_height_px = result.ground_y_px - float(np.percentile(head_tops, 5))
                if result.body_height_px is not None:
                    if result.body_height_px > 0:

                        result.knee_ball_height_ratio = (
                            result.ball_height_ratio(context, state, state.knee_frame, "knee_ball_height_ratio")
                            if state.knee_frame is not None
                            else None
                        )
                        result.extension_ball_height_ratio = (
                            result.ball_height_ratio(
                                context,
                                state,
                                state.extension_frame,
                                "extension_ball_height_ratio",
                            )
                            if state.extension_frame is not None
                            else None
                        )
                        result.takeoff_ball_height_ratio = (
                            result.ball_height_ratio(context, state, state.takeoff_frame, "takeoff_ball_height_ratio")
                            if state.takeoff_frame is not None else None
                        )
                        release_heights = [
                            value for index in (state.release_frame, state.release_frame + 1)
                            if index < len(observations.timestamps)
                            and (
                                value := result.ball_height_ratio(
                                    context,
                                    state,
                                    index,
                                    "release_height_ratio",
                                )
                            )
                            is not None
                        ]
                        result.release_height_ratio = float(np.mean(release_heights)) if release_heights else None

                        feet = np.full(len(observations.timestamps), np.nan, dtype=np.float32)
                        jump_foot_indices = CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"].get(state.shooting_side)
                        if jump_foot_indices is not None:
                            for index, frame_points in enumerate(tracked.analysis_keypoints):
                                visible = (
                                    tracked.valid[index, jump_foot_indices]
                                    & np.isfinite(frame_points[jump_foot_indices, :2]).all(axis=1)
                                    & (
                                        observations.confidence[index, jump_foot_indices]
                                        >= CONFIG["KEYPOINT_THRESHOLD"]
                                    )
                                )
                                if np.any(visible):
                                    foot_points = context.to_reference_xy(
                                        index,
                                        frame_points[jump_foot_indices[visible], :2],
                                    )
                                    feet[index] = np.max(foot_points[:, 1])
                        grounded_feet = feet[grounded_frames]
                        grounded_feet = grounded_feet[np.isfinite(grounded_feet)]
                        if len(grounded_feet):
                            jump_curve_analysis = build_jump_curve(
                                feet,
                                grounded_feet,
                                result.body_height_px,
                                metadata.fps,
                                state.release_frame,
                            )
                            ground_feet_y = jump_curve_analysis.ground_y_px
                            result.grounded_foot_baseline_y_px = ground_feet_y
                            jump_curve = jump_curve_analysis.values
                            result.release_foot_lift_ratio = jump_curve_analysis.release_lift_ratio
                            minimum_lift = jump_curve_analysis.minimum_lift_ratio
                            confirmed_lift = jump_curve_analysis.confirmed_lift_ratio

                            if state.takeoff_frame is None and state.knee_frame is not None:
                                candidate_start = max(
                                    state.knee_frame + 1,
                                    state.motion_start_frame
                                    if state.motion_start_frame is not None
                                    else state.knee_frame + 1,
                                    0,
                                )
                                candidate_end = min(
                                    len(jump_curve),
                                    int(
                                        np.searchsorted(
                                            observations.timestamps,
                                            state.release_timestamp + max(0.15, 4.0 / metadata.fps),
                                            side="right",
                                        )
                                    ),
                                )
                                for index in range(candidate_start, max(candidate_start, candidate_end - 1)):
                                    samples = jump_curve[index : index + 2]
                                    if np.isfinite(samples).all() and np.all(samples >= minimum_lift):
                                        state.takeoff_frame = min(index, state.release_frame)
                                        if index <= state.release_frame and float(np.min(samples)) >= confirmed_lift:
                                            state.takeoff_trigger = "持续脚部抬升"
                                            state.phase_statuses["lift"] = "confirmed"
                                        else:
                                            state.takeoff_trigger = "小幅离地估计"
                                            state.phase_statuses["lift"] = "estimated"
                                            metric_validation_errors.append("takeoff_estimated_low_lift")
                                        result.takeoff_ball_height_ratio = result.ball_height_ratio(
                                            context,
                                            state,
                                            state.takeoff_frame,
                                            "takeoff_ball_height_ratio",
                                        )
                                        state.unavailable_metric_codes = [
                                            code
                                            for code in state.unavailable_metric_codes
                                            if code != "takeoff_unconfirmed"
                                        ]
                                        break

                            if (
                                state.phase_statuses.get("lift") != "confirmed"
                                and state.takeoff_frame is None
                                and state.extension_frame is not None
                                and state.release_frame is not None
                                and (state.knee_frame is None or state.extension_frame > state.knee_frame)
                                and state.extension_frame < state.release_frame
                            ):
                                state.takeoff_frame = state.extension_frame
                                state.takeoff_trigger = "第二阶段回退"
                                state.phase_statuses["lift"] = "estimated"
                                result.takeoff_ball_height_ratio = result.ball_height_ratio(
                                    context,
                                    state,
                                    state.takeoff_frame,
                                    "takeoff_ball_height_ratio",
                                )
                                state.unavailable_metric_codes = [
                                            code
                                            for code in state.unavailable_metric_codes
                                            if code != "takeoff_unconfirmed"
                                ]

                            if state.takeoff_frame is not None:
                                if state.extension_frame is None:
                                    extension_candidates = [
                                        sample_index
                                        for sample_index, sample_angle, _ in state.knee_samples
                                        if sample_index > state.knee_frame
                                        and sample_angle >= CONFIG["KNEE_STRAIGHT_ANGLE_DEG"]
                                    ]
                                    state.extension_frame = (
                                        extension_candidates[0]
                                        if extension_candidates
                                        else state.takeoff_frame
                                    )
                                    state.extension_trigger = "膝盖伸直" if extension_candidates else "脚部抬升"
                                    state.elbow_angle = angle_measurements.at_frame(
                                        state.extension_frame,
                                        state.shooting_side,
                                        "elbow",
                                    )
                                    state.upper_arm_angle = angle_measurements.at_frame(
                                        state.extension_frame, state.shooting_side, "shoulder"
                                    )
                                    state.extension_hip = angle_measurements.at_frame(
                                        state.extension_frame,
                                        state.shooting_side,
                                        "hip",
                                    )
                                if state.takeoff_knee_angle is None:
                                    (
                                        state.takeoff_knee_angle,
                                        state.takeoff_elbow_angle,
                                        state.takeoff_upper_arm_angle,
                                        state.takeoff_hip_angle,
                                    ) = angle_measurements.at_frames(
                                        state.takeoff_frame, state.shooting_side, ("knee", "elbow", "shoulder", "hip")
                                    )

                            motion_search_end = min(
                                len(observations.timestamps),
                                int(
                                    np.searchsorted(
                                        observations.timestamps,
                                        state.release_timestamp + CONFIG["MOTION_POST_RELEASE_WINDOW_SECONDS"],
                                        side="right",
                                    )
                                ),
                            )
                            peak_anchor = (
                                state.takeoff_frame
                                if state.takeoff_frame is not None
                                else state.knee_frame
                                if state.knee_frame is not None
                                else 0
                            )
                            peak_start = min(peak_anchor, state.release_frame)
                            peak_candidates = np.arange(peak_start, motion_search_end)
                            peak_candidates = peak_candidates[np.isfinite(jump_curve[peak_candidates])]
                            if len(peak_candidates):
                                candidate_peak = int(peak_candidates[np.argmax(jump_curve[peak_candidates])])
                                state.motion_peak_frame = candidate_peak
                                candidate_lift = float(max(jump_curve[candidate_peak], 0.0))
                                result.max_jump_height_ratio = candidate_lift
                                if candidate_lift >= minimum_lift:
                                    state.body_peak_frame = candidate_peak
                                    state.max_jump_height_method = "foot_landmark_displacement"
                                else:
                                    state.body_peak_frame = None
                                    state.max_jump_height_method = "foot_trace_below_lift_threshold"
                                    state.phase_statuses["motion"] = "estimated"
                                state.phase_statuses["jump"] = (
                                    "confirmed"
                                    if candidate_lift >= CONFIG["JUMP_CONFIRM_HEIGHT_RATIO"]
                                    else "estimated"
                                )

                            if state.max_jump_height_method == "foot_landmark_displacement":
                                release_samples = jump_curve[state.release_frame : state.release_frame + 2]
                                release_samples = release_samples[np.isfinite(release_samples)]
                                result.release_foot_lift_ratio = (
                                    float(np.mean(release_samples)) if len(release_samples) else None
                                )

                            if result.release_foot_lift_ratio is not None and result.max_jump_height_ratio is not None:
                                if result.release_foot_lift_ratio > result.max_jump_height_ratio + 1e-6:
                                    metric_validation_errors.append("max_jump_below_release_jump")
                                    logging.error(
                                        "指标校验失败：出手时刻弹跳高度 %.3f 高于最大弹跳高度 %.3f。",
                                        result.release_foot_lift_ratio,
                                        result.max_jump_height_ratio,
                                    )
                                    result.max_jump_height_ratio = None
                                    state.body_peak_frame = None
                                    state.max_jump_height_method = None

                            release_event_candidate = (
                                state.release_measurement_frame
                                if state.release_measurement_frame is not None
                                else state.release_frame
                            )
                            minimum_peak_delay = round(
                                metadata.fps * CONFIG["NO_JUMP_MIN_PEAK_DELAY_SECONDS"]
                            )
                            supported_no_jump_evidence = (
                                state.phase_statuses.get("lift") != "confirmed"
                                and state.max_jump_height_method == "foot_landmark_displacement"
                                and result.release_foot_lift_ratio is not None
                                and result.release_foot_lift_ratio <= CONFIG["NO_JUMP_MAX_RELEASE_FOOT_LIFT_RATIO"]
                                and result.max_jump_height_ratio is not None
                                and result.max_jump_height_ratio >= CONFIG["NO_JUMP_MIN_PEAK_LIFT_RATIO"]
                                and result.max_jump_height_ratio
                                / max(result.release_foot_lift_ratio, 1.0 / result.body_height_px)
                                >= CONFIG["NO_JUMP_MIN_PEAK_TO_RELEASE_LIFT_RATIO"]
                                and state.motion_peak_frame is not None
                                and release_event_candidate is not None
                                and state.motion_peak_frame - release_event_candidate >= minimum_peak_delay
                                and state.knee_frame is not None
                                and state.shooting_side is not None
                            )
                            if supported_no_jump_evidence:
                                knee_extension_samples = []
                                for index in range(state.knee_frame + 1, release_event_candidate + 1):
                                    angle = angle_measurements.at_frame(index, state.shooting_side, "knee")
                                    if angle is not None:
                                        knee_extension_samples.append((angle, index))
                                if knee_extension_samples:
                                    state.no_confirmed_airborne_phase = True
                                    state.takeoff_frame = max(
                                        knee_extension_samples,
                                        key=lambda sample: (sample[0], -sample[1]),
                                    )[1]
                                    state.takeoff_trigger = "出手前膝盖最大伸展"
                                    (
                                        state.takeoff_knee_angle,
                                        state.takeoff_elbow_angle,
                                        state.takeoff_upper_arm_angle,
                                        state.takeoff_hip_angle,
                                    ) = angle_measurements.at_frames(
                                        state.takeoff_frame, state.shooting_side, ("knee", "elbow", "shoulder", "hip")
                                    )
                                    result.takeoff_ball_height_ratio = result.ball_height_ratio(
                                        context,
                                        state,
                                        state.takeoff_frame,
                                        "takeoff_ball_height_ratio",
                                    )
                                    state.motion_peak_frame = state.takeoff_frame
                                    state.body_peak_frame = None
                                    result.max_jump_height_ratio = 0.0
                                    result.release_foot_lift_ratio = 0.0
                                    state.max_jump_height_method = "no_confirmed_airborne_phase"
                                    state.phase_statuses["jump"] = "estimated"
                                    state.metric_sources["max_jump_height_ratio"] = state.max_jump_height_method
                                    state.metric_sources["release_foot_lift_ratio"] = state.max_jump_height_method

                            if result.max_jump_height_ratio is None:
                                state.body_peak_frame = None
                                state.max_jump_height_method = None
                                if "jump_unconfirmed" not in state.unavailable_metric_codes:
                                    state.unavailable_metric_codes.append("jump_unconfirmed")

        if (
            result.max_jump_height_ratio is None
            and result.body_height_px is not None
            and result.body_height_px > 0
            and state.release_frame is not None
            and state.release_timestamp is not None
        ):
            peak_anchor = state.takeoff_frame if state.takeoff_frame is not None else state.knee_frame or 0
            peak_start = min(peak_anchor, state.release_frame)
            search_end = min(
                len(observations.timestamps),
                int(
                    np.searchsorted(
                        observations.timestamps,
                        state.release_timestamp + CONFIG["MOTION_POST_RELEASE_WINDOW_SECONDS"],
                        side="right",
                    )
                ),
            )
            torso_analysis = estimate_torso_jump(
                observations,
                camera_y,
                state.anchor_frames,
                state.knee_frame,
                peak_start,
                search_end,
                result.body_height_px,
                state.smoothed_torso_y,
                context.camera_transform_to_reference_xy,
            )
            result.torso_curve = torso_analysis.curve
            if torso_analysis.peak_height_ratio is not None:
                state.motion_peak_frame = torso_analysis.peak_frame
                state.body_peak_frame = torso_analysis.peak_frame
                result.max_jump_height_ratio = torso_analysis.peak_height_ratio
                state.max_jump_height_method = "torso_center_peak_2d"
                state.phase_statuses["jump"] = "estimated"
                state.metric_sources["max_jump_height_ratio"] = state.max_jump_height_method
                state.unavailable_metric_codes = [
                    code for code in state.unavailable_metric_codes if code != "jump_unconfirmed"
                ]

        if (
            result.release_foot_lift_ratio is None
            and state.max_jump_height_method == "torso_center_peak_2d"
            and result.body_height_px is not None
            and result.body_height_px > 0
            and state.release_frame is not None
        ):
            torso_baseline = result.torso_curve[state.anchor_frames] if state.anchor_frames else np.empty(0)
            torso_baseline = torso_baseline[np.isfinite(torso_baseline)]
            release_torso = result.torso_curve[state.release_frame : state.release_frame + 2]
            release_torso = release_torso[np.isfinite(release_torso)]
            if len(torso_baseline) >= 3 and len(release_torso):
                torso_lift = float(
                    max(np.median(torso_baseline) - np.mean(release_torso), 0.0) / result.body_height_px
                )
                if result.max_jump_height_ratio is None or torso_lift <= result.max_jump_height_ratio + 1e-6:
                    result.release_foot_lift_ratio = torso_lift
                    state.metric_sources["release_foot_lift_ratio"] = "torso_center_at_release_2d"

        if state.shooting_side is not None:
            contact_elbow = angle_measurements.at_frame(state.release_frame, state.shooting_side, "elbow")
            separated_elbow = (
                angle_measurements.at_frame(state.release_measurement_frame, state.shooting_side, "elbow")
                if state.release_measurement_frame is not None
                else None
            )
            contact_time = float(observations.timestamps[state.release_frame])
            separated_time = float(observations.timestamps[state.release_measurement_frame])
            if contact_elbow is not None and separated_elbow is not None and separated_time > contact_time:
                interpolation = np.clip(
                    (state.release_timestamp - contact_time) / (separated_time - contact_time),
                    0.0,
                    1.0,
                )
                result.release_elbow_angle_deg = float(
                    contact_elbow + interpolation * (separated_elbow - contact_elbow)
                )
            else:
                result.release_elbow_angle_deg = contact_elbow if contact_elbow is not None else separated_elbow
    return result
