"""Render annotated analysis videos."""

from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from analysis.report import shot_metric_phases
from core.config import CONFIG
from core.data import PoseObservations, ShotMetrics, TrackedPoses, VideoMetadata
from core.geometry import apply_affine_transform
from render.metrics import render_metrics_panel
from render.panels import (
    draw_angle_overlays,
    draw_foot_height_markers,
    draw_source_pose,
    render_3d_panel,
)


def frames_with_pause(
    frame: np.ndarray,
    frame_index: int,
    pause_frames: set[int],
    pause_frame_count: int,
) -> Iterator[np.ndarray]:
    """Yield a frame and its hold frames when it marks a shot event."""
    yield frame
    for _ in range(pause_frame_count if frame_index in pause_frames else 0):
        yield frame


def render_pose_video(
    video_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    observations: PoseObservations,
    tracked: TrackedPoses,
    metrics: ShotMetrics,
    depth_scale: float,
    *,
    annotation: dict | None = None,
    lang: str = "en",
) -> None:
    """Arrange the source video and analysis panels based on video orientation."""
    with ExitStack() as resources:
        capture = cv2.VideoCapture(str(video_path))
        resources.callback(capture.release)
        if not capture.isOpened():
            raise OSError(f"Could not reopen video for rendering: {video_path}")
        callout_background = red_callout_background(video_path, metadata) if annotation is None else None

        portrait_layout = metadata.height > metadata.width
        if portrait_layout:
            output_height = metadata.height + metadata.height % 2
            output_width = int(round(output_height * 4 / 3))
            output_width += output_width % 2
            video_height, video_width = output_height, metadata.width
            metrics_width = output_width - metadata.width
            metrics_scale = min(metrics_width / 440, 1.6)
            metrics_height = min(int(output_height * 0.75), max(640, round(640 * metrics_scale)))
            pose_height = output_height - metrics_height
            skeleton_width = metrics_width
        else:
            output_width = metadata.width
            output_height = int(round(output_width * 4 / 3))
            output_height += output_height % 2
            video_height = min(metadata.height, output_height // 2) // 2 * 2
            video_scale = video_height / metadata.height
            video_width = min(output_width // 2 * 2, int(round(metadata.width * video_scale / 2) * 2))
            extra_height = output_height - video_height
            pose_height = metrics_height = extra_height
            skeleton_width = int(round(output_width * 0.40 / 2) * 2)
            metrics_width = output_width - skeleton_width
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (output_width, output_height),
        )
        resources.callback(writer.release)
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {output_path}")

        knee_color, elbow_color, hip_color = (35, 190, 255), (255, 210, 20), (75, 225, 145)
        upper_arm_color, ball_angle_color = (220, 90, 255), (0, 235, 255)
        pause_frames = {
            phase["frame_index"]
            for phase in shot_metric_phases(metrics, observations.timestamps)
            if phase["frame_index"] is not None
        }
        release_visual_frame = (
            metrics.release_measurement_frame
            if metrics.release_measurement_frame is not None
            else metrics.release_frame
        )
        maximum_jump_visual_frame = (
            metrics.body_peak_frame if metrics.body_peak_frame is not None else metrics.motion_peak_frame
        )
        if metrics.max_jump_height_ratio is None:
            maximum_jump_visual_frame = None
        if maximum_jump_visual_frame is not None and 0 <= maximum_jump_visual_frame < len(observations.timestamps):
            pause_frames.add(maximum_jump_visual_frame)
        foot_lift_events = []
        if release_visual_frame is not None and metrics.release_foot_lift_ratio is not None:
            foot_lift_events.append((release_visual_frame, (0, 165, 255)))
        if maximum_jump_visual_frame is not None and metrics.max_jump_height_ratio is not None:
            foot_lift_events.append((maximum_jump_visual_frame, (220, 100, 255)))
        pause_frame_count = max(round(metadata.fps) - 1, 0)

        render_progress = tqdm(
            total=len(observations.timestamps) + len(pause_frames) * pause_frame_count,
            desc=f"Generating {Path(output_path).name}",
            unit="frames",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        resources.callback(render_progress.close)
        for index in range(len(observations.timestamps)):
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Could not decode frame {index} during rendering")
            if callout_background is not None and (callout_box := red_callout_box(frame)) is not None:
                x0, y0, x1, y1 = callout_box
                frame[y0:y1, x0:x1] = callout_background[y0:y1, x0:x1]

            box = observations.boxes[index]
            if np.isfinite(box).all():
                left = draw_source_pose(
                    frame,
                    tracked.keypoints[index],
                    observations.confidence[index],
                    CONFIG["KEYPOINT_THRESHOLD"],
                )
                panel_keypoints = tracked.visualization_keypoints[index]
                panel_confidence = observations.confidence[index]
            else:
                left = frame.copy()
                panel_keypoints = None
                panel_confidence = None
            rim_box = observations.rim_boxes[index] if observations.rim_boxes is not None else metrics.rim_box
            if np.isfinite(rim_box).all():
                rim_left, rim_top, rim_right, rim_bottom = np.rint(rim_box).astype(int)
                cv2.rectangle(left, (rim_left, rim_top), (rim_right, rim_bottom), (255, 70, 190), 2, cv2.LINE_AA)
            ball_trail = observations.ball_centers_smoothed[: index + 1]
            for first, second in zip(ball_trail[:-1], ball_trail[1:]):
                if np.isfinite(first).all() and np.isfinite(second).all():
                    cv2.line(
                        left,
                        tuple(np.rint(first).astype(int)),
                        tuple(np.rint(second).astype(int)),
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
            ball_box = observations.ball_boxes[index]
            if np.isfinite(ball_box).all():
                x1, y1, x2, y2 = np.rint(ball_box).astype(int)
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 165, 255), 2, cv2.LINE_AA)
                ball_center = tuple(np.rint(observations.ball_centers_smoothed[index]).astype(int))
                cv2.circle(left, ball_center, 4, (0, 220, 255), -1, cv2.LINE_AA)
            panel = render_3d_panel(
                pose_height,
                skeleton_width,
                panel_keypoints,
                panel_confidence,
                CONFIG["KEYPOINT_THRESHOLD"],
                box,
                metrics.grounded_foot_baseline_y_px,
                metrics.body_height_reference_px,
                depth_scale,
                lang=lang,
            )
            if index == metrics.contact_frame and np.isfinite(observations.ball_centers[index]).all():
                contact = tuple(np.rint(observations.ball_centers[index]).astype(int))
                cv2.circle(left, contact, 16, (0, 0, 255), 3, cv2.LINE_AA)
            angle_overlays = []
            side = metrics.shooting_side
            points = tracked.analysis_keypoints[index]
            if side is not None and index == metrics.knee_frame and metrics.knee_angle_deg is not None:
                hip, knee, ankle = (CONFIG["KEYPOINT_INDEX"][f"{side}_{name}"] for name in ("hip", "knee", "ankle"))
                angle_overlays.append((points[hip], points[knee], points[ankle], knee_color))
            if side is not None:
                hip, shoulder, elbow, wrist, hip_knee = (
                    CONFIG["KEYPOINT_INDEX"][f"{side}_{name}"] for name in ("hip", "shoulder", "elbow", "wrist", "knee")
                )
                for phase_frame, phase_elbow, phase_upper_arm, phase_hip in (
                    (
                        metrics.knee_frame,
                        metrics.knee_elbow_angle_deg,
                        metrics.knee_upper_arm_angle_deg,
                        metrics.knee_hip_angle_deg,
                    ),
                    (
                        metrics.takeoff_frame,
                        metrics.takeoff_elbow_angle_deg,
                        metrics.takeoff_upper_arm_angle_deg,
                        metrics.takeoff_hip_angle_deg,
                    ),
                ):
                    if index == phase_frame:
                        if phase_elbow is not None:
                            angle_overlays.append((points[shoulder], points[elbow], points[wrist], elbow_color))
                        if phase_upper_arm is not None:
                            angle_overlays.append((points[hip], points[shoulder], points[elbow], upper_arm_color))
                        if phase_hip is not None:
                            angle_overlays.append((points[shoulder], points[hip], points[hip_knee], hip_color))
            if (
                index == release_visual_frame
                and metrics.release_angle_deg is not None
                and np.isfinite(observations.ball_centers_smoothed[index]).all()
            ):
                center = observations.ball_centers_smoothed[index]
                nearby = observations.ball_centers_smoothed[
                    max(0, index - 2) : min(len(observations.timestamps), index + 4)
                ]
                nearby = nearby[np.isfinite(nearby).all(axis=1)]
                direction = float(np.sign(nearby[-1, 0] - nearby[0, 0])) if len(nearby) > 1 else 1.0
                direction = direction or 1.0
                ray_length = max(45, min(90, left.shape[1] // 12))
                radians = np.deg2rad(metrics.release_angle_deg)
                angle_overlays.append(
                    (
                        center + (direction * ray_length, 0),
                        center,
                        center + (direction * ray_length * np.cos(radians), -ray_length * np.sin(radians)),
                        ball_angle_color,
                    )
                )
            if (
                index == metrics.contact_frame
                and metrics.contact_angle_deg is not None
                and np.isfinite(observations.ball_centers_smoothed[index]).all()
            ):
                center = observations.ball_centers_smoothed[index]
                nearby = observations.ball_centers_smoothed[max(0, index - 4) : index + 1]
                nearby = nearby[np.isfinite(nearby).all(axis=1)]
                direction = float(np.sign(nearby[-1, 0] - nearby[0, 0])) if len(nearby) > 1 else 1.0
                direction = direction or 1.0
                ray_length = max(45, min(90, left.shape[1] // 12))
                radians = np.deg2rad(metrics.contact_angle_deg)
                angle_overlays.append(
                    (
                        center + (direction * ray_length, 0),
                        center,
                        center + (direction * ray_length * np.cos(radians), ray_length * np.sin(radians)),
                        ball_angle_color,
                    )
                )
            draw_angle_overlays(left, angle_overlays)
            current_foot_events = [event for event in foot_lift_events if event[0] == index]
            jump_foot_indices = CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"].get(metrics.shooting_side)
            if (
                current_foot_events
                and jump_foot_indices is not None
                and metrics.grounded_foot_baseline_y_px is not None
                and np.isfinite(metrics.grounded_foot_baseline_y_px)
            ):
                foot_points = tracked.analysis_keypoints[index, jump_foot_indices]
                visible_feet = (
                    tracked.valid[index, jump_foot_indices]
                    & np.isfinite(foot_points[:, :2]).all(axis=1)
                    & (observations.confidence[index, jump_foot_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
                )
                if np.any(visible_feet):
                    visible_points = foot_points[visible_feet]
                    lowest_foot_index = int(np.argmax(visible_points[:, 1]))
                    foot_x, foot_y = np.rint(visible_points[lowest_foot_index, :2]).astype(int)
                    transform = observations.camera_transform_to_reference_xy
                    if metrics.video_profile.camera_compensation_enabled and transform is not None:
                        reference_foot = apply_affine_transform(
                            np.asarray((foot_x, foot_y)),
                            transform[index],
                        )
                        reference_to_frame = np.linalg.inv(
                            np.vstack((transform[index], (0.0, 0.0, 1.0)))
                        )[:2]
                        baseline_point = apply_affine_transform(
                            np.asarray((reference_foot[0], metrics.grounded_foot_baseline_y_px)),
                            reference_to_frame,
                        )
                        baseline_y = int(round(float(baseline_point[1])))
                    else:
                        camera_y = (
                            float(observations.camera_shift_xy[index, 1])
                            if metrics.video_profile.camera_compensation_enabled
                            else 0.0
                        )
                        baseline_y = int(round(metrics.grounded_foot_baseline_y_px + camera_y))
                    draw_foot_height_markers(
                        left,
                        (int(foot_x), int(foot_y)),
                        baseline_y,
                        box,
                        [color for _, color in current_foot_events],
                    )

            metric_panel = render_metrics_panel(
                metrics_height,
                metrics_width,
                index,
                observations.timestamps,
                metrics,
                scale_limit=2.4,
                annotation=annotation,
                lang=lang,
            )
            if portrait_layout:
                if left.shape[0] < output_height:
                    left = cv2.copyMakeBorder(
                        left, 0, output_height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(7, 14, 21)
                    )
                output_frame = np.concatenate([left, np.concatenate([panel, metric_panel], axis=0)], axis=1)
                cv2.line(output_frame, (metadata.width, 0), (metadata.width, output_height - 1), (64, 70, 78), 2)
                cv2.line(
                    output_frame,
                    (metadata.width, pose_height),
                    (output_width - 1, pose_height),
                    (64, 70, 78),
                    2,
                )
            else:
                resized = cv2.resize(left, (video_width, video_height), interpolation=cv2.INTER_AREA)
                video_panel = np.full((video_height, output_width, 3), (7, 14, 21), dtype=np.uint8)
                video_left = (output_width - video_width) // 2
                video_panel[:, video_left : video_left + video_width] = resized
                output_frame = np.concatenate([video_panel, np.concatenate([panel, metric_panel], axis=1)], axis=0)
                cv2.line(output_frame, (0, video_height), (output_width - 1, video_height), (64, 70, 78), 2)
                cv2.line(
                    output_frame,
                    (skeleton_width, video_height),
                    (skeleton_width, output_height - 1),
                    (64, 70, 78),
                    2,
                )
            for paused_frame in frames_with_pause(
                output_frame,
                index,
                pause_frames,
                pause_frame_count,
            ):
                writer.write(paused_frame)
                render_progress.update(1)


def red_callout_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    top = frame[: round(height * 0.22)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 120)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    minimum_area = max(3000, int(height * width * 0.003))
    boxes = [
        (int(x), int(y), int(x + box_width), int(y + box_height))
        for x, y, box_width, box_height, area in stats[1:]
        if area >= minimum_area
        and box_width >= width * 0.08
        and box_height >= height * 0.02
        and box_width >= box_height * 1.5
        and y < height * 0.2
        and x < width * 0.9
    ]
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), default=None)


def red_callout_background(video_path: Path, metadata: VideoMetadata) -> np.ndarray | None:
    sample_count = min(max(metadata.frame_count, 1), 32)
    sample_indices = set(np.linspace(0, max(metadata.frame_count - 1, 0), sample_count, dtype=int))
    capture = cv2.VideoCapture(str(video_path))
    samples, boxes = [], []
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in sample_indices:
                samples.append(frame[: round(metadata.height * 0.22)].copy())
                boxes.append(red_callout_box(frame))
            frame_index += 1
    finally:
        capture.release()
    if sum(box is not None for box in boxes) < max(3, len(samples) // 4):
        return None

    height, width = samples[0].shape[:2]
    unoccluded = np.ones((len(samples), height, width), dtype=bool)
    for index, box in enumerate(boxes):
        if box is not None:
            x0, y0, x1, y1 = box
            unoccluded[index, max(0, y0 - 2) : min(height, y1 + 2), max(0, x0 - 2) : min(width, x1 + 2)] = False
    missing = ~np.any(unoccluded, axis=0)
    background = np.empty_like(samples[0])
    for channel in range(3):
        pixels = np.stack([frame[:, :, channel] for frame in samples]).astype(np.float32)
        pixels[~unoccluded] = np.nan
        if np.any(missing):
            pixels[:, missing] = samples[0][:, :, channel][missing]
        background[:, :, channel] = np.nanmedian(pixels, axis=0).astype(np.uint8)
    return background
