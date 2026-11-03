"""Collect detections and pose observations from a video."""

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from tqdm import tqdm

from analysis.events import detect_ball_pre_release, score_shooting_track
from core.config import CONFIG
from core.data import PoseObservations, VideoMetadata
from core.geometry import box_iou, smooth_trajectory
from models.inference import estimate_pose
from models.rtmw3d import RTMW3DModel
from tracking.ball import extend_ball_flight, fill_ball_contact_gaps, recover_rim_approach, select_pose_ball_track
from tracking.objects import choose_target_track
from tracking.scene import estimate_background_camera_motion, track_rim_boxes
from utils.reader import read_video_metadata


class DetectionOutput(Protocol):
    """Detection fields consumed from the object detector."""

    class_id: np.ndarray | None
    confidence: np.ndarray | None
    xyxy: np.ndarray


class ObjectDetector(Protocol):
    """Minimal detector interface required by observation collection."""

    class_names: Sequence[str]

    def predict(
        self,
        image: np.ndarray,
        *,
        threshold: float,
        include_source_image: bool,
    ) -> DetectionOutput:
        """Run detection on one image and return the selected fields."""
        ...


def collect_pose_observations(
    video_path: Path,
    detector: ObjectDetector,
    pose_model: RTMW3DModel,
    device: torch.device,
) -> tuple[VideoMetadata, PoseObservations]:
    """Track the shooter and ball over the clip, then estimate the selected pose sequence."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    progress = None
    try:
        player_class_id = detector.class_names.index("player")
        ball_class_id = detector.class_names.index("ball")
        rim_class_id = detector.class_names.index("rim")
        holder_class_id = detector.class_names.index("holder") if "holder" in detector.class_names else None
        metadata = read_video_metadata(capture)
        # Seeking back after probing is inaccurate for some codecs. Both
        # detection and pose must start from a freshly opened first frame.
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Could not reopen video after probing: {video_path}")
        fps = metadata.fps
        timestamps = []
        players_by_frame = []
        player_scores_by_frame = []
        balls_by_frame = []
        ball_scores_by_frame = []
        rims_by_frame = []
        rim_scores_by_frame = []
        holders_by_frame = []
        holder_scores_by_frame = []
        raw_keypoints = None
        raw_confidence = None
        visualization_keypoints = None
        boxes = []
        detector_scores = []
        ball_boxes = []
        ball_scores = []
        camera_shift_xy = []
        camera_motion_confidence = []
        camera_transform_to_reference_xy = []
        camera_flow_scale = min(1.0, 960.0 / max(metadata.width, metadata.height, 1))
        camera_flow_size = (
            max(1, round(metadata.width * camera_flow_scale)),
            max(1, round(metadata.height * camera_flow_scale)),
        )
        previous_flow_gray = None
        previous_flow_mask = None
        cumulative_camera_shift = np.zeros(2, dtype=np.float32)
        cumulative_camera_transform = np.eye(3, dtype=np.float32)
        previous_timestamp = -1.0
        frame_index = 0
        progress = tqdm(
            total=metadata.frame_count or None,
            desc=f"Processing {video_path.name}",
            unit="frames",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(timestamp) or timestamp < 0 or (frame_index > 0 and timestamp <= previous_timestamp):
                timestamp = frame_index / fps
            previous_timestamp = timestamp

            detections = detector.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                threshold=CONFIG["BALL_DETECTION_THRESHOLD"],
                include_source_image=False,
            )
            if detections.class_id is None or detections.confidence is None:
                candidate_boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty(0, dtype=np.float32)
                ball_candidates = np.empty((0, 4), dtype=np.float32)
                ball_candidate_scores = np.empty(0, dtype=np.float32)
                rim_candidates = np.empty((0, 4), dtype=np.float32)
                rim_candidate_scores = np.empty(0, dtype=np.float32)
                holder_candidates = np.empty((0, 4), dtype=np.float32)
                holder_candidate_scores = np.empty(0, dtype=np.float32)
            else:
                class_ids = np.asarray(detections.class_id)
                all_boxes = np.asarray(detections.xyxy, dtype=np.float32)
                all_scores = np.asarray(detections.confidence, dtype=np.float32)
                player_mask = class_ids == player_class_id
                ball_mask = (class_ids == ball_class_id) & (all_scores >= CONFIG["BALL_DETECTION_THRESHOLD"])
                rim_mask = (class_ids == rim_class_id) & (all_scores >= CONFIG["RIM_DETECTION_THRESHOLD"])
                holder_mask = (
                    (class_ids == holder_class_id) & (all_scores >= CONFIG["DETECTION_THRESHOLD"])
                    if holder_class_id is not None
                    else np.zeros_like(class_ids, dtype=bool)
                )
                ball_candidates, ball_candidate_scores = all_boxes[ball_mask], all_scores[ball_mask]
                rim_candidates, rim_candidate_scores = all_boxes[rim_mask], all_scores[rim_mask]
                holder_candidates, holder_candidate_scores = all_boxes[holder_mask], all_scores[holder_mask]
                person_boxes = np.concatenate((all_boxes[player_mask], holder_candidates), axis=0)
                person_scores = np.concatenate((all_scores[player_mask], holder_candidate_scores), axis=0)
                person_order = np.argsort(-person_scores)
                person_indices = []
                for candidate_index in person_order:
                    if any(
                        box_iou(person_boxes[candidate_index], person_boxes[kept_index]) >= 0.75
                        for kept_index in person_indices
                    ):
                        continue
                    person_indices.append(int(candidate_index))
                candidate_boxes = person_boxes[person_indices]
                scores = person_scores[person_indices]

            flow_gray = cv2.cvtColor(
                cv2.resize(frame, camera_flow_size, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            flow_mask = np.full(flow_gray.shape, 255, dtype=np.uint8)
            motion_boxes = [candidate_boxes, ball_candidates, rim_candidates]
            for frame_boxes in motion_boxes:
                for motion_box in frame_boxes:
                    x0, y0, x1, y1 = np.rint(motion_box * camera_flow_scale).astype(int)
                    margin = 8
                    flow_mask[
                        max(0, y0 - margin) : min(flow_mask.shape[0], y1 + margin),
                        max(0, x0 - margin) : min(flow_mask.shape[1], x1 + margin),
                    ] = 0
            flow_confidence = 0.0
            if previous_flow_gray is not None and previous_flow_mask is not None:
                estimated_shift, estimated_transform, flow_confidence = estimate_background_camera_motion(
                    previous_flow_gray,
                    previous_flow_mask,
                    flow_gray,
                    flow_mask,
                    camera_flow_scale,
                )
                if estimated_shift is not None:
                    cumulative_camera_shift += estimated_shift
                if estimated_transform is not None:
                    cumulative_camera_transform = estimated_transform @ cumulative_camera_transform
                elif estimated_shift is not None:
                    translation = np.eye(3, dtype=np.float32)
                    translation[:2, 2] = estimated_shift
                    cumulative_camera_transform = translation @ cumulative_camera_transform
                else:
                    flow_confidence = 0.0
            camera_shift_xy.append(cumulative_camera_shift.copy())
            camera_motion_confidence.append(flow_confidence)
            camera_transform_to_reference_xy.append(
                np.linalg.inv(cumulative_camera_transform)[:2].copy()
            )
            previous_flow_gray = flow_gray
            previous_flow_mask = flow_mask

            players_by_frame.append(candidate_boxes.copy())
            player_scores_by_frame.append(scores.copy())
            balls_by_frame.append(ball_candidates.copy())
            ball_scores_by_frame.append(ball_candidate_scores.copy())
            rims_by_frame.append(rim_candidates.copy())
            rim_scores_by_frame.append(rim_candidate_scores.copy())
            holders_by_frame.append(holder_candidates.copy())
            holder_scores_by_frame.append(holder_candidate_scores.copy())
            timestamps.append(timestamp)
            frame_index += 1
            progress.update(1)

        (
            target_boxes,
            target_scores,
            target_diagnostics,
            associated_ball_boxes,
            associated_ball_scores,
        ) = choose_target_track(
            players_by_frame,
            player_scores_by_frame,
            balls_by_frame,
            ball_scores_by_frame,
            holders_by_frame,
            holder_scores_by_frame,
            metadata.width,
            metadata.height,
            np.asarray(timestamps, dtype=np.float64),
            pose_scorer=lambda candidate_boxes: score_shooting_track(
                video_path, candidate_boxes, pose_model, device
            ),
        )
        logging.info(
            "目标关联 track_score=%.3f holder_support=%.3f ball_support=%.3f coverage=%.3f "
            "release_support=%.3f release_frame=%d candidates=%d",
            target_diagnostics["track_score"],
            target_diagnostics["holder_support"],
            target_diagnostics["ball_support"],
            target_diagnostics["coverage"],
            target_diagnostics["release_event_support"],
            int(target_diagnostics["release_event_frame_1based"]),
            int(target_diagnostics["candidate_count"]),
        )
        previous_ball_center = None
        previous_ball_velocity = np.zeros(2, dtype=np.float32)
        previous_ball_timestamp = None
        missed_ball_frames = 0
        ball_departed = False
        departure_streak = 0
        associated_ball_frames = np.flatnonzero(np.isfinite(associated_ball_boxes).all(axis=1))
        associated_ball_track_start = (
            int(associated_ball_frames[0])
            if target_diagnostics["selected_ball_track_locked"] and len(associated_ball_frames)
            else None
        )
        for index, (candidate_boxes, candidate_scores, box) in enumerate(
            zip(balls_by_frame, ball_scores_by_frame, target_boxes)
        ):
            timestamp = timestamps[index]
            ball_box = np.full(4, np.nan, dtype=np.float32)
            ball_center = np.full(2, np.nan, dtype=np.float32)
            ball_score = float("nan")
            if associated_ball_track_start is not None and index >= associated_ball_track_start:
                if np.isfinite(associated_ball_boxes[index]).all():
                    ball_box = associated_ball_boxes[index].copy()
                    ball_center = (ball_box[:2] + ball_box[2:]) * 0.5
                    ball_score = float(associated_ball_scores[index])
            elif len(candidate_boxes):
                candidate_centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) * 0.5
                selected_ball = -1
                if previous_ball_center is None or (
                    missed_ball_frames >= max(3, round(fps * CONFIG["RELEASE_FLIGHT_CONFIRM_SECONDS"]))
                    and not ball_departed
                ):
                    if np.isfinite(box).all():
                        box_height = max(float(box[3] - box[1]), 1.0)
                        hand_region = np.asarray(
                            (
                                box[0] - 0.15 * box_height,
                                box[1] - 0.10 * box_height,
                                box[2] + 0.15 * box_height,
                                box[1] + 0.78 * box_height,
                            ),
                            dtype=np.float32,
                        )
                        outside = np.maximum(hand_region[:2] - candidate_centers, 0.0) + np.maximum(
                            candidate_centers - hand_region[2:], 0.0
                        )
                        distances = np.linalg.norm(outside, axis=1) / box_height
                        eligible = np.flatnonzero(distances <= 0.65)
                        if len(eligible):
                            ranking = candidate_scores[eligible] + 0.35 * np.exp(-distances[eligible] / 0.18)
                            selected_ball = int(eligible[np.argmax(ranking)])
                elif previous_ball_timestamp is not None:
                    elapsed = max(timestamp - previous_ball_timestamp, 1.0 / fps)
                    predicted_center = previous_ball_center + previous_ball_velocity * elapsed
                    distances = np.linalg.norm(candidate_centers - predicted_center, axis=1)
                    gate = max(70.0, min(250.0, float(np.linalg.norm(previous_ball_velocity)) * elapsed * 1.6 + 35.0))
                    eligible = np.flatnonzero(distances <= gate)
                    if len(eligible):
                        ranking = candidate_scores[eligible] + 0.35 * np.exp(-distances[eligible] / gate)
                        selected_ball = int(eligible[np.argmax(ranking)])

                if selected_ball >= 0:
                    ball_box = candidate_boxes[selected_ball]
                    ball_center = candidate_centers[selected_ball]
                    ball_score = float(candidate_scores[selected_ball])

            if np.isfinite(ball_center).all():
                if previous_ball_center is not None and previous_ball_timestamp is not None and missed_ball_frames <= 5:
                    elapsed = max(timestamp - previous_ball_timestamp, 1.0 / fps)
                    measured_velocity = (ball_center - previous_ball_center) / elapsed
                    previous_ball_velocity = 0.5 * previous_ball_velocity + 0.5 * measured_velocity
                else:
                    previous_ball_velocity.fill(0.0)
                previous_ball_center = ball_center.copy()
                previous_ball_timestamp = timestamp
                missed_ball_frames = 0
                if np.isfinite(box).all():
                    box_height = max(float(box[3] - box[1]), 1.0)
                    hand_region = np.asarray(
                        (
                            box[0] - 0.15 * box_height,
                            box[1] - 0.10 * box_height,
                            box[2] + 0.15 * box_height,
                            box[1] + 0.78 * box_height,
                        ),
                        dtype=np.float32,
                    )
                    outside = np.maximum(hand_region[:2] - ball_center, 0.0) + np.maximum(
                        ball_center - hand_region[2:], 0.0
                    )
                    away = float(np.linalg.norm(outside) / box_height) > 0.08
                    departure_streak = departure_streak + 1 if away else 0
                    ball_departed = ball_departed or departure_streak >= 2
            else:
                missed_ball_frames += 1
            boxes.append(box.copy() if np.isfinite(box).all() else np.full(4, np.nan, dtype=np.float32))
            detector_scores.append(float(target_scores[index]))
            ball_boxes.append(ball_box)
            ball_scores.append(ball_score)

        rim_reference, rim_boxes = track_rim_boxes(
            rims_by_frame, rim_scores_by_frame, np.asarray(camera_shift_xy, dtype=np.float32)
        )
        rim_box = rim_boxes[0] if len(rim_boxes) else rim_reference

        progress.set_description(f"Processing {video_path.name}")
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Could not reopen video for pose estimation: {video_path}")
        progress.close()
        progress = tqdm(
            total=frame_index,
            desc=f"Processing {video_path.name}",
            unit="frames",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        previous_visual_depth = None
        for index, box in enumerate(boxes):
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Could not decode frame {index} during pose estimation")
            if np.isfinite(box).all():
                pose, confidence, visualization_pose = estimate_pose(
                    pose_model, frame, box, device, previous_visual_depth
                )
                previous_visual_depth = np.where(confidence > 0.0, visualization_pose[:, 2], np.nan)
            else:
                pose = np.full((CONFIG["KEYPOINT_COUNT"], 3), np.nan, dtype=np.float32)
                confidence = np.full(CONFIG["KEYPOINT_COUNT"], np.nan, dtype=np.float32)
                visualization_pose = np.full((CONFIG["KEYPOINT_COUNT"], 3), np.nan, dtype=np.float32)
                previous_visual_depth = None
            if raw_keypoints is None:
                raw_keypoints = []
                raw_confidence = []
                visualization_keypoints = []
            raw_keypoints.append(pose)
            raw_confidence.append(confidence)
            visualization_keypoints.append(visualization_pose)
            progress.update(1)

        raw_keypoints = np.asarray(raw_keypoints, dtype=np.float32).reshape(-1, CONFIG["KEYPOINT_COUNT"], 3)
        raw_confidence = np.asarray(raw_confidence, dtype=np.float32).reshape(-1, CONFIG["KEYPOINT_COUNT"])
        visualization_keypoints = np.asarray(visualization_keypoints, dtype=np.float32).reshape(
            -1, CONFIG["KEYPOINT_COUNT"], 3
        )
        raw_keypoints[raw_confidence <= 0.0] = np.nan
        if not target_diagnostics["selected_ball_track_locked"]:
            pose_ball_track = select_pose_ball_track(
                balls_by_frame, ball_scores_by_frame, raw_keypoints, raw_confidence,
                np.asarray(boxes, dtype=np.float32), np.asarray(timestamps),
            )
            if pose_ball_track is not None:
                pose_ball_boxes, pose_ball_scores = pose_ball_track
                ball_boxes = pose_ball_boxes
                ball_scores = pose_ball_scores
                target_diagnostics["pose_ball_track_applied"] = 1.0
        ball_boxes, ball_scores = fill_ball_contact_gaps(
            balls_by_frame, ball_scores_by_frame, raw_keypoints, raw_confidence,
            np.asarray(boxes, dtype=np.float32), np.asarray(ball_boxes), np.asarray(ball_scores),
        )
        raw_ball_centers = (ball_boxes[:, :2] + ball_boxes[:, 2:]) * 0.5
        ball_pre_release = detect_ball_pre_release(
            raw_ball_centers,
            ball_boxes,
            boxes,
            visualization_keypoints,
            raw_confidence,
            timestamps,
            metadata,
        )
        held = np.flatnonzero(ball_pre_release)
        previous_ball_boxes = ball_boxes
        ball_boxes, ball_scores, flight_start = extend_ball_flight(
            balls_by_frame, ball_scores_by_frame, np.asarray(timestamps),
            np.asarray(camera_shift_xy), raw_keypoints, raw_confidence,
            np.asarray(boxes), ball_boxes, ball_scores,
            int(held[-1]) if len(held) else None,
        )
        ball_boxes, ball_scores = recover_rim_approach(
            balls_by_frame, ball_scores_by_frame, np.asarray(timestamps), rim_boxes,
            ball_boxes, ball_scores, flight_start,
        )
        if not np.array_equal(previous_ball_boxes, ball_boxes, equal_nan=True):
            target_diagnostics["flight_reassociated_frames"] = float(np.count_nonzero(
                np.isfinite(ball_boxes).all(axis=1) & ~np.isfinite(previous_ball_boxes).all(axis=1)
            ))
            raw_ball_centers = (ball_boxes[:, :2] + ball_boxes[:, 2:]) * 0.5
            ball_pre_release = detect_ball_pre_release(
                raw_ball_centers, ball_boxes, boxes, visualization_keypoints,
                raw_confidence, timestamps, metadata,
            )

        smoothed_ball_centers = smooth_trajectory(raw_ball_centers)
        observations = PoseObservations(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            detector_scores=np.asarray(detector_scores, dtype=np.float32),
            ball_boxes=np.asarray(ball_boxes, dtype=np.float32).reshape(-1, 4),
            ball_centers=raw_ball_centers,
            ball_centers_smoothed=smoothed_ball_centers,
            ball_scores=np.asarray(ball_scores, dtype=np.float32),
            ball_pre_release=ball_pre_release,
            rim_box=rim_box,
            raw_keypoints=raw_keypoints,
            confidence=raw_confidence,
            visualization_keypoints=visualization_keypoints,
            target_diagnostics=target_diagnostics,
            camera_shift_xy=np.asarray(camera_shift_xy, dtype=np.float32).reshape(-1, 2),
            camera_motion_confidence=np.asarray(camera_motion_confidence, dtype=np.float32),
            rim_boxes=rim_boxes,
            camera_transform_to_reference_xy=np.asarray(
                camera_transform_to_reference_xy,
                dtype=np.float32,
            ).reshape(-1, 2, 3),
        )
        return replace(metadata, frame_count=frame_index), observations
    finally:
        if progress is not None:
            progress.close()
        capture.release()
