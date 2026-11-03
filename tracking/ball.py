"""Associate the shooting ball across hand contact and flight."""

import numpy as np

from core.config import CONFIG


def recover_rim_approach(
    boxes_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    timestamps: np.ndarray,
    rim_boxes: np.ndarray,
    ball_boxes: np.ndarray,
    ball_scores: np.ndarray,
    release_frame: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a descending detection fragment after the flight apex is lost.

    A fragment must travel toward the observed rim, have several moving ball
    detections, and agree with the launch direction. Static net false positives
    cannot establish an approach. No positions are synthesized across the gap.
    """
    if release_frame is None or not np.isfinite(ball_boxes[release_frame]).all():
        return ball_boxes, ball_scores
    launch_center = (ball_boxes[release_frame, :2] + ball_boxes[release_frame, 2:]) * 0.5
    launch_size = float(np.mean(ball_boxes[release_frame, 2:] - ball_boxes[release_frame, :2]))
    launch_rim = rim_boxes[release_frame]
    if not np.isfinite(launch_rim).all() or launch_size <= 0:
        return ball_boxes, ball_scores
    launch_dx = float(np.mean(launch_rim[[0, 2]]) - launch_center[0])
    approaches = []
    for track in track_ball_candidates(boxes_by_frame, scores_by_frame, timestamps):
        samples = track["samples"]
        for frame, sample in sorted(samples.items()):
            elapsed = timestamps[frame] - timestamps[release_frame]
            if elapsed <= 0 or elapsed > CONFIG["RIM_CONTACT_WINDOW_SECONDS"]:
                continue
            rim = rim_boxes[frame]
            if not np.isfinite(rim).all():
                continue
            box, center = sample["box"], sample["center"]
            diameter = float(np.mean(box[2:] - box[:2]))
            distance = float(np.linalg.norm(center - np.clip(center, rim[:2], rim[2:])))
            if diameter <= 0 or distance > diameter * 0.5 or not 0.2 <= diameter / launch_size <= 5.0:
                continue
            prior = sorted(
                index for index in samples
                if release_frame < index <= frame
                and timestamps[frame] - timestamps[index] <= CONFIG["BALL_FLIGHT_FIT_WINDOW_SECONDS"]
                and np.isfinite(rim_boxes[index]).all()
            )
            if len(prior) < 3:
                continue
            positions = np.asarray([
                samples[index]["center"] - (rim_boxes[index, :2] + rim_boxes[index, 2:]) * 0.5
                for index in prior
            ])
            sizes = np.asarray([samples[index]["box"][2:] - samples[index]["box"][:2] for index in prior])
            aspect = np.median(np.min(sizes, axis=1) / np.maximum(np.max(sizes, axis=1), 1.0))
            times = timestamps[prior] - timestamps[frame]
            if np.ptp(times) <= 0:
                continue
            velocity = np.linalg.lstsq(np.column_stack((np.ones(len(times)), times)), positions, rcond=None)[0][1]
            travel = float(np.linalg.norm(positions[-1] - positions[0]))
            if (
                travel <= diameter * 2.0 or velocity[1] <= 0
                or positions[0, 1] >= 0 or aspect < CONFIG["RELEASE_ANGLE_MIN_BOX_ASPECT_RATIO"]
                or (abs(launch_dx) > launch_size * 2.0 and launch_dx * velocity[0] <= 0)
            ):
                continue
            approaches.append((frame, track))
            break
    if not approaches:
        return ball_boxes, ball_scores
    contact, selected = min(approaches, key=lambda item: item[0])
    boxes, scores = ball_boxes.copy(), ball_scores.copy()
    for frame, sample in selected["samples"].items():
        if release_frame < frame <= contact + 2:
            boxes[frame], scores[frame] = sample["box"], sample["score"]
    return boxes, scores


def extend_ball_flight(
    boxes_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    timestamps: np.ndarray,
    camera_shift_xy: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    player_boxes: np.ndarray,
    ball_boxes: np.ndarray,
    ball_scores: np.ndarray,
    release_frame: int | None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Reconnect observed flight fragments after a brief disappearance.

    Predictions only gate real detections. No predicted coordinates are emitted
    for unseen ball frames, including the apex outside the image. Return the
    flight start as well so later association can use the same pose fallback.
    """
    boxes = np.asarray(ball_boxes, dtype=np.float32).copy()
    scores = np.asarray(ball_scores, dtype=np.float32).copy()
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    if release_frame is None:
        for frame in np.flatnonzero(np.isfinite(centers).all(axis=1)):
            height = player_boxes[frame, 3] - player_boxes[frame, 1]
            if not np.isfinite(height) or height <= 0:
                continue
            for side in CONFIG["SIDE_NAMES"]:
                ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
                arm = points[frame, ids, :2]
                if not np.isfinite(arm).all() or not np.all(confidence[frame, ids] >= CONFIG["KEYPOINT_THRESHOLD"]):
                    continue
                first, last = arm[0] - arm[1], arm[2] - arm[1]
                length = float(np.linalg.norm(first) * np.linalg.norm(last))
                if length <= 0:
                    continue
                angle = float(np.degrees(np.arccos(np.clip(np.dot(first, last) / length, -1.0, 1.0))))
                if (
                    angle >= 140
                    and arm[0, 1] - arm[2, 1] >= 0.1 * height
                    and np.linalg.norm(centers[frame] - arm[2]) <= 0.2 * height
                ):
                    release_frame = int(frame)
                    break
            if release_frame is not None:
                break
    if release_frame is None:
        return boxes, scores, None
    history = [
        int(frame) for frame in np.flatnonzero(np.isfinite(centers).all(axis=1))
        if release_frame <= frame <= release_frame + 2
    ]
    if len(history) < 2:
        return boxes, scores, release_frame
    start = history[-1] + 1
    for frame in range(start, len(timestamps)):
        elapsed = float(timestamps[frame] - timestamps[history[-1]])
        if elapsed > CONFIG["BALL_FLIGHT_MAX_GAP_SECONDS"]:
            break
        recent = [
            index for index in history
            if timestamps[history[-1]] - timestamps[index] <= CONFIG["BALL_FLIGHT_FIT_WINDOW_SECONDS"]
        ]
        if len(recent) < 3:
            recent = history[-3:]
        times = timestamps[recent] - timestamps[history[-1]]
        positions = centers[recent] - camera_shift_xy[recent]
        dt = float(timestamps[frame] - timestamps[history[-1]])
        linear = np.column_stack((np.ones(len(times)), times))
        x_fit = np.linalg.lstsq(linear, positions[:, 0], rcond=None)[0]
        degree = 2 if len(times) >= 4 else 1
        design = np.column_stack([times ** power for power in range(degree + 1)])
        y_fit = np.linalg.lstsq(design, positions[:, 1], rcond=None)[0]
        predicted = np.array([
            x_fit[0] + x_fit[1] * dt,
            sum(coefficient * dt ** power for power, coefficient in enumerate(y_fit)),
        ])
        last_box = boxes[history[-1]]
        diameter = max(float(np.max(last_box[2:] - last_box[:2])), 1.0)
        speed = float(np.linalg.norm(positions[-1] - positions[0]) / max(float(np.ptp(times)), 1e-3))
        height = player_boxes[history[-1], 3] - player_boxes[history[-1], 1]
        scale = float(height) if np.isfinite(height) and height > 0 else diameter * 5.0
        gate = max(diameter * 3, scale * 0.15) + speed * dt * 0.5
        candidates = boxes_by_frame[frame]
        selected = None
        if len(candidates):
            observed = (candidates[:, :2] + candidates[:, 2:]) * 0.5 - camera_shift_xy[frame]
            residual = np.linalg.norm(observed - predicted, axis=1)
            eligible = np.flatnonzero(
                (residual <= gate) & (scores_by_frame[frame] >= CONFIG["BALL_DETECTION_THRESHOLD"])
            )
            if len(eligible):
                selected = int(eligible[np.argmin(residual[eligible] / gate - 0.1 * scores_by_frame[frame][eligible])])
        boxes[frame] = np.nan
        scores[frame] = np.nan
        if selected is not None:
            boxes[frame] = candidates[selected]
            scores[frame] = scores_by_frame[frame][selected]
            centers[frame] = (boxes[frame, :2] + boxes[frame, 2:]) * 0.5
            history.append(frame)
    return boxes, scores, release_frame


def track_ball_candidates(
    boxes_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    timestamps: np.ndarray,
) -> list[dict[str, object]]:
    """Keep short, independent trajectories for every detected ball."""
    tracks: list[dict[str, object]] = []
    for frame_index, (boxes, scores) in enumerate(zip(boxes_by_frame, scores_by_frame)):
        matches = []
        for track_index, track in enumerate(tracks):
            samples = track["samples"]
            sample_frames = sorted(samples)
            last_frame = sample_frames[-1]
            frame_gap = frame_index - last_frame
            if frame_gap < 1 or frame_gap > 3:
                continue
            last_sample = samples[last_frame]
            last_center = last_sample["center"]
            last_box = last_sample["box"]
            elapsed = float(timestamps[frame_index] - timestamps[last_frame])
            if elapsed <= 0.0:
                continue
            predicted_center = last_center
            if len(sample_frames) >= 2:
                previous_frame = sample_frames[-2]
                previous_sample = samples[previous_frame]
                previous_elapsed = float(timestamps[last_frame] - timestamps[previous_frame])
                if previous_elapsed > 0.0:
                    velocity = (last_center - previous_sample["center"]) / previous_elapsed
                    predicted_center = last_center + velocity * elapsed
            last_size = np.maximum(last_box[2:] - last_box[:2], 1.0)
            last_diameter = max(float(np.linalg.norm(last_size)), 1.0)
            speed = float(np.linalg.norm(predicted_center - last_center) / elapsed)
            gate = max(last_diameter * 1.8, speed * elapsed * 1.6 + last_diameter)
            last_area = max(float(last_size[0] * last_size[1]), 1.0)
            for candidate_index, box in enumerate(boxes):
                center = (box[:2] + box[2:]) * 0.5
                distance = float(np.linalg.norm(center - predicted_center))
                size = np.maximum(box[2:] - box[:2], 1.0)
                area = max(float(size[0] * size[1]), 1.0)
                area_ratio = area / last_area
                if distance > gate or not 0.2 <= area_ratio <= 5.0:
                    continue
                cost = distance / gate + 0.12 * abs(float(np.log(area_ratio))) - 0.04 * float(scores[candidate_index])
                matches.append((cost, track_index, candidate_index))

        assigned_tracks = set()
        assigned_candidates = set()
        for _, track_index, candidate_index in sorted(matches):
            if track_index in assigned_tracks or candidate_index in assigned_candidates:
                continue
            box = boxes[candidate_index].copy()
            center = (box[:2] + box[2:]) * 0.5
            tracks[track_index]["samples"][frame_index] = {
                "box": box,
                "center": center,
                "score": float(scores[candidate_index]),
            }
            assigned_tracks.add(track_index)
            assigned_candidates.add(candidate_index)

        for candidate_index, box in enumerate(boxes):
            if candidate_index in assigned_candidates:
                continue
            copied_box = box.copy()
            tracks.append(
                {
                    "samples": {
                        frame_index: {
                            "box": copied_box,
                            "center": (copied_box[:2] + copied_box[2:]) * 0.5,
                            "score": float(scores[candidate_index]),
                        }
                    }
                }
            )
    return [track for track in tracks if len(track["samples"]) >= 3]


def select_pose_ball_track(
    boxes_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    points: np.ndarray,
    confidence: np.ndarray,
    player_boxes: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Disambiguate multiple balls using contact with the selected player's hands."""
    hand_ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_wrist"] for side in CONFIG["SIDE_NAMES"]]
    candidates = []
    for track in track_ball_candidates(boxes_by_frame, scores_by_frame, timestamps):
        support = []
        for frame, sample in track["samples"].items():
            height = player_boxes[frame, 3] - player_boxes[frame, 1]
            visible = np.isfinite(points[frame, hand_ids, :2]).all(axis=1) & (
                confidence[frame, hand_ids] >= CONFIG["KEYPOINT_THRESHOLD"]
            )
            if not np.isfinite(height) or height <= 0 or not np.any(visible):
                continue
            distances = np.linalg.norm(points[frame, hand_ids, :2][visible] - sample["center"], axis=1)
            distance = float(np.min(distances) / height)
            if distance <= 0.2:
                support.append((1.0 - distance / 0.2) * sample["score"])
        if len(support) >= 3:
            candidates.append((float(np.mean(sorted(support)[-8:])) * min(len(support) / 6.0, 1.0), track))
    if not candidates:
        return None
    _, selected = max(candidates, key=lambda item: item[0])
    boxes = np.full((len(timestamps), 4), np.nan, dtype=np.float32)
    scores = np.full(len(timestamps), np.nan, dtype=np.float32)
    for frame, sample in selected["samples"].items():
        boxes[frame] = sample["box"]
        scores[frame] = sample["score"]
    return boxes, scores


def fill_ball_contact_gaps(
    boxes_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    points: np.ndarray,
    confidence: np.ndarray,
    player_boxes: np.ndarray,
    ball_boxes: np.ndarray,
    ball_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover observed hand-contact detections omitted by a fragmented ball track."""
    result_boxes = np.asarray(ball_boxes, dtype=np.float32).copy()
    result_scores = np.asarray(ball_scores, dtype=np.float32).copy()
    hand_ids = [CONFIG["KEYPOINT_INDEX"][f"{side}_wrist"] for side in CONFIG["SIDE_NAMES"]]
    hand_ids += list(CONFIG["HAND_KEYPOINT_INDICES"])
    for frame, candidates in enumerate(boxes_by_frame):
        if np.isfinite(result_boxes[frame]).all() or not len(candidates):
            continue
        height = player_boxes[frame, 3] - player_boxes[frame, 1]
        visible = np.isfinite(points[frame, hand_ids, :2]).all(axis=1) & (
            confidence[frame, hand_ids] >= CONFIG["KEYPOINT_THRESHOLD"]
        )
        if not np.isfinite(height) or height <= 0 or not np.any(visible):
            continue
        centers = (candidates[:, :2] + candidates[:, 2:]) * 0.5
        distances = np.linalg.norm(centers[:, None, :] - points[frame, hand_ids, :2][visible], axis=2)
        distances = np.min(distances, axis=1) / height
        eligible = np.flatnonzero(
            (distances <= 0.15) & (scores_by_frame[frame] >= CONFIG["BALL_DETECTION_THRESHOLD"])
        )
        if len(eligible):
            selected = int(eligible[np.argmin(distances[eligible])])
            result_boxes[frame] = candidates[selected]
            result_scores[frame] = scores_by_frame[frame][selected]
    return result_boxes, result_scores


