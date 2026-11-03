"""Track detected objects and estimate camera motion."""

from collections.abc import Callable

import numpy as np

from core.config import CONFIG
from core.geometry import box_iou
from tracking.ball import track_ball_candidates


def _player_observation(
    box: np.ndarray,
    score: float,
    ball_boxes: np.ndarray,
    ball_scores: np.ndarray,
    holder_boxes: np.ndarray,
    holder_scores: np.ndarray,
    image_area: float,
) -> dict[str, object]:
    """Build the per-frame evidence shared by new and continuing player tracks."""
    height = max(float(box[3] - box[1]), 1.0)
    upper_body = np.asarray((box[0], box[1], box[2], box[1] + 0.75 * height), dtype=np.float32)
    ball_support = 0.0
    for ball_box, ball_score in zip(ball_boxes, ball_scores):
        ball_center = (ball_box[:2] + ball_box[2:]) * 0.5
        closest = np.clip(ball_center, upper_body[:2], upper_body[2:])
        distance = float(np.linalg.norm(ball_center - closest) / height)
        if distance <= 0.65:
            ball_support = max(ball_support, float(ball_score) * np.exp(-distance / 0.18))

    holder_support = 0.0
    for holder_box, holder_score in zip(holder_boxes, holder_scores):
        overlap = box_iou(box, holder_box)
        intersection_left = max(float(box[0]), float(holder_box[0]))
        intersection_top = max(float(box[1]), float(holder_box[1]))
        intersection_right = min(float(box[2]), float(holder_box[2]))
        intersection_bottom = min(float(box[3]), float(holder_box[3]))
        intersection = max(0.0, intersection_right - intersection_left) * max(
            0.0, intersection_bottom - intersection_top
        )
        holder_area = max(float((holder_box[2] - holder_box[0]) * (holder_box[3] - holder_box[1])), 1.0)
        containment = intersection / holder_area
        support = max(overlap, containment) if max(overlap, containment) >= 0.2 else 0.0
        holder_support = max(holder_support, float(holder_score) * support)

    return {
        "box": box,
        "score": float(score),
        "ball_support": ball_support,
        "holder_support": holder_support,
        "area_ratio": max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0) / image_area,
    }


def _interpolate_short_box_gaps(boxes: np.ndarray, scores: np.ndarray) -> None:
    """Interpolate one- to three-frame gaps in a tracked box sequence."""
    detected = np.flatnonzero(np.isfinite(boxes).all(axis=1))
    for left, right in zip(detected[:-1], detected[1:]):
        gap = int(right - left)
        if 1 < gap <= 3:
            for frame in range(left + 1, right):
                fraction = (frame - left) / gap
                boxes[frame] = boxes[left] * (1.0 - fraction) + boxes[right] * fraction
                scores[frame] = min(scores[left], scores[right])


def choose_target_track(
    players_by_frame: list[np.ndarray],
    scores_by_frame: list[np.ndarray],
    balls_by_frame: list[np.ndarray],
    ball_scores_by_frame: list[np.ndarray],
    holders_by_frame: list[np.ndarray],
    holder_scores_by_frame: list[np.ndarray],
    image_width: int,
    image_height: int,
    timestamps: np.ndarray,
    pose_scorer: Callable[[np.ndarray], float] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    """Select a shooter by linking pre-release possession to the same ball's flight."""
    frame_count = len(players_by_frame)
    tracks: list[dict[str, object]] = []
    ball_tracks = track_ball_candidates(balls_by_frame, ball_scores_by_frame, timestamps)
    image_area = max(float(image_width * image_height), 1.0)

    for frame_index, (boxes, scores) in enumerate(zip(players_by_frame, scores_by_frame)):
        valid_candidates = np.flatnonzero(scores >= CONFIG["DETECTION_THRESHOLD"])
        matches = []
        for track_index, track in enumerate(tracks):
            previous_frame = int(track["last_frame"])
            frame_gap = frame_index - previous_frame
            if frame_gap < 1 or frame_gap > 2:
                continue
            previous_box = np.asarray(track["last_box"], dtype=np.float32)
            previous_center = (previous_box[:2] + previous_box[2:]) * 0.5
            history = track["centers"]
            if len(history) >= 2:
                first_frame, first_center = history[-2]
                seconds = previous_frame - first_frame
                if seconds > 0:
                    previous_center = previous_center + (previous_center - first_center) / seconds * frame_gap
            previous_diag = max(float(np.linalg.norm(previous_box[2:] - previous_box[:2])), 1.0)
            previous_area = max(float((previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])), 1.0)
            for candidate_index in valid_candidates:
                candidate = boxes[candidate_index]
                center = (candidate[:2] + candidate[2:]) * 0.5
                distance = float(np.linalg.norm(center - previous_center) / previous_diag)
                overlap = box_iou(previous_box, candidate)
                area = max(float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])), 1.0)
                area_ratio = area / previous_area
                if (overlap < 0.01 and distance > 0.9) or not 0.35 <= area_ratio <= 2.8:
                    continue
                cost = 0.55 * distance + 0.45 * (1.0 - overlap) + 0.12 * abs(np.log(area_ratio))
                matches.append((float(cost), track_index, int(candidate_index)))

        assigned_tracks = set()
        assigned_candidates = set()
        for _, track_index, candidate_index in sorted(matches):
            if track_index in assigned_tracks or candidate_index in assigned_candidates:
                continue
            assigned_tracks.add(track_index)
            assigned_candidates.add(candidate_index)
            box = boxes[candidate_index].copy()
            center = (box[:2] + box[2:]) * 0.5
            track = tracks[track_index]
            track["samples"][frame_index] = _player_observation(
                box,
                scores[candidate_index],
                balls_by_frame[frame_index],
                ball_scores_by_frame[frame_index],
                holders_by_frame[frame_index],
                holder_scores_by_frame[frame_index],
                image_area,
            )
            track["last_frame"] = frame_index
            track["last_box"] = box
            track["centers"].append((frame_index, center))
            if len(track["centers"]) > 2:
                track["centers"].pop(0)

        for candidate_index in valid_candidates:
            if int(candidate_index) in assigned_candidates:
                continue
            box = boxes[candidate_index].copy()
            center = (box[:2] + box[2:]) * 0.5
            sample = _player_observation(
                box,
                scores[candidate_index],
                balls_by_frame[frame_index],
                ball_scores_by_frame[frame_index],
                holders_by_frame[frame_index],
                holder_scores_by_frame[frame_index],
                image_area,
            )
            tracks.append(
                {
                    "samples": {frame_index: sample},
                    "last_frame": frame_index,
                    "last_box": box,
                    "centers": [(frame_index, center)],
                }
            )

    scored_tracks = []
    for track in tracks:
        samples = list(track["samples"].values())
        if len(samples) < min(3, frame_count):
            continue
        holder_support = float(np.mean([sample["holder_support"] for sample in samples]))
        ball_support = float(np.mean([sample["ball_support"] for sample in samples]))
        prominence = float(np.sqrt(np.mean([sample["area_ratio"] for sample in samples])))
        detector_quality = float(np.mean([sample["score"] for sample in samples]))
        coverage = len(samples) / max(frame_count, 1)
        track_score = (
            2.4 * holder_support + 1.8 * ball_support + 1.4 * prominence + 0.35 * detector_quality + 0.25 * coverage
        )

        shot_support = 0.0
        shot_ball_track = None
        shot_frame = None
        player_samples = track["samples"]
        for ball_track in ball_tracks:
            ball_samples = ball_track["samples"]
            ball_frames = sorted(ball_samples)
            for pre_position, pre_frame in enumerate(ball_frames):
                if pre_frame not in player_samples:
                    continue
                pre_sample = ball_samples[pre_frame]
                pre_box = player_samples[pre_frame]["box"]
                person_height = max(float(pre_box[3] - pre_box[1]), 1.0)
                upper_body = np.asarray(
                    (pre_box[0], pre_box[1], pre_box[2], pre_box[1] + 0.75 * person_height),
                    dtype=np.float32,
                )
                pre_center = pre_sample["center"]
                pre_outside = np.maximum(upper_body[:2] - pre_center, 0.0) + np.maximum(
                    pre_center - upper_body[2:], 0.0
                )
                pre_distance = float(np.linalg.norm(pre_outside) / person_height)
                if pre_distance > 0.65:
                    continue

                future_frames = []
                for frame in ball_frames[pre_position + 1 :]:
                    elapsed = float(timestamps[frame] - timestamps[pre_frame])
                    if elapsed > 0.4:
                        break
                    if elapsed >= 0.05 and frame in player_samples:
                        future_frames.append(frame)
                if len(future_frames) < 2:
                    continue

                for end_position in range(1, len(future_frames)):
                    end_frame = future_frames[end_position]
                    elapsed = float(timestamps[end_frame] - timestamps[pre_frame])
                    if elapsed < 0.08:
                        continue
                    flight_frames = future_frames[max(0, end_position - 2) : end_position + 1]
                    after_center = np.median(
                        np.asarray([ball_samples[frame]["center"] for frame in flight_frames]), axis=0
                    )
                    after_distances = []
                    flight_scores = []
                    for frame in flight_frames:
                        frame_box = player_samples[frame]["box"]
                        frame_height = max(float(frame_box[3] - frame_box[1]), 1.0)
                        frame_upper = np.asarray(
                            (frame_box[0], frame_box[1], frame_box[2], frame_box[1] + 0.75 * frame_height),
                            dtype=np.float32,
                        )
                        frame_center = ball_samples[frame]["center"]
                        outside = np.maximum(frame_upper[:2] - frame_center, 0.0) + np.maximum(
                            frame_center - frame_upper[2:], 0.0
                        )
                        after_distances.append(float(np.linalg.norm(outside) / frame_height))
                        flight_scores.append(float(ball_samples[frame]["score"]))
                    rise_ratio = float((pre_center[1] - after_center[1]) / person_height)
                    separation_ratio = float(np.median(after_distances) - pre_distance)
                    if rise_ratio < 0.08 or separation_ratio < 0.08:
                        continue
                    proximity = float(np.clip(1.0 - pre_distance / 0.65, 0.0, 1.0))
                    upward = float(np.clip(rise_ratio / 0.30, 0.0, 1.0))
                    separation = float(np.clip(separation_ratio / 0.45, 0.0, 1.0))
                    continuity = min(len(flight_frames) / 3.0, 1.0)
                    confidence = min(float(pre_sample["score"]), float(np.median(flight_scores)))
                    support = proximity * np.sqrt(upward * separation) * continuity * confidence
                    if support > shot_support:
                        shot_support = support
                        shot_ball_track = ball_track
                        shot_frame = pre_frame

        scored_tracks.append(
            (
                track_score,
                track,
                holder_support,
                ball_support,
                coverage,
                prominence,
                detector_quality,
                shot_support,
                shot_ball_track,
                shot_frame,
            )
        )

    target_boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    target_scores = np.full(frame_count, np.nan, dtype=np.float32)
    associated_ball_boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    associated_ball_scores = np.full(frame_count, np.nan, dtype=np.float32)
    diagnostics = {
        "track_score": 0.0,
        "holder_support": 0.0,
        "ball_support": 0.0,
        "coverage": 0.0,
        "prominence": 0.0,
        "detector_quality": 0.0,
        "candidate_count": 0.0,
        "runner_up_score": 0.0,
        "score_margin_ratio": 0.0,
        "release_event_supported": 0.0,
        "release_event_support": 0.0,
        "release_event_frame_1based": 0.0,
        "release_ball_track_frames": 0.0,
        "release_override_possession_gate": 0.0,
        "release_possession_score": 0.0,
        "static_possession_score": 0.0,
        "release_possession_ratio": 0.0,
        "selected_ball_track_locked": 0.0,
    }
    if not scored_tracks:
        return target_boxes, target_scores, diagnostics, associated_ball_boxes, associated_ball_scores

    def selection_score(item) -> float:
        track_score, _, holder_support, ball_support, _, _, _, shot_support, _, _ = item
        static_possession_support = 2.4 * holder_support + 1.8 * ball_support
        release_support = 4.2 * shot_support
        non_possession_score = track_score - static_possession_support
        return non_possession_score + max(static_possession_support, release_support)

    release_ranked = sorted(scored_tracks, key=selection_score, reverse=True)
    static_ranked = sorted(scored_tracks, key=lambda item: item[0], reverse=True)
    release_margin = (
        (selection_score(release_ranked[0]) - selection_score(release_ranked[1]))
        / max(abs(selection_score(release_ranked[0])), 1e-6)
        if len(release_ranked) > 1
        else 1.0
    )
    release_possession_score = 2.4 * release_ranked[0][2] + 1.8 * release_ranked[0][3]
    static_possession_score = 2.4 * static_ranked[0][2] + 1.8 * static_ranked[0][3]
    release_possession_ratio = release_possession_score / max(static_possession_score, 1e-6)
    release_override_possession_gate = (
        release_possession_score >= CONFIG["RELEASE_OVERRIDE_MIN_POSSESSION_SCORE"]
        and release_possession_ratio >= CONFIG["RELEASE_OVERRIDE_MIN_POSSESSION_RATIO"]
    )
    release_override_applied = bool(
        release_ranked[0][7] > 0.0
        and release_ranked[0] is not static_ranked[0]
        and release_margin >= 0.10
        and release_override_possession_gate
    )
    scored_tracks = release_ranked if release_override_applied else static_ranked
    pose_scores = {}
    pose_override_applied = False
    pose_selection_applied = False
    if pose_scorer is not None and (
        scored_tracks[0][7] == 0.0
        or scored_tracks[0] is not release_ranked[0]
        or release_margin < 0.25
    ):
        # Holding a second ball or occupying more pixels does not identify the
        # shooter. Check arm motion for the leading association candidates.
        for item in release_ranked[: CONFIG["SHOOTER_POSE_CANDIDATES"]]:
            candidate_boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
            for index, sample in item[1]["samples"].items():
                candidate_boxes[index] = sample["box"]
            pose_scores[id(item[1])] = pose_scorer(candidate_boxes)
        ranked = sorted(
            scored_tracks,
            key=lambda item: (pose_scores.get(id(item[1]), 0.0), selection_score(item)),
            reverse=True,
        )
        if pose_scores.get(id(ranked[0][1]), 0.0) >= CONFIG["SHOOTER_POSE_MIN_SCORE"]:
            pose_selection_applied = True
            pose_override_applied = ranked[0] is not scored_tracks[0]
            scored_tracks = ranked
    (
        track_score,
        selected_track,
        holder_support,
        ball_support,
        coverage,
        prominence,
        detector_quality,
        release_event_support,
        selected_ball_track,
        release_event_frame,
    ) = scored_tracks[0]
    selected_score = selection_score(scored_tracks[0]) if release_override_applied else float(scored_tracks[0][0])
    runner_up_score = (
        (selection_score(scored_tracks[1]) if release_override_applied else float(scored_tracks[1][0]))
        if len(scored_tracks) > 1
        else 0.0
    )
    if pose_selection_applied:
        selected_score = pose_scores[id(selected_track)]
        runner_up_score = max((pose_scores.get(id(item[1]), 0.0) for item in scored_tracks[1:]), default=0.0)
    score_margin_ratio = max(selected_score - runner_up_score, 0.0) / max(abs(selected_score), 1e-6)
    for frame_index, sample in selected_track["samples"].items():
        target_boxes[frame_index] = sample["box"]
        target_scores[frame_index] = sample["score"]
    _interpolate_short_box_gaps(target_boxes, target_scores)

    if selected_ball_track is not None:
        for frame_index, sample in selected_ball_track["samples"].items():
            associated_ball_boxes[frame_index] = sample["box"]
            associated_ball_scores[frame_index] = sample["score"]
        _interpolate_short_box_gaps(associated_ball_boxes, associated_ball_scores)

    diagnostics = {
        "track_score": float(track_score),
        "holder_support": holder_support,
        "ball_support": ball_support,
        "coverage": coverage,
        "prominence": prominence,
        "detector_quality": detector_quality,
        "candidate_count": float(len(scored_tracks)),
        "runner_up_score": runner_up_score,
        "score_margin_ratio": float(score_margin_ratio),
        "selection_score": float(selected_score),
        "release_override_applied": float(release_override_applied),
        "release_event_supported": float(release_event_support > 0.0 and selected_ball_track is not None),
        "release_event_support": float(release_event_support),
        "release_event_frame_1based": float(release_event_frame + 1) if release_event_frame is not None else 0.0,
        "release_ball_track_frames": float(len(selected_ball_track["samples"]))
        if selected_ball_track is not None
        else 0.0,
        "release_override_possession_gate": float(release_override_possession_gate),
        "release_possession_score": float(release_possession_score),
        "static_possession_score": float(static_possession_score),
        "release_possession_ratio": float(release_possession_ratio),
        "selected_ball_track_locked": float(selected_ball_track is not None),
    }
    if pose_scorer is not None:
        diagnostics["pose_shot_score"] = float(pose_scores.get(id(selected_track), 0.0))
        diagnostics["pose_override_applied"] = float(pose_override_applied)
        diagnostics["pose_selection_applied"] = float(pose_selection_applied)
    return (
        target_boxes,
        target_scores,
        diagnostics,
        associated_ball_boxes,
        associated_ball_scores,
    )
