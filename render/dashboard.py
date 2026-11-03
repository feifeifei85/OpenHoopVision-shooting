"""Render shot-analysis dashboards from in-memory analysis results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from core.config import CONFIG
from core.data import PoseObservations, TrackedPoses, VideoMetadata
from core.geometry import apply_affine_transform
from render.records import (
    ShotAnalysisView,
    phase_frame,
    phase_metric,
    prepare_shot_analysis_view,
)
from render.i18n import SUPPORTED_LANGUAGES, translate
from render.metrics import (
    DASHBOARD_ANGLE_COLORS,
    DASHBOARD_BACKGROUND,
    DASHBOARD_BORDER,
    DASHBOARD_MUTED,
    DASHBOARD_PANEL,
    DASHBOARD_TEXT,
    draw_ball_stats_panel,
    draw_card,
    draw_dashed_line,
    draw_joint_angle_chart,
    draw_jump_height_panel,
    draw_release_metrics,
    draw_text,
)
from render.panels import (
    draw_angle_overlays,
    draw_foot_height_markers,
    draw_source_pose,
    load_font,
)
from render.renderer import frames_with_pause

_WIDTH = 1600
_HEIGHT = 1000
_TIMELINE_CONTENT_OFFSET_Y = 20
_RELEASE_JUMP_COLOR_BGR = (0, 165, 255)
_MAXIMUM_JUMP_COLOR_BGR = (220, 100, 255)
_PHASE_COLORS = {
    "preparation": (112, 125, 143),
    "dip": (143, 91, 255),
    "ball_lift": (75, 164, 255),
    "rise": (255, 158, 37),
    "release": (255, 48, 56),
    "follow_through": (86, 107, 127),
}

@dataclass(frozen=True)
class _DashboardLayout:
    left_box: tuple[int, int, int, int]
    video_area: tuple[int, int, int, int]
    right_x: int
    right_width: int
    release_cards_box: tuple[int, int, int, int]
    jump_box: tuple[int, int, int, int]
    ball_height_box: tuple[int, int, int, int]
    chart_box: tuple[int, int, int, int]

@dataclass(frozen=True)
class _DashboardContext:
    view: ShotAnalysisView
    phase_markers: list[tuple[int, str, tuple[int, int, int]]]
    layout: _DashboardLayout
    fonts: dict[str, ImageFont.FreeTypeFont]

def _dashboard_layout(video_metadata: dict[str, Any]) -> _DashboardLayout:
    width = int(video_metadata["width_px"])
    height = int(video_metadata["height_px"])
    aspect_ratio = width / max(height, 1)
    left_panel_right = 924 if aspect_ratio >= 1.45 else 640 if aspect_ratio <= 0.7 else 744
    right_x, right_width = left_panel_right + 16, _WIDTH - 24 - (left_panel_right + 16)
    second_row_top, second_row_bottom = 186, 326
    second_row_gap = 10
    jump_width = (right_width - second_row_gap) // 3
    return _DashboardLayout(
        left_box=(24, 24, left_panel_right, 720),
        video_area=(42, 78, left_panel_right - 18, 660),
        right_x=right_x,
        right_width=right_width,
        release_cards_box=(right_x, 24, right_x + right_width, 174),
        jump_box=(right_x, second_row_top, right_x + jump_width, second_row_bottom),
        ball_height_box=(
            right_x + jump_width + second_row_gap,
            second_row_top,
            right_x + right_width,
            second_row_bottom,
        ),
        chart_box=(right_x, 342, right_x + right_width, 720),
    )

def _dashboard_fonts(compact: bool = False) -> dict[str, ImageFont.FreeTypeFont]:
    return {
        "title": load_font(19 if compact else 22, True),
        "metric_title": load_font(12 if compact else 14, True),
        "panel_title": load_font(16 if compact else 18, True),
        "label": load_font(12 if compact else 14, False),
        "value": load_font(24 if compact else 28, True),
        "detail": load_font(10 if compact else 12, False),
        "small": load_font(13 if compact else 14, False),
        "tiny": load_font(10 if compact else 12, False),
    }

def _draw_frame_chip(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    frame_index: int,
    frames: list[dict[str, Any]],
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
) -> None:
    _, y0, x1, _ = area
    timestamp = float(frames[frame_index].get("timestamp_sec", 0.0))
    label = translate("dashboard.frame_summary", lang).format(
        frame=frame_index + 1,
        total=len(frames),
        seconds=timestamp,
    )
    width = int(draw.textlength(label, font=fonts["small"])) + 20
    left, top = x1 - width - 10, y0 + 10
    draw.rounded_rectangle((left, top, x1 - 10, top + 34), radius=8, fill=DASHBOARD_PANEL, outline=DASHBOARD_BORDER, width=1)
    draw_text(draw, (left + 10, top + 8), label, fonts["small"], DASHBOARD_TEXT)

def _draw_context_badges(
    draw: ImageDraw.ImageDraw,
    layout: _DashboardLayout,
    metrics: dict[str, Any],
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
) -> None:
    x0, _, x1, y1 = layout.left_box
    hand = metrics.get("shooter", {}).get("shooting_hand")
    hand_key = "hand.left" if hand == "left" else "hand.right" if hand == "right" else "hand.unknown"
    orientation = metrics.get("scene_context", {}).get("player_orientation_to_camera", {})
    view_mode = orientation.get("mode", "unknown")
    view_key = f"view.{view_mode}"
    if view_key not in {"view.frontal", "view.profile", "view.oblique", "view.unknown"}:
        view_key = "view.unknown"
    view = translate(view_key, lang)
    badges = [translate(hand_key, lang), translate("dashboard.camera_badge", lang).format(view=view)]
    rim_entry = phase_metric(metrics, "entry_angle_deg")
    rim_label = (
        translate("dashboard.rim_entry", lang).format(angle=f"{rim_entry:.1f}°")
        if rim_entry is not None
        else translate("dashboard.rim_not_observed", lang)
    )
    y = y1 - 43
    left = x0 + 15
    for label in badges:
        width = int(draw.textlength(label, font=fonts["small"])) + 20
        draw.rounded_rectangle(
            (left, y, left + width, y + 28), radius=8, fill=DASHBOARD_PANEL, outline=DASHBOARD_BORDER, width=1
        )
        draw_text(draw, (left + 10, y + 6), label, fonts["small"], DASHBOARD_TEXT)
        left += width + 8
    width = int(draw.textlength(rim_label, font=fonts["small"])) + 20
    rim_left = max(x1 - width - 15, left)
    draw.rounded_rectangle(
        (rim_left, y, rim_left + width, y + 28), radius=8, fill=DASHBOARD_PANEL, outline=DASHBOARD_BORDER, width=1
    )
    draw_text(draw, (rim_left + 10, y + 6), rim_label, fonts["small"], DASHBOARD_MUTED)

def _fit_source_video_frame(
    frame_bgr: np.ndarray,
    area: tuple[int, int, int, int],
) -> tuple[Image.Image, int, int]:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = area
    image.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    return image, x0 + (x1 - x0 - image.width) // 2, y0 + (y1 - y0 - image.height) // 2

def _read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if ok:
        return frame

    # Some variable-frame-rate clips advertise more frames than random seeking
    # can decode near the end. Re-read sequentially before treating the frame as
    # unavailable; sequential decoding can still reach valid frames in those files.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame_index + 1):
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Could not decode source frame {frame_index}")
    return frame

def _draw_pose_annotations(
    frame: np.ndarray,
    record: dict[str, Any],
    side: str | None,
    keypoint_index: dict[str, int],
) -> np.ndarray:
    points_data = record.get("keypoints_3d_filtered")
    confidence_data = record.get("keypoint_confidence")
    if points_data is None or confidence_data is None:
        return frame
    points = np.asarray(points_data, dtype=np.float32)
    confidence = np.asarray(confidence_data, dtype=np.float32)
    if points.ndim != 2 or confidence.ndim != 1 or len(points) != len(confidence):
        return frame

    annotated = draw_source_pose(frame, points, confidence, CONFIG["KEYPOINT_THRESHOLD"])
    if side is None:
        return annotated
    triplets = {
        "knee": (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
        "hip": (f"{side}_shoulder", f"{side}_hip", f"{side}_knee"),
        "shoulder": (f"{side}_hip", f"{side}_shoulder", f"{side}_elbow"),
        "elbow": (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"),
    }
    overlays = [
        tuple(points[keypoint_index[name]] for name in names) + (tuple(reversed(DASHBOARD_ANGLE_COLORS[joint])),)
        for joint, names in triplets.items()
        if all(name in keypoint_index for name in names)
    ]
    draw_angle_overlays(annotated, overlays)
    return annotated

def _draw_jump_height_markers(
    frame_bgr: np.ndarray,
    record: dict[str, Any],
    metrics: dict[str, Any],
    side: str | None,
    frame_index: int,
) -> np.ndarray:
    """Mark release and maximum jump height against the grounded-foot baseline."""
    calculation_details = metrics.get("calculation_details", {})
    event_frames = calculation_details.get("event_frames", {})
    release_frame = phase_frame(metrics, "release", -1)
    if release_frame < 0 and event_frames.get("release_last_contact") is not None:
        release_frame = int(event_frames["release_last_contact"]) - 1
    maximum_frame = calculation_details.get("jump_height", {}).get("maximum_frame")

    events = []
    if release_frame >= 0 and frame_index == int(release_frame):
        if phase_metric(metrics, "release_foot_lift_ratio") is not None:
            events.append(_RELEASE_JUMP_COLOR_BGR)
    if maximum_frame is not None and frame_index == int(maximum_frame) - 1:
        if phase_metric(metrics, "max_jump_height_ratio") is not None:
            events.append(_MAXIMUM_JUMP_COLOR_BGR)
    if not events:
        return frame_bgr

    foot_indices = (
        CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"].get(side)
        if side is not None
        else CONFIG["FOOT_KEYPOINT_INDICES"]
    )
    points_data = record.get("keypoints_3d_filtered")
    confidence_data = record.get("keypoint_confidence")
    valid_data = record.get("keypoint_track_valid")
    baseline_y = metrics.get("measurement_reference", {}).get("grounded_foot_baseline_y_px")
    if foot_indices is None or points_data is None or confidence_data is None or valid_data is None:
        return frame_bgr
    if baseline_y is None or not np.isfinite(baseline_y):
        return frame_bgr

    points = np.asarray(points_data, dtype=np.float32)
    confidence = np.asarray(confidence_data, dtype=np.float32)
    valid = np.asarray(valid_data, dtype=bool)
    foot_indices = np.asarray(foot_indices, dtype=np.int32)
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or confidence.ndim != 1
        or valid.ndim != 1
        or len(points) != len(confidence)
        or len(points) != len(valid)
        or np.any(foot_indices >= len(points))
    ):
        return frame_bgr

    foot_points = points[foot_indices, :2]
    visible_feet = (
        valid[foot_indices]
        & np.isfinite(foot_points).all(axis=1)
        & (confidence[foot_indices] >= CONFIG["KEYPOINT_THRESHOLD"])
    )
    if not np.any(visible_feet):
        return frame_bgr

    visible_points = foot_points[visible_feet]
    foot_x, foot_y = np.rint(visible_points[np.argmax(visible_points[:, 1])]).astype(int)
    camera_compensation = bool(
        metrics.get("scene_context", {}).get("camera_motion", {}).get("compensation_applied")
    )
    transform_data = record.get("camera_transform_to_reference_xy")
    transform = np.asarray(transform_data, dtype=np.float32) if transform_data is not None else None
    if camera_compensation and transform is not None and transform.shape == (2, 3):
        reference_foot = apply_affine_transform(np.asarray((foot_x, foot_y)), transform)
        reference_to_frame = np.linalg.inv(np.vstack((transform, (0.0, 0.0, 1.0))))[:2]
        baseline_point = apply_affine_transform(
            np.asarray((reference_foot[0], float(baseline_y))),
            reference_to_frame,
        )
        baseline_y = int(round(float(baseline_point[1])))
    else:
        camera_shift_data = record.get("camera_shift_xy_px")
        camera_shift = np.asarray(
            camera_shift_data if camera_shift_data is not None else [0.0, 0.0],
            dtype=np.float32,
        )
        camera_y = float(camera_shift[1]) if camera_compensation and camera_shift.shape == (2,) else 0.0
        baseline_y = int(round(float(baseline_y) + camera_y))
    bbox_data = record.get("player_bbox_xyxy_px")
    bbox = np.asarray(bbox_data, dtype=np.float32) if bbox_data is not None else np.full(4, np.nan)
    draw_foot_height_markers(
        frame_bgr,
        (int(foot_x), int(foot_y)),
        baseline_y,
        bbox,
        events,
    )
    return frame_bgr

def _draw_timeline(
    image: Image.Image,
    capture: cv2.VideoCapture,
    context: _DashboardContext,
    selected_index: int | None,
    lang: str,
) -> dict[int, tuple[tuple[int, int, int, int], int]]:
    frames = context.view.frames
    metrics = context.view.metrics
    phase_markers = context.phase_markers
    fonts = context.fonts
    draw = ImageDraw.Draw(image)
    box = (24, 744, _WIDTH - 24, _HEIGHT - 24)
    draw_card(draw, box)
    draw_text(draw, (44, 757), translate("dashboard.timeline", lang), fonts["title"], DASHBOARD_TEXT)
    if selected_index is not None:
        selected_time = frames[selected_index].get("timestamp_sec", 0.0)
        summary = translate("dashboard.frame_summary", lang).format(
            frame=selected_index + 1,
            total=len(frames),
            seconds=float(selected_time),
        )
        summary_width = int(draw.textlength(summary, font=fonts["small"]))
        draw_text(draw, (_WIDTH - 46 - summary_width, 761), summary, fonts["small"], DASHBOARD_MUTED)

    load_index = int(
        np.clip(
            phase_frame(metrics, "load", 0),
            0,
            len(frames) - 1,
        )
    )
    lift_index = int(
        np.clip(
            phase_frame(metrics, "lift", load_index),
            load_index,
            len(frames) - 1,
        )
    )
    has_release = context.view.has_release
    release_index = (
        int(
            np.clip(
                phase_frame(metrics, "release", lift_index),
                lift_index,
                len(frames) - 1,
            )
        )
        if has_release
        else len(frames) - 1
    )
    follow_start = min(release_index + 2, len(frames) - 1) if has_release else release_index
    left, right = 44, _WIDTH - 44
    bar_y, bar_height = 798 + _TIMELINE_CONTENT_OFFSET_Y, 10
    track_width = right - left
    dip_index = next(
        (index for index, title, _ in phase_markers if title == "dashboard.phase_dip"),
        None,
    )
    boundaries = [0]
    phase_segments = [(None, _PHASE_COLORS["preparation"])]
    if dip_index is not None:
        boundaries.append(dip_index)
        phase_segments.append((None, _PHASE_COLORS["dip"]))
    boundaries.extend((load_index, lift_index))
    phase_segments.extend(
        (
            (None, _PHASE_COLORS["ball_lift"]),
            (None, _PHASE_COLORS["rise"]),
        )
    )
    if has_release:
        boundaries.extend((release_index, follow_start, len(frames) - 1))
        phase_segments.extend(
            (
                (None, _PHASE_COLORS["release"]),
                ("dashboard.phase_follow_through", _PHASE_COLORS["follow_through"]),
            )
        )
    else:
        boundaries.append(len(frames) - 1)
    sample_count = 14
    sample_indices = {
        round(index * (len(frames) - 1) / (sample_count - 1)) for index in range(sample_count)
    }
    sample_indices.update(
        int(phase["frame"]) - 1
        for phase in metrics.get("phases", [])
        if phase.get("frame") is not None and 1 <= int(phase["frame"]) <= len(frames)
    )
    sample_indices.update(
        index for index, _, _ in phase_markers if 0 <= index < len(frames)
    )
    sample_indices = sorted(sample_indices)
    for (label_key, color), start, boundary_end in zip(
        phase_segments,
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        end = max(boundary_end, start + 1)
        start_x = left + round(start / max(len(frames) - 1, 1) * track_width)
        end_x = left + round(end / max(len(frames) - 1, 1) * track_width)
        draw.rounded_rectangle(
            (start_x, bar_y, max(end_x, start_x + 4), bar_y + bar_height),
            radius=4,
            fill=color,
        )
        if label_key is None:
            continue
        label = translate(label_key, lang)
        label_width = int(draw.textlength(label, font=fonts["small"]))
        if end_x - start_x > label_width + 2:
            draw_text(
                draw,
                (start_x + (end_x - start_x - label_width) // 2, 778 + _TIMELINE_CONTENT_OFFSET_Y),
                label,
                fonts["small"],
                color,
            )
    label_rows: list[list[tuple[int, int]]] = []
    label_height = fonts["small"].size
    label_top = bar_y - label_height - 6
    label_gap = 6
    for frame_index, title_key, color in sorted(phase_markers, key=lambda marker: marker[0]):
        marker_x = left + round(frame_index / max(len(frames) - 1, 1) * track_width)
        label = translate(title_key, lang)
        label_width = int(draw.textlength(label, font=fonts["small"]))
        label_left = int(
            np.clip(marker_x - label_width // 2, left, right - label_width)
        )
        row = 0
        while row < len(label_rows) and any(
            label_left < placed_right + label_gap
            and label_left + label_width > placed_left - label_gap
            for placed_left, placed_right in label_rows[row]
        ):
            row += 1
        if row == len(label_rows):
            label_rows.append([])
        label_rows[row].append((label_left, label_left + label_width))
        draw_text(
            draw,
            (label_left, label_top - row * (label_height + 4)),
            label,
            fonts["small"],
            color,
        )
        draw.ellipse(
            (marker_x - 5, bar_y + 1, marker_x + 5, bar_y + 11),
            fill=color,
            outline=DASHBOARD_TEXT,
            width=1,
        )

    slots = len(sample_indices)
    slot_width = track_width // slots
    thumb_top, thumb_height = 821 + _TIMELINE_CONTENT_OFFSET_Y, 98
    thumb_width = max(slot_width - 10, 1)
    thumbnail_boxes = {}
    for slot, frame_index in enumerate(sample_indices):
        frame = _read_video_frame(capture, frame_index)
        thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        slot_left = left + slot * slot_width
        thumb_x = slot_left + (slot_width - thumb.width) // 2
        thumb_y = thumb_top + (thumb_height - thumb.height) // 2
        image.paste(thumb, (thumb_x, thumb_y))
        bounds = (thumb_x - 2, thumb_y - 2, thumb_x + thumb.width + 1, thumb_y + thumb.height + 1)
        thumbnail_boxes[frame_index] = (bounds, slot)
        draw.rounded_rectangle(
            bounds,
            radius=5,
            outline=DASHBOARD_BORDER,
            width=1,
        )
        if slot % 2 == 0:
            frame_label = str(frame_index + 1)
            label_width = int(draw.textlength(frame_label, font=fonts["tiny"]))
            draw_text(draw, (slot_left + (slot_width - label_width) // 2, 925 + _TIMELINE_CONTENT_OFFSET_Y), frame_label, fonts["tiny"], DASHBOARD_MUTED)

    if selected_index is not None:
        _draw_timeline_cursor(
            image,
            context,
            selected_index,
            thumbnail_boxes,
            lang,
            _read_video_frame(capture, selected_index),
        )
    return thumbnail_boxes

def _draw_timeline_cursor(
    image: Image.Image,
    context: _DashboardContext,
    selected_index: int,
    thumbnail_boxes: dict[int, tuple[tuple[int, int, int, int], int]],
    lang: str,
    current_frame_bgr: np.ndarray,
) -> None:
    frames = context.view.frames
    fonts = context.fonts
    phase_markers = context.phase_markers
    draw = ImageDraw.Draw(image)
    selected_time = float(frames[selected_index].get("timestamp_sec", 0.0))
    summary = translate("dashboard.frame_summary", lang).format(
        frame=selected_index + 1,
        total=len(frames),
        seconds=selected_time,
    )
    summary_width = int(draw.textlength(summary, font=fonts["small"]))
    draw_text(draw, (_WIDTH - 46 - summary_width, 761), summary, fonts["small"], DASHBOARD_MUTED)

    marker_x = 44 + round(selected_index / max(len(frames) - 1, 1) * (_WIDTH - 88))
    if thumbnail_boxes:
        template_bounds, _ = min(
            thumbnail_boxes.values(),
            key=lambda item: abs((item[0][0] + item[0][2]) / 2 - marker_x),
        )
        box_width = template_bounds[2] - template_bounds[0]
        box_height = template_bounds[3] - template_bounds[1]
        box_left = int(np.clip(marker_x - box_width // 2, 0, _WIDTH - box_width))
        box_top = template_bounds[1]
        current_bounds = (box_left, box_top, box_left + box_width, box_top + box_height)
        preview_area = (
            current_bounds[0] + 2,
            current_bounds[1] + 2,
            current_bounds[2] - 2,
            current_bounds[3] - 2,
        )
        preview, preview_x, preview_y = _fit_source_video_frame(current_frame_bgr, preview_area)
        image.paste(preview, (preview_x, preview_y))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(current_bounds, radius=5, outline=DASHBOARD_ANGLE_COLORS["knee"], width=3)
        frame_label = str(selected_index + 1)
        label_width = int(draw.textlength(frame_label, font=fonts["tiny"]))
        label_left = int(np.clip(marker_x - label_width // 2, 44, _WIDTH - 44 - label_width))
        draw.rectangle((label_left - 2, 922 + _TIMELINE_CONTENT_OFFSET_Y, label_left + label_width + 2, 940 + _TIMELINE_CONTENT_OFFSET_Y), fill=DASHBOARD_PANEL)
        draw_text(draw, (label_left, 925 + _TIMELINE_CONTENT_OFFSET_Y), frame_label, fonts["tiny"], DASHBOARD_ANGLE_COLORS["knee"])

    draw_dashed_line(draw, (marker_x, 792 + _TIMELINE_CONTENT_OFFSET_Y), (marker_x, 944 + _TIMELINE_CONTENT_OFFSET_Y), DASHBOARD_ANGLE_COLORS["knee"], dash=7, gap=6, width=2)
    for frame_index, _, color in phase_markers:
        event_x = 44 + round(frame_index / max(len(frames) - 1, 1) * (_WIDTH - 88))
        draw.ellipse(
            (event_x - 5, 799 + _TIMELINE_CONTENT_OFFSET_Y, event_x + 5, 809 + _TIMELINE_CONTENT_OFFSET_Y),
            fill=color,
            outline=DASHBOARD_TEXT,
            width=1,
        )

def _prepare_dashboard_context(
    metrics: dict[str, Any],
    metadata: VideoMetadata,
    observations: PoseObservations,
    tracked: TrackedPoses,
) -> _DashboardContext:
    view = prepare_shot_analysis_view(metrics, metadata, observations, tracked)
    layout = _dashboard_layout(view.metrics["video_metadata"])
    fonts = _dashboard_fonts(compact=layout.right_width < 700)
    marker_colors = {
        "dashboard.phase_ball_lift": _PHASE_COLORS["ball_lift"],
        "dashboard.phase_rise": _PHASE_COLORS["rise"],
        "dashboard.phase_release": _PHASE_COLORS["release"],
        "dashboard.phase_dip": _PHASE_COLORS["dip"],
        "dashboard.phase_forearm_peak": _PHASE_COLORS["preparation"],
    }
    phase_markers = [(index, title, marker_colors[title]) for index, title in view.phase_markers]
    return _DashboardContext(view, phase_markers, layout, fonts)

def _draw_video_panel(
    canvas: Image.Image,
    frame_bgr: np.ndarray,
    context: _DashboardContext,
    frame_index: int,
    lang: str,
) -> None:
    layout = context.layout
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(layout.video_area, fill=DASHBOARD_BACKGROUND)
    video_image, video_x, video_y = _fit_source_video_frame(frame_bgr, layout.video_area)
    canvas.paste(video_image, (video_x, video_y))
    draw.rounded_rectangle(
        (video_x - 1, video_y - 1, video_x + video_image.width, video_y + video_image.height),
        radius=8,
        outline=DASHBOARD_BORDER,
        width=1,
    )
    _draw_frame_chip(draw, layout.video_area, frame_index, context.view.frames, context.fonts, lang)

def _draw_analysis_panels(
    draw: ImageDraw.ImageDraw,
    context: _DashboardContext,
    current_index: int,
    lang: str,
    *,
    complete_metrics: bool = False,
) -> None:
    draw_release_metrics(
        draw,
        context.layout.release_cards_box,
        context.view,
        current_index,
        context.fonts,
        lang,
        complete_metrics=complete_metrics,
    )
    draw_jump_height_panel(
        draw,
        context.layout.jump_box,
        context.view,
        current_index,
        context.fonts,
        lang,
        complete_metrics=complete_metrics,
    )
    draw_ball_stats_panel(
        draw,
        context.layout.ball_height_box,
        context.view,
        current_index,
        context.fonts,
        lang,
        complete_metrics=complete_metrics,
    )
    draw_joint_angle_chart(
        draw,
        context.layout.chart_box,
        context.view,
        current_index,
        context.phase_markers,
        context.fonts,
        lang,
    )

def _draw_dashboard_base(
    capture: cv2.VideoCapture,
    context: _DashboardContext,
    selected_index: int | None,
    lang: str,
) -> tuple[Image.Image, dict[int, tuple[tuple[int, int, int, int], int]]]:
    release_index = context.view.release_index
    frame = _draw_pose_annotations(
        _read_video_frame(capture, release_index),
        context.view.frames[release_index],
        context.view.side,
        context.view.keypoint_index,
    )
    frame = _draw_jump_height_markers(
        frame,
        context.view.frames[release_index],
        context.view.metrics,
        context.view.side,
        release_index,
    )

    dashboard = Image.new("RGB", (_WIDTH, _HEIGHT), DASHBOARD_BACKGROUND)
    draw = ImageDraw.Draw(dashboard)
    draw_card(draw, context.layout.left_box)
    _draw_video_panel(
        dashboard,
        frame,
        context,
        release_index,
        lang,
    )
    _draw_context_badges(draw, context.layout, context.view.metrics, context.fonts, lang)
    _draw_analysis_panels(
        draw,
        context,
        release_index,
        lang,
        complete_metrics=True,
    )
    thumbnails = _draw_timeline(
        dashboard,
        capture,
        context,
        selected_index,
        lang,
    )
    return dashboard, thumbnails

def render_analysis_dashboard_video(
    video_path: Path,
    metrics: dict[str, Any],
    metadata: VideoMetadata,
    observations: PoseObservations,
    tracked: TrackedPoses,
    output_path: Path,
    *,
    lang: str = "en",
) -> None:
    """Render the dashboard video with phase holds and a moving source-frame playhead."""
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}")
    context = _prepare_dashboard_context(metrics, metadata, observations, tracked)

    if Path(video_path).resolve() == Path(output_path).resolve():
        raise ValueError("Dashboard output path must differ from the source video path")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {video_path}")
    try:
        dashboard, thumbnail_boxes = _draw_dashboard_base(capture, context, None, lang)
    except Exception:
        capture.release()
        raise
    fps = float(
        context.view.metrics["video_metadata"].get("fps")
        or capture.get(cv2.CAP_PROP_FPS)
        or 30.0
    )
    pause_frames = {
        int(phase["frame"]) - 1
        for phase in context.view.metrics.get("phases", [])
        if phase.get("id") in {"load", "lift", "release"} and phase.get("frame") is not None
    }
    pause_frames.update(
        index
        for index, title, _ in context.phase_markers
        if title in {"dashboard.phase_dip", "dashboard.phase_forearm_peak"}
    )
    maximum_jump_frame = (
        context.view.metrics.get("calculation_details", {})
        .get("jump_height", {})
        .get("maximum_frame")
    )
    if (
        phase_metric(context.view.metrics, "max_jump_height_ratio") is not None
        and maximum_jump_frame is not None
        and 0 <= int(maximum_jump_frame) - 1 < len(context.view.frames)
    ):
        pause_frames.add(int(maximum_jump_frame) - 1)
    pause_frame_count = max(round(fps) - 1, 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temporary_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (_WIDTH, _HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        temporary_path.unlink(missing_ok=True)
        raise OSError(f"Could not create dashboard video: {output_path}")

    render_progress = tqdm(
        total=len(context.view.frames) + len(pause_frames) * pause_frame_count,
        desc=f"Generating {output_path.name}",
        unit="frames",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_index, record in enumerate(context.view.frames):
            ok, source_frame = capture.read()
            if not ok:
                raise OSError(f"Could not decode source frame {frame_index} during dashboard rendering")

            timeline_frame = source_frame.copy()
            source_frame = _draw_pose_annotations(
                source_frame,
                record,
                context.view.side,
                context.view.keypoint_index,
            )
            source_frame = _draw_jump_height_markers(
                source_frame,
                record,
                context.view.metrics,
                context.view.side,
                frame_index,
            )
            canvas = dashboard.copy()
            _draw_video_panel(
                canvas,
                source_frame,
                context,
                frame_index,
                lang,
            )
            draw = ImageDraw.Draw(canvas)
            _draw_analysis_panels(
                draw,
                context,
                frame_index,
                lang,
            )
            _draw_timeline_cursor(
                canvas,
                context,
                frame_index,
                thumbnail_boxes,
                lang,
                timeline_frame,
            )
            output_frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            for paused_frame in frames_with_pause(
                output_frame,
                frame_index,
                pause_frames,
                pause_frame_count,
            ):
                writer.write(paused_frame)
                render_progress.update(1)
    except Exception:
        render_progress.close()
        writer.release()
        capture.release()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        render_progress.close()
        writer.release()
        capture.release()

    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise OSError(f"Dashboard renderer produced no video: {output_path}")
    temporary_path.replace(output_path)

