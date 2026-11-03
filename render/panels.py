"""Draw pose overlays and 3D scenes with shared rendering primitives."""

from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import CONFIG
from core.geometry import SKELETON_EDGES
from render.i18n import translate


@lru_cache(maxsize=16)
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load and cache the configured font for rendered labels."""
    return ImageFont.truetype(CONFIG["CJK_BOLD_FONT"] if bold else CONFIG["CJK_REGULAR_FONT"], size, index=2)


def keypoint_color(index: int) -> tuple[int, int, int]:
    if index in CONFIG["LEFT_KEYPOINT_IDS"]:
        return (255, 90, 20)  # left side, blue in BGR
    if index in CONFIG["RIGHT_KEYPOINT_IDS"]:
        return (25, 105, 255)  # right side, orange/red in BGR
    return (75, 225, 80)  # face and central torso


def draw_depth_tested_segment(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_depth: float,
    end_depth: float,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Rasterize a projected 3D segment with interpolated camera depth."""
    height, width = depth_buffer.shape
    margin = thickness // 2 + 1
    x0 = max(min(int(start[0]), int(end[0])) - margin, 0)
    x1 = min(max(int(start[0]), int(end[0])) + margin + 1, width)
    y0 = max(min(int(start[1]), int(end[1])) - margin, 0)
    y1 = min(max(int(start[1]), int(end[1])) + margin + 1, height)
    if x0 >= x1 or y0 >= y1:
        return

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    p0 = (int(start[0]) - x0, int(start[1]) - y0)
    p1 = (int(end[0]) - x0, int(end[1]) - y0)
    cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_8)
    local_y, local_x = np.nonzero(mask)
    if not len(local_x):
        return
    xs, ys = local_x + x0, local_y + y0
    delta = (end - start).astype(np.float32)
    length_squared = float(np.dot(delta, delta))
    if length_squared <= 1e-6:
        t = np.full(len(xs), 0.5, dtype=np.float32)
    else:
        t = np.clip(
            ((xs - start[0]) * delta[0] + (ys - start[1]) * delta[1]) / length_squared,
            0.0,
            1.0,
        )
    depth = start_depth + t * (end_depth - start_depth)
    closer = depth > depth_buffer[ys, xs]
    ys, xs, depth = ys[closer], xs[closer], depth[closer]
    canvas[ys, xs] = color
    depth_buffer[ys, xs] = depth


def draw_depth_tested_joint(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    center: np.ndarray,
    depth: float,
    radius: int = 4,
) -> None:
    height, width = depth_buffer.shape
    x0, x1 = max(int(center[0]) - radius, 0), min(int(center[0]) + radius + 1, width)
    y0, y1 = max(int(center[1]) - radius, 0), min(int(center[1]) + radius + 1, height)
    if x0 >= x1 or y0 >= y1:
        return
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]) - x0, int(center[1]) - y0), radius, 255, -1, cv2.LINE_8)
    local_y, local_x = np.nonzero(mask)
    xs, ys = local_x + x0, local_y + y0
    closer = depth >= depth_buffer[ys, xs]
    ys, xs = ys[closer], xs[closer]
    canvas[ys, xs] = (245, 245, 245)
    depth_buffer[ys, xs] = depth


def draw_source_pose(
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Draw a two-dimensional skeleton over a copy of the source frame."""
    image = frame_bgr.copy()
    visible = np.isfinite(keypoints).all(axis=1) & (confidence >= threshold)
    for a, b in SKELETON_EDGES:
        if visible[a] and visible[b]:
            pa = tuple(np.rint(keypoints[a, :2]).astype(int))
            pb = tuple(np.rint(keypoints[b, :2]).astype(int))
            cv2.line(image, pa, pb, keypoint_color(b), 2, cv2.LINE_AA)
    for index in np.flatnonzero(visible):
        point = tuple(np.rint(keypoints[index, :2]).astype(int))
        cv2.circle(image, point, 3, keypoint_color(int(index)), -1, cv2.LINE_AA)
    return image


def draw_foot_height_markers(
    frame_bgr: np.ndarray,
    foot_xy: tuple[int, int],
    baseline_y: int,
    bbox_xyxy: np.ndarray | None,
    colors_bgr: Sequence[tuple[int, int, int]],
) -> None:
    """Draw the ground baseline and colored foot-height event markers."""
    if not colors_bgr:
        return

    foot_x, foot_y = foot_xy
    bbox = np.asarray(bbox_xyxy, dtype=np.float32) if bbox_xyxy is not None else None
    half_line_width = (
        max(24, int(round((bbox[2] - bbox[0]) * 0.3)))
        if bbox is not None and bbox.shape == (4,) and np.isfinite(bbox).all()
        else 24
    )
    line_start = max(0, foot_x - half_line_width)
    line_end = min(frame_bgr.shape[1] - 1, foot_x + half_line_width)
    line_width = max(round(frame_bgr.shape[1] / 960), 2)
    dash_length = max(round(frame_bgr.shape[1] / 120), 8)
    for dash_start in range(line_start, line_end, dash_length * 2):
        cv2.line(
            frame_bgr,
            (dash_start, baseline_y),
            (min(dash_start + dash_length, line_end), baseline_y),
            colors_bgr[0],
            line_width,
            cv2.LINE_AA,
        )

    for event_index, color in enumerate(colors_bgr):
        marker_x = min(max(foot_x + event_index * 8, 0), frame_bgr.shape[1] - 1)
        cv2.line(
            frame_bgr,
            (marker_x, min(foot_y, baseline_y)),
            (marker_x, max(foot_y, baseline_y)),
            color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame_bgr,
            (marker_x, baseline_y),
            line_width * 2,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(frame_bgr, (marker_x, foot_y), line_width * 2, color, -1, cv2.LINE_AA)


def draw_angle_overlays(
    image: np.ndarray,
    angles: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]],
) -> None:
    """Draw joint-angle guides over the rendered pose."""
    if not angles:
        return
    for first, vertex, last, color in angles:
        if not (np.isfinite(first[:2]).all() and np.isfinite(vertex[:2]).all() and np.isfinite(last[:2]).all()):
            continue
        first_vector, last_vector = first[:2] - vertex[:2], last[:2] - vertex[:2]
        lengths = np.linalg.norm((first_vector, last_vector), axis=1)
        if np.min(lengths) < 8:
            continue
        center = tuple(np.rint(vertex[:2]).astype(int))
        endpoints = (tuple(np.rint(first[:2]).astype(int)), tuple(np.rint(last[:2]).astype(int)))
        for endpoint in endpoints:
            cv2.line(image, center, endpoint, (15, 20, 28), 7, cv2.LINE_AA)
            cv2.line(image, center, endpoint, color, 3, cv2.LINE_AA)
        first_angle = np.arctan2(first_vector[1], first_vector[0])
        last_angle = np.arctan2(last_vector[1], last_vector[0])
        delta = (last_angle - first_angle + np.pi) % (2 * np.pi) - np.pi
        radius = max(14, min(int(np.min(lengths) * 0.28), int(image.shape[1] * 0.045)))
        arc_angles = np.linspace(first_angle, first_angle + delta, max(12, int(abs(delta) * radius / 3)))
        arc = np.rint(vertex[:2] + radius * np.column_stack((np.cos(arc_angles), np.sin(arc_angles)))).astype(np.int32)
        cv2.polylines(image, [arc], False, (15, 20, 28), 8, cv2.LINE_AA)
        cv2.polylines(image, [arc], False, color, 4, cv2.LINE_AA)
        cv2.circle(image, center, 6, (15, 20, 28), -1, cv2.LINE_AA)
        cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)


def pose_world_coordinates(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    bbox: np.ndarray | None,
    ground_y_px: float | None,
    body_height_px: float | None,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    visible = np.isfinite(keypoints).all(axis=1) & (confidence >= threshold)
    world = np.full_like(keypoints, np.nan, dtype=np.float32)
    if not np.any(visible):
        return world, visible

    box_valid = bbox is not None and np.isfinite(bbox).all() and bbox[3] > bbox[1]
    box_height = float(bbox[3] - bbox[1]) if box_valid else np.nan
    # Normalize each frame locally so camera distance does not resize the displayed body.
    frame_body_height = box_height if box_valid else body_height_px
    body_height = (
        float(frame_body_height)
        if frame_body_height is not None and np.isfinite(frame_body_height) and frame_body_height > 0
        else 1.0
    )
    ground_y = (
        ground_y_px
        if ground_y_px is not None and np.isfinite(ground_y_px)
        else (float(bbox[3]) if box_valid else float(np.max(keypoints[visible, 1])))
    )
    hips = np.array((CONFIG["KEYPOINT_INDEX"]["left_hip"], CONFIG["KEYPOINT_INDEX"]["right_hip"]))
    visible_hips = hips[visible[hips]]
    root_x = (
        float(np.mean(keypoints[visible_hips, 0]))
        if len(visible_hips)
        else (float((bbox[0] + bbox[2]) * 0.5) if box_valid else float(np.mean(keypoints[visible, 0])))
    )
    root_depth = (
        float(np.mean(keypoints[visible_hips, 2])) if len(visible_hips) else float(np.median(keypoints[visible, 2]))
    )

    world[visible, 0] = (keypoints[visible, 0] - root_x) / body_height
    world[visible, 1] = (keypoints[visible, 2] - root_depth) / CONFIG["RTMW3D_CONFIG"]["z_range"] * depth_scale
    world[visible, 2] = (ground_y - keypoints[visible, 1]) / body_height
    return world, visible


def project_scene_points(
    points: np.ndarray,
    width: int,
    height: int,
    header_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = points - CONFIG["POSE_CAMERA_POSITION"]
    distance = relative @ CONFIG["POSE_CAMERA_FORWARD"]
    view_height = height - header_height
    focal_length = min(view_height * 2.0, width * 2.0)
    safe_distance = np.maximum(distance, 1e-4)
    center_fraction = 0.5 + 0.1 * max(0.0, 1.0 - width / max(view_height, 1))
    center = np.array((width * 0.5, header_height + view_height * center_fraction), dtype=np.float32)
    screen = np.column_stack((relative @ CONFIG["POSE_CAMERA_RIGHT"], relative @ CONFIG["POSE_CAMERA_UP"]))
    screen[:, 0] = center[0] + focal_length * screen[:, 0] / safe_distance
    screen[:, 1] = center[1] - focal_length * screen[:, 1] / safe_distance
    return screen, -distance, distance > 0.05


def pose_depth_scale(boxes: np.ndarray, body_height_px: float | None) -> float:
    """Scale model depth relative to the detected player's body size."""
    box_heights = boxes[:, 3] - boxes[:, 1]
    valid_heights = box_heights[np.isfinite(box_heights) & (box_heights > 0)]
    reference_height = (
        body_height_px
        if body_height_px is not None and np.isfinite(body_height_px) and body_height_px > 0
        else (float(np.median(valid_heights)) if len(valid_heights) else 1.0)
    )
    return float(np.median(valid_heights) / reference_height) if len(valid_heights) else 1.0


@lru_cache(maxsize=16)
def pose_scene_background(width: int, height: int, header_height: int, lang: str) -> np.ndarray:
    panel = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(header_height, height):
        fade = (y - header_height) / max(height - header_height, 1)
        panel[y] = (17 + round(10 * fade), 15 + round(12 * fade), 13 + round(13 * fade))

    header = Image.new("RGB", (width, header_height), (7, 14, 20))
    header_draw = ImageDraw.Draw(header)
    header_draw.text(
        (24, 18), translate("panel.skeleton", lang), font=load_font(min(max(22, width // 34), 32), True),
        fill=(232, 241, 245),
    )
    depth_label = translate("panel.floor", lang)
    depth_font = load_font(min(max(11, width // 62), 18), False)
    depth_width = header_draw.textbbox((0, 0), depth_label, font=depth_font)[2]
    header_draw.text((width - 24 - depth_width, 34), depth_label, font=depth_font, fill=(128, 151, 165))
    panel[:header_height] = cv2.cvtColor(np.asarray(header), cv2.COLOR_RGB2BGR)
    cv2.line(panel, (0, 0), (width - 1, 0), (229, 220, 48), 2, cv2.LINE_AA)
    cv2.line(panel, (24, header_height), (width - 24, header_height), (45, 65, 75), 1, cv2.LINE_AA)

    tile_size = 0.5
    tile_count_x = 56
    tile_count_y = 50
    x_start, y_start = -tile_count_x * tile_size * 0.5, -1.5
    tiles = []
    for row in range(tile_count_y):
        near_y, far_y = y_start + row * tile_size, y_start + (row + 1) * tile_size
        for column in range(tile_count_x):
            left_x, right_x = x_start + column * tile_size, x_start + (column + 1) * tile_size
            corners = np.array(
                (
                    (left_x, near_y, 0.0),
                    (right_x, near_y, 0.0),
                    (right_x, far_y, 0.0),
                    (left_x, far_y, 0.0),
                ),
                dtype=np.float32,
            )
            projected, camera_depth, in_front = project_scene_points(corners, width, height, header_height)
            if not np.all(in_front):
                continue
            distance = -float(np.mean(camera_depth))
            fade = float(np.clip(1.0 - max(distance - 2.5, 0.0) / 24.0, 0.35, 1.0))
            base = np.array((59, 69, 71) if (row + column) % 2 else (41, 51, 53), dtype=np.float32)
            background = np.array((20, 20, 18), dtype=np.float32)
            color = tuple(int(value) for value in np.rint(background * (1.0 - fade) + base * fade))
            polygon = np.rint(projected).astype(np.int32)
            tiles.append((distance, polygon, color))

    floor_panel = panel[header_height:]
    for _, polygon, color in sorted(tiles, key=lambda tile: tile[0], reverse=True):
        polygon[:, 1] -= header_height
        cv2.fillConvexPoly(floor_panel, polygon, color, cv2.LINE_AA)
        if np.max(np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1)) > 4:
            cv2.polylines(floor_panel, [polygon], True, (54, 64, 66), 1, cv2.LINE_AA)
    return panel


def render_3d_panel(
    height: int,
    width: int,
    keypoints: np.ndarray | None,
    confidence: np.ndarray | None,
    threshold: float,
    bbox: np.ndarray | None = None,
    ground_y_px: float | None = None,
    body_height_px: float | None = None,
    depth_scale: float = 1.0,
    *,
    lang: str = "en",
) -> np.ndarray:
    """Render the pose in a calibrated scene with its ground reference."""
    header_height = 76
    panel = pose_scene_background(width, height, header_height, lang).copy()
    if keypoints is None or confidence is None:
        keypoints = np.full((CONFIG["KEYPOINT_COUNT"], 3), np.nan, dtype=np.float32)
        confidence = np.zeros(CONFIG["KEYPOINT_COUNT"], dtype=np.float32)
    world_points, visible = pose_world_coordinates(
        keypoints,
        confidence,
        threshold,
        bbox,
        ground_y_px,
        body_height_px,
        depth_scale,
    )
    projected, camera_depth, in_front = project_scene_points(world_points, width, height, header_height)
    visible &= in_front
    if not np.any(visible):
        return panel

    screen = np.zeros((len(keypoints), 2), dtype=np.int32)
    screen[visible] = np.rint(projected[visible]).astype(np.int32)
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float32)
    for a, b in SKELETON_EDGES:
        if visible[a] and visible[b]:
            draw_depth_tested_segment(
                panel,
                depth_buffer,
                screen[a],
                screen[b],
                float(camera_depth[a]),
                float(camera_depth[b]),
                keypoint_color(b),
            )
    for index in np.flatnonzero(visible[: CONFIG["KEYPOINT_COUNT"]]):
        draw_depth_tested_joint(panel, depth_buffer, screen[index], float(camera_depth[index]))
    return panel
