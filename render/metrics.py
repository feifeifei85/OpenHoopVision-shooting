"""Render shot metrics and joint-angle charts."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis.report import (
    shot_metric_phases,
    shot_phase_time_intervals,
    shot_validation_error_codes,
)
from core.data import ShotMetrics
from render.records import ShotAnalysisView, phase_frame, phase_metric
from render.i18n import metric_label, metric_unit, phase_title, translate
from render.panels import load_font

@lru_cache(maxsize=4096)
def _text_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    return font.getbbox(text)

@lru_cache(maxsize=4096)
def _text_mask(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    left, top, right, bottom = _text_bbox(text, font)
    mask = Image.new("L", (max(right - left, 1), max(bottom - top, 1)), 0)
    ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255)
    return mask

def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, _, _ = _text_bbox(text, font)
    draw.bitmap((position[0] + left, position[1] + top), _text_mask(text, font), fill=fill)

def render_metrics_panel(
    height: int,
    width: int,
    frame_index: int,
    timestamps: np.ndarray,
    metrics: ShotMetrics,
    scale_limit: float = 2.4,
    *,
    annotation: dict | None = None,
    lang: str = "en",
) -> np.ndarray:
    """Draw a progressive shot-analysis timeline with the current metric highlighted."""
    source_phases = shot_metric_phases(metrics, timestamps) if annotation is None else [
        {
            "id": phase["id"], "title": phase["name"], "status": phase["status"],
            "frame_index": phase["frame"] - 1 if phase["frame"] is not None else None,
            "timestamp_sec": phase["time_sec"], "trigger": phase["name"], "metrics": phase["metrics"],
        }
        for phase in annotation["phases"]
    ]
    phases = []
    for source_phase in source_phases:
        if source_phase["id"] == "rim_contact" and source_phase["frame_index"] is None:
            continue
        phases.append(
            {
                **source_phase,
                "title": phase_title(source_phase["id"], source_phase["title"], lang),
                "metrics": [
                    {
                        **item,
                        "label": metric_label(item["id"], item["label"], lang),
                        "display_unit": metric_unit(item["unit"], lang),
                    }
                    for item in source_phase["metrics"]
                ],
            }
        )
    metric_columns = 3 if width >= 480 else 2
    row_counts = [max(1, (len(phase["metrics"]) + metric_columns - 1) // metric_columns) for phase in phases]
    card_units = [42 + 42 * rows for rows in row_counts]
    base_scale = min(width / 440, scale_limit)
    timeline_events, phase_time_intervals = shot_phase_time_intervals(metrics, timestamps, phases)
    if annotation is not None:
        phase_time_intervals = {phase["id"]: phase for phase in annotation["phases"]}

    header_units = 80
    footer_units = 20 if not phases else 0
    card_gap_units = 4
    bottom_padding_units = 8
    layout_units = (
        header_units + footer_units + sum(card_units) + card_gap_units * max(len(phases) - 1, 0) + bottom_padding_units
    )
    scale = min(base_scale, max(height - 6, 1) / max(layout_units, 1))
    for _ in range(4):
        measured_height = (
            round(header_units * scale)
            + sum(round(units * scale) for units in card_units)
            + max(round(card_gap_units * scale), 2) * max(len(phases) - 1, 0)
            + round(footer_units * scale)
            + max(round(bottom_padding_units * scale), 2)
        )
        if measured_height <= height:
            break
        scale *= max((height - 4) / measured_height, 0.1)
    label, small, value_font, index_font = (
        load_font(max(round(size * scale), 1), bold)
        for size, bold in ((16, False), (14, False), (20, True), (14, True))
    )
    margin = round(24 * scale)
    cyan, amber = (48, 220, 229), (255, 186, 66)
    image = Image.new("RGB", (width, height), (7, 14, 21))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(31, 54, 68))
    draw.rectangle((0, 0, width - 1, max(round(2 * scale), 2)), fill=cyan)
    rim_found = np.isfinite(metrics.rim_box).all()
    has_metric_errors = bool(shot_validation_error_codes(metrics.validation_errors))
    badge_text = (
        translate("status.validation_warning", lang)
        if has_metric_errors
        else translate("status.metrics_unavailable", lang)
        if metrics.unavailable_metric_codes
        else translate("status.rim_missing", lang)
        if not rim_found
        else ""
    )
    badge_color = (
        (255, 186, 66)
        if has_metric_errors or metrics.unavailable_metric_codes
        else (70, 222, 171)
        if rim_found
        else (133, 152, 164)
    )
    badge_warning = bool(has_metric_errors or metrics.unavailable_metric_codes)
    if badge_text:
        badge_width = _text_bbox(badge_text, small)[2] + round(28 * scale)
        badge_y = round(4 * scale)
        draw.rounded_rectangle(
            (width - margin - badge_width, badge_y, width - margin, badge_y + round(27 * scale)),
            radius=round(5 * scale),
            fill=(39, 29, 14) if badge_warning else (19, 27, 33),
            outline=(111, 77, 27) if badge_warning else (43, 58, 68),
        )
        draw.ellipse(
            (
                width - margin - badge_width + round(9 * scale),
                badge_y + round(9 * scale),
                width - margin - badge_width + round(17 * scale),
                badge_y + round(17 * scale),
            ),
            fill=badge_color,
        )
        _draw_text(
            draw,
            (width - margin - badge_width + round(22 * scale), badge_y + round(3 * scale)),
            badge_text,
            font=small,
            fill=badge_color,
        )

    hand_key = (
        "hand.left"
        if metrics.shooting_side == "left"
        else "hand.right"
        if metrics.shooting_side == "right"
        else "hand.unknown"
    )
    view = (
        annotation["scene_context"]["player_orientation_to_camera"]["mode"]
        if annotation is not None else metrics.video_profile.view_mode
    )
    view_key = f"view.{view}"
    if view_key not in {"view.frontal", "view.profile", "view.oblique"}:
        view_key = "view.unknown"
    meta = translate("meta", lang).format(
        hand=translate(hand_key, lang),
        view=translate(view_key, lang),
        seconds=timestamps[frame_index],
        frame=frame_index + 1,
        total=len(timestamps),
    )
    meta_y = round(29 * scale)
    _draw_text(draw, (margin, meta_y), meta, font=small, fill=(153, 174, 188))

    def fit_font(
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        bold: bool = False,
    ) -> ImageFont.FreeTypeFont:
        size = font.size
        minimum_size = max(6, round(8 * scale))
        while size > minimum_size:
            bounds = _text_bbox(text, font)
            if bounds[2] - bounds[0] <= max_width:
                break
            size -= 1
            font = load_font(size, bold)
        return font

    progress = (frame_index + 1) / max(len(timestamps), 1)
    active_phase_index = max(
        (
            index
            for index, phase in enumerate(phases)
            if phase["frame_index"] is not None and phase["frame_index"] <= frame_index
        ),
        default=None,
    )
    progress_y = round(62 * scale)
    progress_right = width - margin
    draw.rounded_rectangle(
        (margin, progress_y, progress_right, progress_y + max(round(4 * scale), 3)),
        radius=round(3 * scale),
        fill=(25, 42, 52),
    )
    progress_x = margin + int((progress_right - margin) * progress)
    marker_colors = {
        "ball_low": (112, 160, 255),
        "load": cyan,
        "lift": (100, 205, 156),
        "extension": (100, 205, 156),
        "release": amber,
        "rim_contact": (220, 100, 255),
    }
    tail_color = (151, 181, 198)
    phase_markers = [
        (
            {"id": event_id, "frame_index": event_frame},
            margin + round((progress_right - margin) * event_frame / max(len(timestamps) - 1, 1)),
            marker_colors.get(event_id, cyan),
        )
        for event_id, event_frame in timeline_events
    ]
    if phase_markers:
        phase_markers[0] = (phase_markers[0][0], phase_markers[0][1], (205, 216, 222))
    segment_start = margin
    segment_color = phase_markers[0][2] if phase_markers else cyan
    for marker_index, (_, marker_x, marker_color) in enumerate(phase_markers):
        segment_end = min(marker_x, progress_x)
        if segment_end > segment_start:
            draw.rectangle(
                (segment_start, progress_y, segment_end, progress_y + max(round(4 * scale), 3)),
                fill=segment_color,
            )
        if progress_x <= marker_x:
            segment_start = progress_x
            break
        segment_start = marker_x
        segment_color = phase_markers[marker_index + 1][2] if marker_index + 1 < len(phase_markers) else tail_color
    if progress_x > segment_start:
        draw.rectangle(
            (segment_start, progress_y, progress_x, progress_y + max(round(4 * scale), 3)),
            fill=segment_color,
        )

    marker_y = progress_y + round(2 * scale)
    marker_radius = max(round(4 * scale), 3)
    for phase, marker_x, marker_color in phase_markers:
        draw.ellipse(
            (marker_x - marker_radius, marker_y - marker_radius, marker_x + marker_radius, marker_y + marker_radius),
            fill=marker_color,
            outline=(7, 14, 21),
            width=max(round(scale), 1),
        )

    top, footer = round(header_units * scale), round(footer_units * scale)
    if not phases:
        center_y = (top + height - footer) // 2
        center_x = width // 2
        reticle_color = (35, 61, 73)
        draw.ellipse((center_x - 34, center_y - 34, center_x + 34, center_y + 34), outline=reticle_color, width=2)
        draw.ellipse((center_x - 4, center_y - 4, center_x + 4, center_y + 4), fill=cyan)
        draw.line((center_x - 48, center_y, center_x - 14, center_y), fill=reticle_color, width=2)
        draw.line((center_x + 14, center_y, center_x + 48, center_y), fill=reticle_color, width=2)
        message_key = "event.none" if frame_index == len(timestamps) - 1 else "event.waiting"
        message = translate(message_key, lang)
        message_width = _text_bbox(message, label)[2]
        _draw_text(
            draw,
            (center_x - message_width // 2, center_y + round(56 * scale)), message, font=label, fill=(137, 164, 178)
        )
    else:
        card_gap = max(round(card_gap_units * scale), 2)
        current_y = float(top)
        for phase_index, phase in enumerate(phases):
            active = phase_index == active_phase_index
            completed = active_phase_index is not None and phase_index < active_phase_index
            card_height = card_units[phase_index] * scale
            y = round(current_y)
            row_bottom = round(current_y + card_height)
            current_y += card_height + card_gap
            if active:
                row_fill, row_edge = (13, 36, 46), (28, 126, 139)
                node_color, title_color, value_color = amber, (232, 245, 248), amber
                pill_fill = (58, 43, 22)
            elif completed:
                row_fill, row_edge = (10, 22, 30), (20, 42, 54)
                node_color, title_color, value_color = (54, 160, 171), (164, 185, 195), (138, 163, 174)
                pill_fill = (14, 35, 41)
            else:
                row_fill, row_edge = (8, 16, 23), (24, 42, 53)
                node_color, title_color, value_color = (61, 90, 102), (119, 142, 153), (93, 116, 130)
                pill_fill = (15, 25, 31)
            draw.rounded_rectangle(
                (margin, y, width - margin, row_bottom),
                radius=round(7 * scale),
                fill=row_fill,
                outline=row_edge,
                width=max(round(scale), 1),
            )
            pill_x = margin + round(8 * scale)
            pill_y = y + round(5 * scale)
            pill_width, pill_height = round(26 * scale), round(23 * scale)
            draw.rounded_rectangle(
                (pill_x, pill_y, pill_x + pill_width, pill_y + pill_height),
                radius=round(5 * scale),
                fill=pill_fill,
            )
            _draw_text(
                draw,
                (pill_x + round(5 * scale), y + round(7 * scale)),
                f"{phase_index + 1:02d}",
                font=index_font,
                fill=node_color,
            )

            event_frame = phase["frame_index"]
            if event_frame is not None and (active or completed):
                detail = translate("phase.frame", lang).format(title=phase["title"], frame=event_frame + 1)
            elif completed:
                detail = translate("phase.unconfirmed", lang).format(title=phase["title"])
            else:
                detail = translate("phase.waiting", lang).format(title=phase["title"])
            text_x = margin + round(42 * scale)
            phase_duration = phase_time_intervals.get(phase["id"], {}).get("duration_sec")
            time_text = f"{phase_duration:.1f}s" if event_frame is not None and phase_duration is not None else None
            time_bounds = _text_bbox(time_text, small) if time_text is not None else None
            time_width = time_bounds[2] - time_bounds[0] if time_bounds is not None else 0
            right_text_edge = width - margin - round(12 * scale)
            detail_width = right_text_edge - text_x - time_width - round(12 * scale)
            detail_font = fit_font(detail, small, max(detail_width, 1))
            _draw_text(draw, (text_x, y + round(8 * scale)), detail, font=detail_font, fill=node_color)
            if time_text is not None:
                _draw_text(
                    draw,
                    (right_text_edge - time_bounds[2], y + round(8 * scale)),
                    time_text,
                    font=small,
                    fill=marker_colors.get(phase["id"], (153, 174, 188)),
                )

            measurements = phase["metrics"]
            columns = min(max(len(measurements), 1), metric_columns)
            grid_left = margin + round(18 * scale)
            grid_right = width - margin - round(18 * scale)
            cell_width = (grid_right - grid_left) / columns
            cell_text_width = max(1, int(cell_width - round(14 * scale)))
            grid_top = y + round(32 * scale)
            grid_row_height = round(42 * scale)
            widest_label = max(
                (item["label"] for item in measurements),
                default="",
                key=lambda text: _text_bbox(text, small)[2],
            )
            label_font = fit_font(widest_label, small, cell_text_width)
            numeric_values = []
            for item in measurements:
                amount = item["value"]
                if amount is None:
                    continue
                if item["unit"] in {"倍身高", "倍可见身高"}:
                    numeric_values.append(f"{amount:.2f} {item['display_unit']}")
                elif item["display_unit"]:
                    numeric_values.append(f"{amount:.1f}{item['display_unit']}")
                else:
                    numeric_values.append(f"{amount:.2f}")
            value_font_phase = (
                fit_font(
                    max(numeric_values, key=lambda text: _text_bbox(text, value_font)[2]),
                    value_font,
                    cell_text_width,
                    True,
                )
                if numeric_values
                else value_font
            )
            for item_index, item in enumerate(measurements):
                column, row = item_index % columns, item_index // columns
                x = round(grid_left + column * cell_width + round(4 * scale))
                metric_y = grid_top + row * grid_row_height
                _draw_text(draw, (x, metric_y), item["label"], font=label_font, fill=title_color)
                amount = item["value"]
                if not (active or completed):
                    value_text, value_font_row = translate("metric.pending", lang), small
                    value_bold = False
                elif amount is None:
                    value_text, value_font_row = translate("metric.unavailable", lang), label
                    value_bold = False
                elif item["unit"] in {"倍身高", "倍可见身高"}:
                    value_text, value_font_row = f"{amount:.2f} {item['display_unit']}", value_font_phase
                    value_bold = True
                elif item["display_unit"]:
                    value_text, value_font_row = f"{amount:.1f}{item['display_unit']}", value_font_phase
                    value_bold = True
                else:
                    value_text, value_font_row = f"{amount:.2f}", value_font_phase
                    value_bold = True
                if not (active or completed) or amount is None:
                    value_font_row = fit_font(value_text, value_font_row, cell_text_width, value_bold)
                _draw_text(
                    draw,
                    (x, metric_y + round(16 * scale)),
                    value_text,
                    font=value_font_row,
                    fill=value_color,
                )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

DASHBOARD_BACKGROUND = (9, 15, 22)
DASHBOARD_PANEL = (15, 24, 33)
DASHBOARD_BORDER = (37, 52, 66)
DASHBOARD_TEXT = (237, 243, 249)
DASHBOARD_MUTED = (151, 168, 184)
DASHBOARD_ANGLE_COLORS = {
    "knee": (255, 48, 56),
    "hip": (25, 158, 122),
    "shoulder": (255, 213, 0),
    "elbow": (36, 106, 255),
}

def draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(position, value, font=font, fill=fill)

def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 14) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=DASHBOARD_PANEL, outline=DASHBOARD_BORDER, width=1)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int],
    dash: int = 9,
    gap: int = 7,
    width: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = max(abs(x1 - x0), abs(y1 - y0))
    if length == 0:
        return
    for offset in range(0, length, dash + gap):
        stop = min(offset + dash, length)
        first = (x0 + (x1 - x0) * offset // length, y0 + (y1 - y0) * offset // length)
        last = (x0 + (x1 - x0) * stop // length, y0 + (y1 - y0) * stop // length)
        draw.line((first, last), fill=fill, width=width)


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    detail: str,
    accent: tuple[int, int, int],
    fonts: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
) -> None:
    draw_card(draw, box)
    x0, y0, x1, _ = box
    label_font, value_font, detail_font = fonts
    compact = x1 - x0 < 140
    inset = 11 if compact else 16
    dot_size = 8 if compact else 9
    title_x = x0 + inset + dot_size + 8
    draw.ellipse((x0 + inset, y0 + 19, x0 + inset + dot_size, y0 + 19 + dot_size), fill=accent)
    draw_text(draw, (title_x, y0 + 14), title, label_font, DASHBOARD_MUTED)
    value_width = int(draw.textlength(value, font=value_font))
    value_x = x0 + (x1 - x0 - value_width) // 2
    draw_text(draw, (value_x, y0 + 48), value, value_font, DASHBOARD_TEXT)
    detail_width = int(draw.textlength(detail, font=detail_font))
    detail_x = x0 + (x1 - x0 - detail_width) // 2
    draw_text(draw, (detail_x, y0 + 99), detail, detail_font, DASHBOARD_MUTED)

def draw_release_metrics(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    view: ShotAnalysisView,
    current_index: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
    *,
    complete_metrics: bool = False,
) -> None:
    metrics = view.metrics
    series = view.series
    frames = view.frames
    release_index = view.release_index
    chart_start_index = view.chart_start_index
    chart_index = int(np.clip(current_index, chart_start_index, release_index))
    x0, y0, x1, y1 = box
    gap = 8
    card_width = (x1 - x0 - gap * 4) // 5
    release_angle = phase_metric(metrics, "release_angle_deg")
    release_reached = view.has_release and (complete_metrics or current_index >= release_index)
    elbow_angle = series["elbow"][chart_index]
    lift_index = phase_frame(metrics, "lift", chart_start_index)
    elbow_speed = None
    if current_index >= lift_index:
        speed_end_index = chart_index
        start_angle = series["elbow"][lift_index]
        end_angle = series["elbow"][speed_end_index]
        elapsed = float(frames[speed_end_index]["timestamp_sec"]) - float(frames[lift_index]["timestamp_sec"])
        if np.isfinite(start_angle) and np.isfinite(end_angle) and elapsed > 0:
            elbow_speed = abs(end_angle - start_angle) / elapsed
        if view.has_release and current_index >= release_index:
            release_speed = phase_metric(metrics, "elbow_angular_velocity_deg_s")
            if release_speed is not None:
                elbow_speed = release_speed
    motion_value = phase_metric(metrics, "motion_value")
    calculation_details = metrics.get("calculation_details", {})
    motion_details = calculation_details.get("motion", {})
    motion_class = motion_details.get("class")
    peak_frame = motion_details.get("peak_frame")
    peak_index = int(peak_frame) - 1 if peak_frame is not None else None
    motion_complete = view.has_release and (
        complete_metrics or (current_index >= release_index and (peak_index is None or current_index >= peak_index))
    )
    if not view.has_release:
        motion_label = translate("metric.unavailable", lang)
    elif not motion_complete:
        motion_label = translate("dashboard.motion_pending", lang)
    else:
        motion_label = {
            "一段式": translate("dashboard.motion_single", lang),
            "二段式": translate("dashboard.motion_two", lang),
        }.get(motion_class, translate("metric.unavailable", lang))
    current_timestamp = float(frames[current_index].get("timestamp_sec", 0.0))
    cards = (
        (
            "dashboard.release_frame",
            str(release_index + 1) if view.has_release else "N/A",
            translate("dashboard.current_frame_progress", lang).format(
                frame=current_index + 1,
                total=len(frames),
                seconds=current_timestamp,
            )
            if view.has_release
            else translate("metric.unavailable", lang),
            DASHBOARD_ANGLE_COLORS["knee"],
        ),
        (
            "dashboard.release_angle",
            f"{release_angle:.1f}°" if release_reached and release_angle is not None else "N/A",
            translate("dashboard.at_release", lang)
            if release_reached
            else translate("dashboard.event_pending", lang)
            if view.has_release
            else translate("metric.unavailable", lang),
            DASHBOARD_ANGLE_COLORS["elbow"],
        ),
        (
            "dashboard.forearm_angle",
            f"{elbow_angle:.1f}°"
            if current_index >= chart_start_index and np.isfinite(elbow_angle)
            else "N/A",
            translate("dashboard.at_release", lang)
            if view.has_release and current_index >= release_index
            else translate("dashboard.at_shot_start", lang)
            if chart_index == chart_start_index
            else translate("dashboard.at_current_frame", lang),
            DASHBOARD_ANGLE_COLORS["elbow"],
        ),
        (
            "dashboard.elbow_speed",
            f"{elbow_speed:.0f}°/s" if elbow_speed is not None else "N/A",
            translate("dashboard.at_release", lang)
            if view.has_release and current_index >= release_index
            else translate("dashboard.at_current_frame", lang),
            DASHBOARD_ANGLE_COLORS["shoulder"],
        ),
        (
            "dashboard.shot_motion",
            f"{motion_value:.2f}" if motion_complete and motion_value is not None else "N/A",
            motion_label,
            DASHBOARD_ANGLE_COLORS["hip"],
        ),
    )
    card_fonts = (fonts["metric_title"], fonts["value"], fonts["detail"])
    for index, (title_key, value, detail, color) in enumerate(cards):
        left = x0 + index * (card_width + gap)
        _draw_metric_card(
            draw,
            (left, y0, left + card_width, y1),
            translate(title_key, lang),
            value,
            detail,
            color,
            card_fonts,
        )

def draw_jump_height_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    view: ShotAnalysisView,
    current_index: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
    *,
    complete_metrics: bool = False,
) -> None:
    metrics = view.metrics
    frames = view.frames
    release_index = view.release_index
    chart_start_index = view.chart_start_index
    draw_card(draw, box)
    x0, y0, x1, _ = box
    title = translate("dashboard.jump_height", lang)
    draw_text(draw, (x0 + 18, y0 + 12), title, fonts["panel_title"], DASHBOARD_TEXT)
    subtitle_x = x0 + 18 + int(draw.textlength(title, font=fonts["panel_title"])) + 8
    draw_text(draw, (subtitle_x, y0 + 17), translate("dashboard.body_height_ratio", lang), fonts["tiny"], DASHBOARD_MUTED)

    release_reached = view.has_release and (complete_metrics or current_index >= release_index)
    release_value = phase_metric(metrics, "release_foot_lift_ratio") if release_reached else None
    peak_frame = metrics.get("calculation_details", {}).get("jump_height", {}).get("maximum_frame")
    peak_index = int(peak_frame) - 1 if peak_frame is not None else None
    maximum_value = phase_metric(metrics, "max_jump_height_ratio") if complete_metrics else None
    if not complete_metrics and current_index >= chart_start_index:
        last_index = min(current_index, peak_index) if peak_index is not None else current_index
        observed_heights = [
            float(frame["jump_height_ratio"])
            for frame in frames[chart_start_index : last_index + 1]
            if frame.get("jump_height_ratio") is not None
            and np.isfinite(frame["jump_height_ratio"])
        ]
        if observed_heights:
            maximum_value = max(observed_heights)
        if peak_index is not None and current_index >= peak_index:
            annotated_maximum = phase_metric(metrics, "max_jump_height_ratio")
            if annotated_maximum is not None:
                maximum_value = annotated_maximum
    columns = (
        (
            "dashboard.jump_at_release",
            release_value,
            ""
            if release_value is not None
            else translate("dashboard.event_pending", lang)
            if view.has_release
            else translate("metric.unavailable", lang),
        ),
        (
            "dashboard.jump_maximum",
            maximum_value,
            translate("dashboard.at_frame", lang).format(frame=int(peak_frame))
            if peak_frame is not None and (complete_metrics or current_index >= peak_index)
            else translate("dashboard.jump_so_far", lang)
            if maximum_value is not None
            else "",
        ),
    )
    inner_width = x1 - x0 - 32
    column_width = inner_width // 2
    for index, (label_key, value, detail) in enumerate(columns):
        left = x0 + 16 + index * column_width
        if index:
            draw.line((left, y0 + 48, left, y0 + 124), fill=DASHBOARD_BORDER, width=1)
        draw_text(draw, (left + 10, y0 + 48), translate(label_key, lang), fonts["small"], DASHBOARD_MUTED)
        value_text = f"{value:.2f}x" if value is not None else "N/A"
        draw_text(draw, (left + 10, y0 + 70), value_text, fonts["value"], DASHBOARD_TEXT)
        draw_text(draw, (left + 10, y0 + 108), detail, fonts["detail"], DASHBOARD_MUTED)

def draw_ball_stats_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    view: ShotAnalysisView,
    current_index: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
    *,
    complete_metrics: bool = False,
) -> None:
    metrics = view.metrics
    frames = view.frames
    draw_card(draw, box)
    x0, y0, x1, _ = box
    title = translate("dashboard.ball_stats", lang)
    draw_text(draw, (x0 + 18, y0 + 12), title, fonts["panel_title"], DASHBOARD_TEXT)
    subtitle_x = x0 + 18 + int(draw.textlength(title, font=fonts["panel_title"])) + 8
    draw_text(
        draw,
        (subtitle_x, y0 + 17),
        translate("dashboard.body_height_ratio", lang),
        fonts["tiny"],
        DASHBOARD_MUTED,
    )

    phase_frames = {
        phase.get("id"): int(phase["frame"]) - 1
        for phase in metrics.get("phases", [])
        if phase.get("id") in {"load", "lift", "release"} and phase.get("frame") is not None
    }
    values = (
        ("dashboard.phase_load_short", "load", "knee_ball_height_ratio"),
        ("dashboard.phase_lift_short", "lift", "takeoff_ball_height_ratio"),
        ("dashboard.phase_release_short", "release", "release_height_ratio"),
        ("dashboard.ball_speed_short", None, "ball_speed_body_heights_per_s"),
    )
    inner_width = x1 - x0 - 28
    column_width = inner_width // len(values)
    for index, (label_key, phase_id, metric_id) in enumerate(values):
        left = x0 + 14 + index * column_width
        if index:
            draw.line((left, y0 + 49, left, y0 + 124), fill=DASHBOARD_BORDER, width=1)
        label = translate(label_key, lang)
        label_width = int(draw.textlength(label, font=fonts["small"]))
        draw_text(draw, (left + (column_width - label_width) // 2, y0 + 48), label, fonts["small"], DASHBOARD_MUTED)
        if phase_id is None:
            value = frames[current_index].get(metric_id)
            value_text = (
                f"{value:.1f}x/s"
                if value is not None and np.isfinite(value)
                else "N/A"
            )
        else:
            event_frame = phase_frames.get(phase_id)
            reached = complete_metrics or (event_frame is not None and current_index >= event_frame)
            value = phase_metric(metrics, metric_id) if reached else None
            value_text = f"{value:.2f}x" if value is not None else "N/A"
        value_width = int(draw.textlength(value_text, font=fonts["value"]))
        value_x = left + (column_width - value_width) // 2
        draw_text(draw, (value_x, y0 + 70), value_text, fonts["value"], DASHBOARD_TEXT)

def _smooth_curve(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Interpolate chart points with a shape-preserving cubic curve."""
    if len(points) < 3:
        return [(round(x), round(y)) for x, y in points]

    coordinates = np.asarray(points, dtype=np.float64)
    x, y = coordinates[:, 0], coordinates[:, 1]
    intervals = np.diff(x)
    slopes = np.diff(y) / intervals
    tangents = np.empty_like(y)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    for index in range(1, len(y) - 1):
        previous_slope, next_slope = slopes[index - 1], slopes[index]
        if previous_slope * next_slope <= 0:
            tangents[index] = 0.0
            continue
        previous_interval, next_interval = intervals[index - 1], intervals[index]
        first_weight = 2 * next_interval + previous_interval
        second_weight = next_interval + 2 * previous_interval
        tangents[index] = (first_weight + second_weight) / (
            first_weight / previous_slope + second_weight / next_slope
        )

    curve = []
    for index, interval in enumerate(intervals):
        first_x, next_x = x[index], x[index + 1]
        first_y, next_y = y[index], y[index + 1]
        for step in range(8):
            position = step / 8
            position_squared = position * position
            position_cubed = position_squared * position
            first_weight = 2 * position_cubed - 3 * position_squared + 1
            first_tangent_weight = position_cubed - 2 * position_squared + position
            next_weight = -2 * position_cubed + 3 * position_squared
            next_tangent_weight = position_cubed - position_squared
            curve.append(
                (
                    round(first_x + position * interval),
                    round(
                        first_weight * first_y
                        + first_tangent_weight * interval * tangents[index]
                        + next_weight * next_y
                        + next_tangent_weight * interval * tangents[index + 1]
                    ),
                )
            )
    curve.append((round(x[-1]), round(y[-1])))
    return curve

def draw_joint_angle_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    view: ShotAnalysisView,
    selected_index: int,
    phase_markers: list[tuple[int, str, tuple[int, int, int]]],
    fonts: dict[str, ImageFont.FreeTypeFont],
    lang: str,
) -> None:
    series = view.series
    timestamps = view.timestamps
    start_index = view.chart_start_index
    end_index = view.release_index
    x0, y0, x1, y1 = box
    draw_card(draw, box)
    draw_text(draw, (x0 + 20, y0 + 13), translate("dashboard.joint_angles", lang), fonts["panel_title"], DASHBOARD_TEXT)
    frame_count = min(
        len(timestamps),
        max((len(values) for values in series.values()), default=0),
    )
    start_index = int(np.clip(start_index, 0, max(frame_count - 1, 0)))
    end_index = int(np.clip(end_index, start_index, max(frame_count - 1, start_index)))
    shot_started = selected_index >= start_index
    selected_index = int(np.clip(selected_index, start_index, end_index))
    plot_left, plot_right = x0 + 62, x1 - 148
    plot_top, plot_bottom = y0 + 92, y1 - 52
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    legend = (
        ("knee", "dashboard.knee_angle"),
        ("hip", "dashboard.hip_angle"),
        ("shoulder", "dashboard.upper_arm_angle"),
        ("elbow", "dashboard.forearm_angle"),
    )
    legend_x = x0 + 20
    legend_y = y0 + 42
    for name, label_key in legend:
        label = translate(label_key, lang)
        label_width = int(draw.textlength(label, font=fonts["tiny"]))
        draw.ellipse((legend_x, legend_y + 4, legend_x + 8, legend_y + 12), fill=DASHBOARD_ANGLE_COLORS[name])
        draw_text(draw, (legend_x + 13, legend_y), label, fonts["tiny"], DASHBOARD_MUTED)
        legend_x += 21 + label_width + 9
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill=(22, 23, 24))
    for degrees in range(0, 181, 20):
        y = plot_bottom - round(degrees / 180 * plot_height)
        draw.line((plot_left, y, plot_right, y), fill=(58, 58, 58), width=1)
        draw_text(draw, (x0 + 13, y - 9), f"{degrees}°", fonts["tiny"], DASHBOARD_MUTED)
    if frame_count and len(timestamps) == frame_count:
        start_time = float(timestamps[start_index])
        end_time = float(timestamps[end_index])
        duration = max(end_time - start_time, 1e-6)
        tick_values = [duration * tick / 5 for tick in range(6)]
        tick_precision = 1
        if len({f"{elapsed:.1f}" for elapsed in tick_values}) < len(tick_values):
            tick_precision = 2
        for tick in range(6):
            elapsed = tick_values[tick]
            x = plot_left + round(tick / 5 * plot_width)
            draw.line((x, plot_top, x, plot_bottom), fill=(51, 51, 51), width=1)
            label = "0s" if tick == 0 else f"{elapsed:.{tick_precision}f}s"
            label_width = int(draw.textlength(label, font=fonts["tiny"]))
            draw_text(draw, (x - label_width // 2, plot_bottom + 9), label, fonts["tiny"], DASHBOARD_MUTED)

        visible_end_index = min(selected_index, end_index) if shot_started else start_index - 1
        for name, values in series.items():
            color = DASHBOARD_ANGLE_COLORS[name]
            points: list[tuple[float, float]] = []
            first_point: tuple[float, float] | None = None
            selected_point: tuple[float, float] | None = None
            for index in range(start_index, visible_end_index + 1):
                value = values[index]
                if np.isfinite(value):
                    elapsed = float(timestamps[index]) - start_time
                    x = plot_left + elapsed / duration * plot_width
                    y = plot_bottom - np.clip(value, 0, 180) / 180 * plot_height
                    point = (x, y)
                    points.append(point)
                    if first_point is None:
                        first_point = point
                    if index == selected_index:
                        selected_point = point
                elif len(points) > 1:
                    draw.line(_smooth_curve(points), fill=color, width=4, joint="curve")
                    points = []
                else:
                    points = []
            if len(points) > 1:
                draw.line(_smooth_curve(points), fill=color, width=4, joint="curve")
            if first_point is not None:
                x, y = map(round, first_point)
                draw.ellipse(
                    (x - 5, y - 5, x + 5, y + 5),
                    fill=color,
                    outline=(238, 238, 238),
                    width=1,
                )
            if selected_point is not None:
                x, y = map(round, selected_point)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(238, 238, 238), width=2)

        visible_markers = sorted(
            (
                (int(index), title_key, color)
                for index, title_key, color in phase_markers
                if start_index <= index <= end_index and index <= visible_end_index
            ),
            key=lambda marker: marker[0],
        )
        marker_labels = []
        for index, title_key, color in visible_markers:
            elapsed = float(timestamps[index]) - start_time
            marker_x = int(round(plot_left + elapsed / duration * plot_width))
            draw_dashed_line(
                draw,
                (marker_x, plot_top),
                (marker_x, plot_bottom),
                color,
                dash=7,
                gap=6,
                width=2,
            )
            elbow_value = series["elbow"][index]
            if np.isfinite(elbow_value):
                marker_y = int(round(plot_bottom - np.clip(elbow_value, 0, 180) / 180 * plot_height))
                draw.ellipse(
                    (marker_x - 6, marker_y - 6, marker_x + 6, marker_y + 6),
                    fill=color,
                    outline=(238, 238, 238),
                    width=2,
                )

            label = translate(title_key, lang)
            label_width = int(draw.textlength(label, font=fonts["tiny"])) + 14
            label_left = int(
                np.clip(marker_x - label_width // 2, plot_left, plot_right - label_width)
            )
            marker_labels.append((marker_x, label, color, label_left, label_width))

        label_gap = 6
        for index in range(1, len(marker_labels)):
            previous_left = marker_labels[index - 1][3]
            previous_width = marker_labels[index - 1][4]
            marker_x, label, color, label_left, label_width = marker_labels[index]
            label_left = max(label_left, previous_left + previous_width + label_gap)
            marker_labels[index] = (marker_x, label, color, label_left, label_width)
        if marker_labels and marker_labels[-1][3] + marker_labels[-1][4] > plot_right:
            overflow = marker_labels[-1][3] + marker_labels[-1][4] - plot_right
            marker_labels = [
                (marker_x, label, color, label_left - overflow, label_width)
                for marker_x, label, color, label_left, label_width in marker_labels
            ]

        for _, label, color, label_left, label_width in marker_labels:
            label_top = plot_top - 25
            label_right = label_left + label_width
            draw.rounded_rectangle(
                (label_left, label_top, label_right, label_top + 21),
                radius=6,
                fill=DASHBOARD_BACKGROUND,
                outline=color,
                width=1,
            )
            draw_text(draw, (label_left + 7, label_top + 3), label, fonts["tiny"], DASHBOARD_TEXT)

    draw.line((x1 - 135, plot_top, x1 - 135, plot_bottom), fill=DASHBOARD_BORDER, width=1)
    readouts = (
        ("hip", "dashboard.hip_angle"),
        ("knee", "dashboard.knee_angle"),
        ("elbow", "dashboard.forearm_angle"),
        ("shoulder", "dashboard.upper_arm_angle"),
    )
    row_height = (plot_bottom - plot_top) // len(readouts)
    for row, (name, label_key) in enumerate(readouts):
        value = (
            series[name][selected_index]
            if shot_started and 0 <= selected_index < len(series[name])
            else float("nan")
        )
        y = plot_top + row * row_height + 8
        value_label = f"{value:.1f}°" if np.isfinite(value) else "N/A"
        draw_text(draw, (x1 - 119, y), value_label, fonts["value"], DASHBOARD_ANGLE_COLORS[name])
        draw_text(draw, (x1 - 119, y + 44), translate(label_key, lang), fonts["tiny"], DASHBOARD_MUTED)
