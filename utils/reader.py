"""Read video metadata and prepare videos for OpenCV."""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.data import VideoMetadata


def _has_decodable_frame(video_path: Path) -> bool:
    """Check the first frame while ensuring the capture is closed."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return capture.isOpened() and capture.read()[0]
    finally:
        capture.release()


def prepare_video_for_opencv(video_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Check first-frame decoding and create a temporary H.264 copy if needed."""
    if _has_decodable_frame(video_path):
        return video_path, None

    temporary_directory = tempfile.TemporaryDirectory(prefix="basketball_video_compat_")
    compatible_path = Path(temporary_directory.name) / f"{video_path.stem}_opencv_compatible.mp4"
    try:
        conversion = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                str(compatible_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        temporary_directory.cleanup()
        raise RuntimeError(
            f"OpenCV cannot decode {video_path.name}; ffmpeg is required for the decoder fallback."
        ) from error

    if conversion.returncode != 0 or not compatible_path.is_file():
        detail = conversion.stderr.strip()[-1000:]
        temporary_directory.cleanup()
        raise RuntimeError(
            f"OpenCV cannot decode {video_path.name}, and ffmpeg could not create a compatible video: {detail}"
        )

    if not _has_decodable_frame(compatible_path):
        temporary_directory.cleanup()
        raise RuntimeError(f"The compatible video created for {video_path.name} still has no decodable frames.")

    print(f"Video decoder fallback: {video_path.name} -> temporary H.264", flush=True)
    return compatible_path, temporary_directory


def read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read dimensions and estimate FPS from initial frame timestamps."""
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 30.0

    timestamp_samples = []
    for _ in range(12):
        has_frame, _ = capture.read()
        if not has_frame:
            break
        frame_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if np.isfinite(frame_timestamp) and frame_timestamp >= 0:
            timestamp_samples.append(frame_timestamp)
    timestamp_intervals = np.diff(timestamp_samples)
    valid_intervals = timestamp_intervals[np.isfinite(timestamp_intervals) & (timestamp_intervals > 0)]
    if len(valid_intervals) >= 3:
        timestamp_fps = 1.0 / float(np.median(valid_intervals))
        if not np.isfinite(reported_fps) or reported_fps <= 0 or np.isclose(timestamp_fps, reported_fps, rtol=0.1):
            fps = timestamp_fps
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    metadata = VideoMetadata(
        fps=fps,
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return metadata
