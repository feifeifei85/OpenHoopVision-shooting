"""Parse and validate command-line arguments."""

import argparse
from pathlib import Path

from core.config import CONFIG
from pipeline.runner import run_analysis
from render.i18n import SUPPORTED_LANGUAGES


def main() -> None:
    """Parse command-line options and analyze the selected video."""
    parser = argparse.ArgumentParser(description="Run RF-DETR player detection and RTMW3D-L pose analysis on a video.")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip annotated video output",
    )
    parser.add_argument(
        "--lang",
        choices=SUPPORTED_LANGUAGES,
        default="en",
        help="Language for text rendered in output.mp4 (default: en)",
    )
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not CONFIG["DETECTOR_CHECKPOINT"].is_file():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {CONFIG['DETECTOR_CHECKPOINT']}")

    run_analysis(args.video, metrics_only=args.metrics_only, lang=args.lang)


if __name__ == "__main__":
    main()
