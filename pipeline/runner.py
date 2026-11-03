"""Orchestrate shooting analysis and optional output generation."""

import logging
import warnings
from pathlib import Path

import torch

from analysis.report import build_shot_metrics_record
from analysis.shots import calculate_shot_metrics
from core.config import CONFIG
from models.inference import load_checkpoint_model
from models.rtmw3d import RTMW3DModel
from pipeline.observations import collect_pose_observations
from render.dashboard import render_analysis_dashboard_video
from tracking.keypoints import filter_pose_observations
from utils.output import write_shot_metrics_json
from utils.reader import prepare_video_for_opencv


def run_analysis(video_path: Path, *, metrics_only: bool = False, lang: str = "en") -> None:
    """Analyze one video and write its metrics and optional visualizations."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from rfdetr import from_checkpoint

    logging.getLogger("rf-detr").setLevel(logging.ERROR)
    pose_model = load_checkpoint_model(RTMW3DModel(), CONFIG["POSE_CHECKPOINT"], device, "RTMW3D-L")
    detector = from_checkpoint(str(CONFIG["DETECTOR_CHECKPOINT"]))
    detector.model.device = device

    print(f"Device: {device}; RF-DETR and RTMW3D-L weights loaded")
    print(f"Analyzing video: {video_path}", flush=True)
    compatible_video_path, compatibility_directory = prepare_video_for_opencv(video_path)
    try:
        metadata, observations = collect_pose_observations(
            compatible_video_path,
            pose_model=pose_model,
            detector=detector,
            device=device,
        )
        tracked = filter_pose_observations(observations, metadata)
        shot_metrics = calculate_shot_metrics(observations, tracked, metadata)
        metrics_record = build_shot_metrics_record(
            video_path,
            shot_metrics,
            observations,
            metadata,
        )
        write_shot_metrics_json(Path("metrics.json"), metrics_record)
        if not metrics_only:
            try:
                render_analysis_dashboard_video(
                    compatible_video_path,
                    metrics_record,
                    metadata,
                    observations,
                    tracked,
                    Path("output.mp4"),
                    lang=lang,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not render dashboard output; metrics.json is still available"
                )
                raise
    finally:
        if compatibility_directory is not None:
            compatibility_directory.cleanup()
