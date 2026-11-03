"""Load RTMW3D weights and decode pose estimates."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from core.config import CONFIG
from models.rtmw3d import RTMW3DModel


def load_checkpoint_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    model_name: str,
) -> nn.Module:
    """Load checkpoint weights and prepare the model for inference."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{model_name} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    del checkpoint, state_dict
    return model


def affine_crop(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply MMPose GetBBoxCenterScale + TopdownAffine for a single box."""
    input_w, input_h, _ = CONFIG["RTMW3D_CONFIG"]["input_size_whd"]
    center = (bbox_xyxy[:2] + bbox_xyxy[2:]) * 0.5
    scale = (bbox_xyxy[2:] - bbox_xyxy[:2]).astype(np.float32)
    scale *= CONFIG["RTMW3D_CONFIG"]["bbox_padding"]
    aspect_ratio = input_w / input_h
    if scale[0] > scale[1] * aspect_ratio:
        scale[1] = scale[0] / aspect_ratio
    else:
        scale[0] = scale[1] * aspect_ratio

    source_direction = np.array([-0.5 * scale[0], 0.0], dtype=np.float32)
    destination_direction = np.array([-0.5 * input_w, 0.0], dtype=np.float32)
    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    destination[0] = (input_w * 0.5, input_h * 0.5)
    destination[1] = destination[0] + destination_direction
    source_direction = source[0] - source[1]
    destination_direction = destination[0] - destination[1]
    source[2] = source[1] + np.array([-source_direction[1], source_direction[0]])
    destination[2] = destination[1] + np.array([-destination_direction[1], destination_direction[0]])
    warp = cv2.getAffineTransform(source, destination)
    crop = cv2.warpAffine(frame_bgr, warp, (input_w, input_h), flags=cv2.INTER_LINEAR)
    return crop, center.astype(np.float32), scale.astype(np.float32)


@torch.inference_mode()
def estimate_pose(
    model: RTMW3DModel,
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    device: torch.device,
    previous_visual_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate 3D keypoints and confidence for one detected person."""
    pose_bbox = bbox_xyxy.copy()
    pose_bbox[0] = max(
        0.0,
        pose_bbox[0] - (pose_bbox[2] - pose_bbox[0]) * CONFIG["RTMW3D_CONFIG"]["bbox_left_expansion_ratio"],
    )
    crop, center, scale = affine_crop(frame_bgr, pose_bbox)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.asarray(CONFIG["RTMW3D_CONFIG"]["mean_rgb"], dtype=np.float32)
    std = np.asarray(CONFIG["RTMW3D_CONFIG"]["std_rgb"], dtype=np.float32)
    tensor = torch.from_numpy(((crop_rgb - mean) / std).transpose(2, 0, 1).copy())
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    pred_x, pred_y, pred_z = model(tensor)
    pred_x = pred_x[0].float().cpu().numpy()
    pred_y = pred_y[0].float().cpu().numpy()
    pred_z = pred_z[0].float().cpu().numpy()

    # Official SimCC3DLabel decoder: argmax each axis, score=min(max_x,max_y).
    x_index = np.argmax(pred_x, axis=-1).astype(np.float32)
    y_index = np.argmax(pred_y, axis=-1).astype(np.float32)
    z_index = np.argmax(pred_z, axis=-1).astype(np.float32)
    visual_z_index = z_index.copy()
    confidence = np.minimum(np.max(pred_x, axis=-1), np.max(pred_y, axis=-1))
    split_ratio = CONFIG["RTMW3D_CONFIG"]["simcc_split_ratio"]
    input_w, input_h, input_d = CONFIG["RTMW3D_CONFIG"]["input_size_whd"]
    if previous_visual_depth is not None:
        max_jump = CONFIG["RTMW3D_CONFIG"]["visualization_depth_max_jump_bins"]
        score_tolerance = CONFIG["RTMW3D_CONFIG"]["visualization_depth_score_tolerance"]
        z_range = CONFIG["RTMW3D_CONFIG"]["z_range"]
        depth_bin_scale = (input_d / 2.0) * split_ratio
        for index, previous_depth in enumerate(previous_visual_depth):
            if not np.isfinite(previous_depth):
                continue
            previous_bin = int(round((previous_depth / z_range + 1.0) * depth_bin_scale))
            low = max(previous_bin - max_jump, 0)
            high = min(previous_bin + max_jump, pred_z.shape[-1] - 1)
            if low > high:
                continue
            nearby = pred_z[index, low : high + 1]
            nearby_bin = low + int(np.argmax(nearby))
            global_score = float(np.max(pred_z[index]))
            nearby_score = float(pred_z[index, nearby_bin])
            score_gap = global_score - nearby_score
            if score_gap <= score_tolerance * max(abs(global_score), 1e-6):
                visual_z_index[index] = nearby_bin
    keypoints = np.stack([x_index, y_index, z_index], axis=-1) / split_ratio
    visualization_keypoints = np.stack([x_index, y_index, visual_z_index], axis=-1) / split_ratio
    keypoints[:, 0] = keypoints[:, 0] / input_w * scale[0] + center[0] - 0.5 * scale[0]
    keypoints[:, 1] = keypoints[:, 1] / input_h * scale[1] + center[1] - 0.5 * scale[1]
    visualization_keypoints[:, :2] = keypoints[:, :2]
    invalid = confidence <= 0.0
    for points in (keypoints, visualization_keypoints):
        points[:, 2] = (points[:, 2] / (input_d / 2.0) - 1.0) * CONFIG["RTMW3D_CONFIG"]["z_range"]
        points[invalid] = -1.0
    return (
        keypoints.astype(np.float32),
        confidence.astype(np.float32),
        visualization_keypoints.astype(np.float32),
    )
