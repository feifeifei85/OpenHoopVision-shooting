"""Load shooting analysis settings from the project configuration."""

from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PROJECT_ROOT = CONFIG_PATH.parent

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

for key in ("DETECTOR_CHECKPOINT", "POSE_CHECKPOINT"):
    CONFIG[key] = PROJECT_ROOT / CONFIG[key]

# Preserve tuple semantics for skeleton landmark names.
for key in ("SIDE_NAMES", "TOE_NAMES"):
    CONFIG[key] = tuple(CONFIG[key])

CONFIG["POSE_WORLD_UP"] = np.asarray(CONFIG["CAMERA_CONFIG"]["world_up"], dtype=np.float32)
CONFIG["POSE_CAMERA_TARGET"] = np.asarray(CONFIG["CAMERA_CONFIG"]["target"], dtype=np.float32)
CONFIG["POSE_CAMERA_POSITION"] = CONFIG["POSE_CAMERA_TARGET"] + np.asarray(
    CONFIG["CAMERA_CONFIG"]["position_offset"], dtype=np.float32
)
CONFIG["POSE_CAMERA_FORWARD"] = CONFIG["POSE_CAMERA_TARGET"] - CONFIG["POSE_CAMERA_POSITION"]
CONFIG["POSE_CAMERA_FORWARD"] /= np.linalg.norm(CONFIG["POSE_CAMERA_FORWARD"])
CONFIG["POSE_CAMERA_RIGHT"] = np.cross(CONFIG["POSE_CAMERA_FORWARD"], CONFIG["POSE_WORLD_UP"])
CONFIG["POSE_CAMERA_RIGHT"] /= np.linalg.norm(CONFIG["POSE_CAMERA_RIGHT"])
CONFIG["POSE_CAMERA_UP"] = np.cross(CONFIG["POSE_CAMERA_RIGHT"], CONFIG["POSE_CAMERA_FORWARD"])
CONFIG["KEYPOINT_COUNT"] = CONFIG["RTMW3D_CONFIG"]["keypoint_count"]
CONFIG["FOOT_LANDMARK_NAMES"] = CONFIG["TOE_NAMES"] + (CONFIG["SKELETON_CONFIG"]["heel_name"],)
CONFIG["FOOT_NAMES"] = [f"{side}_{name}" for side in CONFIG["SIDE_NAMES"] for name in CONFIG["FOOT_LANDMARK_NAMES"]]
CONFIG["FACE_NAMES"] = [f"face_{i:02d}" for i in range(CONFIG["SKELETON_CONFIG"]["face_keypoint_count"])]
CONFIG["HAND_JOINT_COUNT"] = 1 + len(CONFIG["HAND_FINGER_BASE_OFFSETS"]) * CONFIG["HAND_JOINTS_PER_FINGER"]
CONFIG["FACE_START_INDEX"] = len(CONFIG["BODY_NAMES"]) + len(CONFIG["FOOT_NAMES"])
CONFIG["FACE_END_INDEX"] = CONFIG["FACE_START_INDEX"] + len(CONFIG["FACE_NAMES"])
CONFIG["HAND_BASES"] = {
    "left": CONFIG["FACE_END_INDEX"],
    "right": CONFIG["FACE_END_INDEX"] + CONFIG["HAND_JOINT_COUNT"],
}
CONFIG["HAND_KEYPOINT_INDICES"] = range(
    CONFIG["HAND_BASES"]["left"], CONFIG["HAND_BASES"]["right"] + CONFIG["HAND_JOINT_COUNT"]
)
CONFIG["HAND_JOINT_NAMES"] = ["wrist"] + [
    f"{finger}_{joint}"
    for finger in CONFIG["HAND_FINGER_BASE_OFFSETS"]
    for joint in range(1, CONFIG["HAND_JOINTS_PER_FINGER"] + 1)
]
CONFIG["KEYPOINT_NAMES"] = (
    CONFIG["BODY_NAMES"]
    + CONFIG["FOOT_NAMES"]
    + CONFIG["FACE_NAMES"]
    + [f"{side}_hand_{name}" for side in CONFIG["SIDE_NAMES"] for name in CONFIG["HAND_JOINT_NAMES"]]
)
CONFIG["KEYPOINT_INDEX"] = {name: index for index, name in enumerate(CONFIG["KEYPOINT_NAMES"])}
CONFIG["FOOT_KEYPOINT_INDICES"] = np.asarray(
    [CONFIG["KEYPOINT_INDEX"][f"{side}_ankle"] for side in CONFIG["SIDE_NAMES"]]
    + [CONFIG["KEYPOINT_INDEX"][name] for name in CONFIG["FOOT_NAMES"]],
    dtype=np.int32,
)
CONFIG["FOOT_KEYPOINT_INDICES_BY_SIDE"] = {
    side: np.asarray(
        [CONFIG["KEYPOINT_INDEX"][f"{side}_ankle"]]
        + [CONFIG["KEYPOINT_INDEX"][f"{side}_{name}"] for name in CONFIG["FOOT_LANDMARK_NAMES"]],
        dtype=np.int32,
    )
    for side in CONFIG["SIDE_NAMES"]
}
assert len(CONFIG["KEYPOINT_NAMES"]) == CONFIG["KEYPOINT_COUNT"]
CONFIG["LEFT_KEYPOINT_IDS"] = set(CONFIG["SKELETON_CONFIG"]["left_body_indices"]) | set(
    range(CONFIG["HAND_BASES"]["left"], CONFIG["HAND_BASES"]["left"] + CONFIG["HAND_JOINT_COUNT"])
)
CONFIG["RIGHT_KEYPOINT_IDS"] = set(CONFIG["SKELETON_CONFIG"]["right_body_indices"]) | set(
    range(CONFIG["HAND_BASES"]["right"], CONFIG["HAND_BASES"]["right"] + CONFIG["HAND_JOINT_COUNT"])
)
