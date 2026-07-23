"""Build the model-agnostic ``yam_bimanual_v1`` workstation observation."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from yam_policy import image_tools
from yam_policy.yam_contract import ACTION_DIM, validate_observation

from workstation.lerobot_recorder.config import ARM_DOF, ARMS


def build_observation(
    robot_obs: Mapping,
    images: Mapping[str, np.ndarray],
    *,
    prompt: str,
    image_keys: Mapping[str, str],
    image_shape: tuple[int, int],
) -> Dict:
    positions = []
    for arm in ARMS:
        side = robot_obs.get(arm)
        if not side or side.get("pos") is None:
            raise ValueError(f"Robot observation is missing {arm}.pos")
        position = np.asarray(side["pos"], dtype=np.float32).reshape(-1)
        if position.shape != (ARM_DOF,):
            raise ValueError(f"Robot {arm}.pos must have shape ({ARM_DOF},), got {position.shape}")
        positions.append(position)
    state = np.concatenate(positions).astype(np.float32)
    if state.shape != (ACTION_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError("Bimanual robot state must contain 14 finite positions")

    obs: Dict = {"observation/state": state, "prompt": prompt}
    height, width = image_shape
    for role, key in image_keys.items():
        if role not in images:
            raise ValueError(f"Required camera role {role!r} has no current frame")
        image = image_tools.resize_with_pad(images[role], height, width)
        obs[key] = image_tools.convert_to_uint8(image)
    validate_observation(obs, image_keys=image_keys, image_shape=image_shape)
    return obs
