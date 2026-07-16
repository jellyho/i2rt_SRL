"""Validation and translation for the versioned bimanual YAM policy contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

CONTRACT = "yam_bimanual_v1"
ACTION_DIM = 14
REQUIRED_IMAGE_ROLES = ("agentview", "wrist_left", "wrist_right")
DEFAULT_IMAGE_KEYS = {
    "agentview": "observation/images/agentview",
    "wrist_left": "observation/images/wrist_left",
    "wrist_right": "observation/images/wrist_right",
}


@dataclass(frozen=True)
class ContractSpec:
    contract: str
    action_dim: int
    execution_horizon: int
    model_action_horizon: int
    control_hz: float
    image_shape: tuple[int, int]
    image_keys: Dict[str, str]


def validate_observation(
    obs: Mapping,
    *,
    image_keys: Mapping[str, str] = DEFAULT_IMAGE_KEYS,
    image_shape: tuple[int, int] | None = None,
) -> None:
    state = np.asarray(obs.get("observation/state"))
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"observation/state must have shape ({ACTION_DIM},), got {state.shape}")
    if not np.issubdtype(state.dtype, np.number) or not np.all(np.isfinite(state)):
        raise ValueError("observation/state must contain only finite numeric values")

    missing_roles = [role for role in REQUIRED_IMAGE_ROLES if role not in image_keys]
    if missing_roles:
        raise ValueError(f"image_keys is missing required camera roles: {missing_roles}")
    for role in REQUIRED_IMAGE_ROLES:
        key = image_keys[role]
        if key not in obs:
            raise ValueError(f"Missing required camera {role!r} at observation key {key!r}")
        image = np.asarray(obs[key])
        expected = (*image_shape, 3) if image_shape is not None else None
        if image.ndim != 3 or image.shape[-1] != 3 or (expected is not None and image.shape != expected):
            suffix = f"; expected {expected}" if expected is not None else ""
            raise ValueError(f"Camera {role!r} must be an HWC RGB image, got {image.shape}{suffix}")
        if image.dtype != np.uint8:
            raise ValueError(f"Camera {role!r} must have dtype uint8, got {image.dtype}")

    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")


def to_openpi_input(
    obs: Mapping,
    *,
    image_keys: Mapping[str, str] = DEFAULT_IMAGE_KEYS,
    image_shape: tuple[int, int] | None = None,
) -> Dict:
    validate_observation(obs, image_keys=image_keys, image_shape=image_shape)
    return {
        "state": np.asarray(obs["observation/state"], dtype=np.float32),
        "images": {
            "cam_high": np.asarray(obs[image_keys["agentview"]]),
            "cam_left_wrist": np.asarray(obs[image_keys["wrist_left"]]),
            "cam_right_wrist": np.asarray(obs[image_keys["wrist_right"]]),
        },
        "prompt": obs["prompt"],
    }


def validate_action_chunk(response: Mapping, *, execution_horizon: int, action_dim: int = ACTION_DIM) -> Dict:
    if "actions" not in response:
        raise ValueError("Policy response is missing 'actions'")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Policy actions must be rank 2, got shape {actions.shape}")
    if actions.shape[1] != action_dim:
        raise ValueError(f"Policy actions must have final dimension {action_dim}, got shape {actions.shape}")
    if actions.shape[0] < execution_horizon:
        raise ValueError(
            f"Policy chunk has {actions.shape[0]} steps, shorter than execution horizon {execution_horizon}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy actions contain NaN or infinity")
    validated = dict(response)
    validated["actions"] = actions
    return validated


def validate_action_step(action: object, *, action_dim: int = ACTION_DIM) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32)
    if result.shape != (action_dim,):
        raise ValueError(f"Policy action step must have shape ({action_dim},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Policy action step contains NaN or infinity")
    return result


def validate_server_metadata(
    metadata: Mapping,
    *,
    configured_contract: str = CONTRACT,
    configured_execution_horizon: int | None = None,
    configured_control_hz: float | None = None,
    allow_legacy: bool = False,
) -> ContractSpec:
    required = {
        "contract",
        "action_dim",
        "action_horizon",
        "model_action_horizon",
        "control_hz",
        "image_keys",
    }
    missing = sorted(required - set(metadata))
    if missing and not allow_legacy:
        raise ValueError(f"Policy server metadata is missing required fields: {missing}")

    contract = str(metadata.get("contract", configured_contract))
    if contract != configured_contract or contract != CONTRACT:
        raise ValueError(f"Unsupported policy contract {contract!r}; expected {configured_contract!r}")
    action_dim = int(metadata.get("action_dim", ACTION_DIM))
    if action_dim != ACTION_DIM:
        raise ValueError(f"Policy action_dim must be {ACTION_DIM}, got {action_dim}")
    execution_horizon = int(metadata.get("action_horizon", configured_execution_horizon or 0))
    model_horizon = int(metadata.get("model_action_horizon", execution_horizon))
    if execution_horizon <= 0 or model_horizon <= 0 or execution_horizon > model_horizon:
        raise ValueError(f"Invalid execution/model horizons: execution={execution_horizon}, model={model_horizon}")
    if configured_execution_horizon is not None and execution_horizon != configured_execution_horizon:
        raise ValueError(
            f"Policy execution horizon {execution_horizon} disagrees with configured {configured_execution_horizon}"
        )
    control_hz = float(metadata.get("control_hz", configured_control_hz or 0))
    if control_hz <= 0:
        raise ValueError(f"Policy control_hz must be positive, got {control_hz}")
    if configured_control_hz is not None and not np.isclose(control_hz, configured_control_hz):
        raise ValueError(f"Policy control_hz {control_hz:g} disagrees with configured {configured_control_hz:g}")

    image_keys = dict(metadata.get("image_keys", DEFAULT_IMAGE_KEYS))
    missing_roles = sorted(set(REQUIRED_IMAGE_ROLES) - set(image_keys))
    if missing_roles:
        raise ValueError(f"Policy metadata is missing required image roles: {missing_roles}")
    if "image_shape" in metadata:
        raw_shape = metadata["image_shape"]
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ValueError(f"Policy image_shape must contain [height, width], got {raw_shape!r}")
        image_shape = (int(raw_shape[0]), int(raw_shape[1]))
    else:
        image_size = int(metadata.get("image_size", 0))
        image_shape = (image_size, image_size)
    if min(image_shape) <= 0:
        raise ValueError(f"Policy image shape must be positive, got {image_shape}")
    return ContractSpec(
        contract=contract,
        action_dim=action_dim,
        execution_horizon=execution_horizon,
        model_action_horizon=model_horizon,
        control_hz=control_hz,
        image_shape=image_shape,
        image_keys=image_keys,
    )
