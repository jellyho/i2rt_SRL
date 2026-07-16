"""Configuration shared by the headless bridge and deployment UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BridgeConfig:
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    contract: str = "yam_bimanual_v1"
    execution_horizon: int = 16
    rate_hz: float = 30.0
    image_size: int = 224
    prompt: str = "do the task"
    use_async: bool = True
    require_operator_arm: bool = True
    arm_on_start: bool = False
    allow_legacy_metadata: bool = False
    camera_max_age_s: float = 0.25
    inference_timeout_s: float = 2.0
    action_limits: Optional[List[tuple[float, float]]] = None
    image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "agentview": "observation/images/agentview",
            "wrist_left": "observation/images/wrist_left",
            "wrist_right": "observation/images/wrist_right",
        }
    )
