"""Configuration shared by the headless bridge and deployment UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BridgeConfig:
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    action_horizon: int = 16
    rate_hz: float = 30.0
    image_size: int = 224
    prompt: str = "do the task"
    use_async: bool = True
    prefetch_steps: int = 2
    rtc_enabled: bool = False
    rtc_min_execute_steps: int = 8
    image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "agentview": "observation/images/agentview",
            "wrist_left": "observation/images/wrist_left",
            "wrist_right": "observation/images/wrist_right",
        }
    )

    def prefetch_at(self, action_horizon: int) -> int:
        """Convert the workstation-facing lead into the broker's launch index."""
        horizon = int(action_horizon)
        lead = int(self.prefetch_steps)
        if not 0 < lead < horizon:
            raise ValueError(
                f"prefetch_steps must be between 1 and action_horizon - 1; got {lead} for horizon {horizon}"
            )
        return horizon - lead
