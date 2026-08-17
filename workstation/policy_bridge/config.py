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
    rate_hz: float = 30.0
    image_size: int = 224
    prompt: str = "do the task"

    # Camera role -> the observation key the policy reads. These are openpi's slot names,
    # shared by every multi-camera policy there (libero uses the first two; RoboCasa and
    # YAM all three), so one client drives any of them without a per-robot branch.
    #
    # The chunk size is deliberately NOT here: the broker takes it from what the policy
    # returns, so there is no number to keep in sync with the checkpoint.
    # How many action chunks to ask the policy for per observation. 0/1 is normal deployment:
    # one chunk, one forward pass, today's latency. Above that the server returns the extra
    # draws under `action_samples` and the viewer can show the spread -- which costs N forward
    # passes, so it is a deliberate inspection mode rather than something a rollout leaves on.
    num_samples: int = 0

    # Replay source: when `replay_dataset` is set, the deploy stack drives the robot from a
    # recorded dataset instead of a live policy server -- an in-process DatasetPolicy (no server,
    # no websocket, no subprocess) wrapped exactly like any policy, so every deployment safeguard
    # (smoother, e-stop, takeover, overlay) still applies. `replay_dataset` is the folder name
    # under the recorder root; `replay_episode` which episode; `replay_speed`/`replay_loop` how.
    replay_dataset: str = ""
    replay_episode: int = 0
    replay_speed: float = 1.0
    replay_loop: bool = False

    image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "agentview": "observation/image",
            "wrist_left": "observation/wrist_image",
            "wrist_right": "observation/image_right",
        }
    )
