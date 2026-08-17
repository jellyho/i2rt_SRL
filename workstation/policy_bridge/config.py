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

    # Overlap inference with execution (AsyncChunkBroker): the next chunk is inferred on a
    # background thread while the current one is still being executed, so the control loop keeps
    # sending an action every tick instead of freezing for a round-trip at every chunk boundary --
    # which is what made the arm dead-stop about once a second in every rollout. The reply is
    # spliced in at the index matching the inference delay, so nothing is executed twice.
    # Turn off to get the old synchronous behaviour (one stall per chunk) for comparison.
    async_inference: bool = True
    # Steps of chunk that must remain before the next inference is started. 0 = derive it from the
    # measured latency (recommended: it adapts to the server and the network on its own).
    prefetch_ticks: int = 0
    # Extra ticks of headroom on top of the measured latency when prefetch_ticks is 0.
    prefetch_margin_ticks: int = 2

    # Replay source: when `replay_mode` is on, the deploy stack drives the robot from a recorded
    # dataset instead of a live policy server -- an in-process DatasetPolicy (no server, no
    # websocket, no subprocess) wrapped exactly like any policy, so every deployment safeguard
    # (smoother, e-stop, takeover, overlay) still applies. The specific `replay_dataset` (folder
    # name under the recorder root) + `replay_episode` are chosen on the run page's reference panel
    # and pushed in via DeploymentPolicyRunner.set_replay_source; until one is chosen the policy
    # simply is not connected ("select an episode to replay"). `replay_mode` on with no dataset
    # yet is the waiting state; `replay_mode` off is a live policy.
    replay_mode: bool = False
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
