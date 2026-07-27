"""Deployment policy runner observation assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")

from workstation.lerobot_recorder.config import CONTROL_MODE, RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner


def test_deploy_runner_builds_lerobot_observation_schema():
    runner = DeploymentPolicyRunner(
        BridgeConfig(prompt="pick up the banana cloth"),
        RecorderConfig(mock=True),
        lambda: {},
    )
    runner.cfg.image_keys = {"agentview": "observation.images.agentview"}
    runner._image_shape = runner._image_shape_from_meta({"image_shape": [480, 640]})

    robot_obs = {
        "left": {
            "pos": np.arange(0, 7, dtype=np.float32),
            "vel": np.arange(10, 17, dtype=np.float32),
            "eff": np.arange(20, 27, dtype=np.float32),
            "leader_pos": np.arange(30, 36, dtype=np.float32),
            "eef": np.arange(40, 47, dtype=np.float32),
        },
        "right": {
            "pos": np.arange(50, 57, dtype=np.float32),
            "vel": np.arange(60, 67, dtype=np.float32),
            "eff": np.arange(70, 77, dtype=np.float32),
            "leader_pos": np.arange(80, 86, dtype=np.float32),
            "eef": np.arange(90, 97, dtype=np.float32),
        },
    }
    images = {"agentview": np.ones((240, 320, 3), dtype=np.uint8)}

    obs = runner._build_obs(robot_obs, images)

    assert obs["observation/state"].shape == (14,)
    assert obs["observation.state"].shape == (42,)
    assert obs["observation.leader"].shape == (12,)
    assert obs["observation.eef"].shape == (14,)
    assert obs["observation.control_mode"].tolist() == [CONTROL_MODE["teleop"]]
    assert obs["observation.images.agentview"].shape == (480, 640, 3)
    assert np.allclose(obs["observation.state"][:21], np.concatenate([robot_obs["left"][k] for k in ("pos", "vel", "eff")]))


@pytest.mark.parametrize("blocked_key", ["intervention", "returning", "homing", "estop"])
def test_deploy_runner_pauses_policy_for_robot_control_transitions(blocked_key):
    obs = {"policy_running": True, blocked_key: True}
    assert DeploymentPolicyRunner._should_stream(obs) is False


def test_deploy_runner_streams_during_unblocked_policy_state():
    assert DeploymentPolicyRunner._should_stream({"policy_running": True}) is True
