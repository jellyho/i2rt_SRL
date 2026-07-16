"""Deployment policy runner observation assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner


def test_deploy_runner_builds_yam_bimanual_v1_observation_schema():
    runner = DeploymentPolicyRunner(
        BridgeConfig(prompt="pick up the banana cloth"),
        RecorderConfig(mock=True),
        lambda: {},
    )
    runner._image_shape = (224, 224)

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
    images = {role: np.ones((240, 320, 3), dtype=np.uint8) for role in ("agentview", "wrist_left", "wrist_right")}

    obs = runner._build_obs(robot_obs, images)

    assert set(obs) == {
        "observation/state",
        "observation/images/agentview",
        "observation/images/wrist_left",
        "observation/images/wrist_right",
        "prompt",
    }
    assert obs["observation/state"].shape == (14,)
    assert obs["observation/state"].dtype == np.float32
    assert obs["observation/images/agentview"].shape == (224, 224, 3)
    assert obs["observation/images/agentview"].dtype == np.uint8
    assert np.allclose(
        obs["observation/state"],
        np.concatenate([robot_obs["left"]["pos"], robot_obs["right"]["pos"]]),
    )
