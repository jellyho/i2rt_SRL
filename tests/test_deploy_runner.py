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


def test_deploy_runner_resets_chunks_across_home_and_restart():
    runner = DeploymentPolicyRunner(
        BridgeConfig(rate_hz=100_000.0),
        RecorderConfig(mock=False),
        lambda: {},
    )

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.infer_calls = 0

        def reset(self):
            self.reset_calls += 1

        def infer(self, obs):
            self.infer_calls += 1
            return {"actions": np.zeros(14, dtype=np.float32)}

    class FakeRobot:
        def __init__(self):
            self.states = [
                {"policy_running": True, "homing": False},
                {"policy_running": False, "homing": True},
                {"policy_running": True, "homing": False},
            ]
            self.actions = []

        def get_observation(self):
            state = self.states.pop(0)
            if not self.states:
                runner._stop.set()
            return state

        def set_policy_action(self, action):
            self.actions.append(action)

    policy = FakePolicy()
    robot = FakeRobot()
    runner._policy = policy
    runner._robot = robot
    runner._build_obs = lambda robot_obs, images: {"observation/state": np.zeros(14)}

    runner._loop()

    assert policy.infer_calls == 2
    assert policy.reset_calls == 3  # initial stream, Home entry, post-Home restart
    assert len(robot.actions) == 2
