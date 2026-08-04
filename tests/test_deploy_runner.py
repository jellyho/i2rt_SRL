"""Deployment policy runner — the observation it sends must match openpi's contract.

openpi is the standard for this link, and its policies (libero, RoboCasa, YAM) all read the
same slot names: one `observation/state` plus `observation/image` / `observation/wrist_image`
/ `observation/image_right`. Sending anything else is not an error anyone sees — the server
raises deep inside a transform, or normalizes against the wrong statistics.
"""

from __future__ import annotations

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner


def _runner(image_shape=(480, 640)):
    runner = DeploymentPolicyRunner(
        BridgeConfig(prompt="pick up the banana cloth"),
        RecorderConfig(mock=True),
        lambda: {},
    )
    runner._image_shape = runner._image_shape_from_meta({"image_shape": list(image_shape)})
    return runner


def _robot_obs():
    return {
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


def test_state_is_the_full_42_the_policy_was_trained_on():
    """The trap this guards: training repacks the dataset's 42-d `observation.state` into
    `observation/state`, so sending only the 14 joint positions keeps the key valid, raises
    nothing, and silently normalizes against the wrong statistics."""
    obs = _runner()._build_obs(_robot_obs(), {})

    assert obs["observation/state"].shape == (42,)
    assert obs["observation/state"].dtype == np.float32
    np.testing.assert_allclose(
        obs["observation/state"][:21],
        np.concatenate([_robot_obs()["left"][k] for k in ("pos", "vel", "eff")]),
    )


def test_uses_openpi_image_slot_names_by_default():
    """The default mapping has to be openpi's slots, not our dataset's key names."""
    runner = _runner()
    images = {r: np.ones((240, 320, 3), np.uint8) for r in ("agentview", "wrist_left", "wrist_right")}
    obs = runner._build_obs(_robot_obs(), images)

    assert set(runner.cfg.image_keys.values()) == {
        "observation/image",
        "observation/wrist_image",
        "observation/image_right",
    }
    for key in runner.cfg.image_keys.values():
        assert obs[key].shape == (480, 640, 3)
        assert obs[key].dtype == np.uint8


def test_sends_nothing_the_policy_does_not_read():
    """Leader/eef/control_mode are recorder features, not policy inputs; a YAM or RoboCasa
    transform ignores them, so they would only be wasted bandwidth every tick."""
    obs = _runner()._build_obs(_robot_obs(), {})
    assert set(obs) == {"observation/state", "prompt"}
    assert obs["prompt"] == "pick up the banana cloth"


def test_missing_arm_state_yields_no_observation():
    """A half-populated robot snapshot must not be sent as a partial observation."""
    partial = _robot_obs()
    partial["right"].pop("vel")
    assert _runner()._build_obs(partial, {}) == {}


def test_image_keys_from_server_metadata_win():
    """A policy that names its slots differently is followed, not overridden."""
    runner = _runner()
    runner.cfg.image_keys = {"agentview": "observation/exterior_image_1_left"}
    obs = runner._build_obs(_robot_obs(), {"agentview": np.ones((240, 320, 3), np.uint8)})
    assert "observation/exterior_image_1_left" in obs
