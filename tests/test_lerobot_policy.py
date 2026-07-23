"""Lightweight contract tests for the LeRobot/ACT server adapter."""

from __future__ import annotations

from types import SimpleNamespace

from yam_policy.policies.lerobot_policy import LeRobotPolicy


def test_lerobot_act_observation_spec_can_be_extended_with_yam_contract():
    config = SimpleNamespace(
        image_features={
            "observation.images.agentview": SimpleNamespace(shape=(3, 224, 224)),
            "observation.images.wrist_left": SimpleNamespace(shape=(3, 224, 224)),
            "observation.images.wrist_right": SimpleNamespace(shape=(3, 224, 224)),
        }
    )

    spec = LeRobotPolicy._obs_spec(config)

    assert spec["image_shape"] == [224, 224]
    assert set(spec["image_keys"]) == {"agentview", "wrist_left", "wrist_right"}


def test_lerobot_act_advertises_same_versioned_contract_as_openpi():
    config = SimpleNamespace(chunk_size=100, n_action_steps=16)

    spec = LeRobotPolicy._contract_spec(config, execution_horizon=4, control_hz=30)

    assert spec == {
        "contract": "yam_bimanual_v1",
        "action_dim": 14,
        "action_horizon": 4,
        "model_action_horizon": 100,
        "control_hz": 30.0,
    }
