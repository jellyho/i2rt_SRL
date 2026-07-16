"""OpenPI adapter tests that do not import or load the heavy OpenPI runtime."""

from __future__ import annotations

import numpy as np
import pytest
from yam_policy.policies.openpi_policy import OpenPiPolicy
from yam_policy.yam_contract import DEFAULT_IMAGE_KEYS


def _wire_obs() -> dict:
    return {
        "observation/state": np.arange(14, dtype=np.float32),
        "observation/images/agentview": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/images/wrist_left": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/images/wrist_right": np.full((224, 224, 3), 2, dtype=np.uint8),
        "prompt": "Insert the USB-C plug into the USB-C port.",
    }


class _FakeOpenPiPolicy:
    def __init__(self, actions=None):
        self.input = None
        self.actions = np.zeros((50, 14), dtype=np.float32) if actions is None else actions

    def infer(self, obs):
        self.input = obs
        return {"actions": self.actions, "policy_timing": {"infer_ms": 12.5}}


def _adapter(fake: _FakeOpenPiPolicy) -> OpenPiPolicy:
    adapter = OpenPiPolicy.__new__(OpenPiPolicy)
    adapter._policy = fake
    adapter.action_horizon = 16
    adapter.obs_spec = {
        "image_keys": dict(DEFAULT_IMAGE_KEYS),
        "image_size": 224,
    }
    return adapter


def test_openpi_policy_translates_wire_observation_and_preserves_timing():
    fake = _FakeOpenPiPolicy()
    response = _adapter(fake).infer(_wire_obs())

    assert set(fake.input) == {"state", "images", "prompt"}
    assert fake.input["state"].shape == (14,)
    assert set(fake.input["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_array_equal(fake.input["images"]["cam_high"], _wire_obs()["observation/images/agentview"])
    assert response["actions"].shape == (50, 14)
    assert response["actions"].dtype == np.float32
    assert response["policy_timing"]["infer_ms"] == 12.5


@pytest.mark.parametrize(
    "mutate",
    [
        lambda obs: obs.pop("observation/images/wrist_left"),
        lambda obs: obs.update({"observation/state": np.zeros(13, dtype=np.float32)}),
        lambda obs: obs.update({"observation/state": np.full(14, np.nan, dtype=np.float32)}),
        lambda obs: obs.update({"observation/images/agentview": np.zeros((3, 224, 224), dtype=np.uint8)}),
    ],
)
def test_openpi_policy_rejects_malformed_observations(mutate):
    obs = _wire_obs()
    mutate(obs)
    with pytest.raises(ValueError):
        _adapter(_FakeOpenPiPolicy()).infer(obs)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((50, 13), dtype=np.float32),
        np.zeros((15, 14), dtype=np.float32),
        np.full((50, 14), np.nan, dtype=np.float32),
    ],
)
def test_openpi_policy_rejects_invalid_model_output(actions):
    with pytest.raises(ValueError):
        _adapter(_FakeOpenPiPolicy(actions)).infer(_wire_obs())
