"""Deployment policy runner observation assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.deploy_main import build_configs
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner


class _SharedRobot:
    connected = True

    def get_observation(self):
        return {}

    def set_policy_action(self, _action):
        pass


class _WarmupClient:
    def __init__(self, response):
        self.response = response
        self.timeout = None

    def infer(self, _obs, *, timeout=None):
        self.timeout = timeout
        return self.response


class _ResettablePolicy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


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


def test_deploy_runner_reuses_recorder_robot_connection():
    shared = _SharedRobot()
    runner = DeploymentPolicyRunner(
        BridgeConfig(),
        RecorderConfig(mock=False),
        lambda: {},
        robot_io=shared,
    )

    runner._connect_robot()

    assert runner._robot is shared
    assert runner.get_status()["robot_connected"] is True


def test_deploy_runner_rewind_invalidates_cached_and_prefetched_actions():
    runner = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=False), lambda: {})
    policy = _ResettablePolicy()
    runner._policy = policy
    runner._was_streaming = True

    runner._pause_streaming({"policy_running": True, "rewinding": True})

    assert policy.resets == 1
    assert runner._was_streaming is False
    assert runner.get_status()["streaming"] is False
    assert runner.get_status()["rollout_state"] == "REWINDING"


@pytest.mark.parametrize(
    ("adapter", "response"),
    [
        ("lerobot_act", {"actions": np.zeros((4, 14), np.float32)}),
        (
            "openpi_pi05",
            {
                "actions": np.zeros((50, 14), np.float32),
                "policy_timing": {"infer_ms": 2500.0},
            },
        ),
    ],
)
def test_deploy_runner_warmup_uses_shared_action_contract(adapter, response):
    runner = DeploymentPolicyRunner(
        BridgeConfig(execution_horizon=4, inference_timeout_s=2.0),
        RecorderConfig(mock=False),
        lambda: {},
    )
    client = _WarmupClient(response)
    runner._policy_client = client
    runner._status["execution_horizon"] = 4

    runner._warm_policy({"adapter": adapter})

    assert client.timeout == 30.0
    assert runner.get_status()["policy_ready"] is True


def test_deploy_defaults_to_gpu_codec_above_three_gib_free(monkeypatch):
    monkeypatch.setattr("workstation.lerobot_recorder.deploy_main._free_vram_mib", lambda: 3073)
    recorder_cfg, _ = build_configs(["--mock"])

    assert recorder_cfg.vcodec == "h264_nvenc"


@pytest.mark.parametrize("free_mib", [3072, 1024, None])
def test_deploy_defaults_to_cpu_codec_at_threshold_or_when_vram_unknown(monkeypatch, free_mib):
    monkeypatch.setattr("workstation.lerobot_recorder.deploy_main._free_vram_mib", lambda: free_mib)
    recorder_cfg, _ = build_configs(["--mock"])

    assert recorder_cfg.vcodec == "h264"


@pytest.mark.parametrize(
    ("flag", "codec", "free_mib"),
    [
        ("--codec", "h264", 8192),
        ("--codec", "h264_nvenc", 0),
        ("--vcodec", "libsvtav1", 8192),
        ("--codec", "auto", 0),
    ],
)
def test_deploy_respects_explicit_codec(monkeypatch, flag, codec, free_mib):
    monkeypatch.setattr("workstation.lerobot_recorder.deploy_main._free_vram_mib", lambda: free_mib)
    recorder_cfg, _ = build_configs(["--mock", flag, codec])

    assert recorder_cfg.vcodec == codec
