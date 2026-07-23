"""Portal + websocket + simulated DAggerController policy loopback."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("portal")
pytest.importorskip("msgpack_numpy")

from yam_policy import WebsocketPolicyServer
from yam_policy.policies.dummy import DummyPolicy

from i2rt.serving.controllers import DaggerConfig, DaggerController
from i2rt.serving.robot_client import RobotClient
from i2rt.serving.robot_server import RobotServer
from tests._util import free_port, wait_port
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.policy_bridge.bridge import PolicyBridge
from workstation.policy_bridge.config import BridgeConfig


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition did not become true before timeout")


def test_portal_websocket_dagger_loopback_and_fail_closed_estop():
    robot_port = free_port()
    policy_port = free_port()

    controller = DaggerController(DaggerConfig(sim=True, rate=120.0, command_timeout=0.25))
    robot_server = RobotServer(controller, port=robot_port, rate_hz=120.0)
    threading.Thread(target=robot_server.serve, daemon=True).start()
    assert wait_port(robot_port)

    dummy = DummyPolicy(action_horizon=4, mode="hold")
    metadata = {**dummy.obs_spec, "action_horizon": 4}
    policy_server = WebsocketPolicyServer(dummy, host="127.0.0.1", port=policy_port, metadata=metadata)
    threading.Thread(target=policy_server.serve_forever, daemon=True).start()
    assert wait_port(policy_port)

    bridge = PolicyBridge(
        BridgeConfig(
            robot_port=robot_port,
            policy_port=policy_port,
            execution_horizon=4,
            rate_hz=30,
            prompt="Insert the USB-C plug into the USB-C port.",
            use_async=False,
            arm_on_start=True,
        ),
        RecorderConfig(mock=True),
    )
    bridge_thread = threading.Thread(target=bridge.run, daemon=True)
    bridge_thread.start()
    robot = RobotClient("127.0.0.1", robot_port, timeout=1.0)

    def policy_ready():
        obs = robot.get_observation()
        return obs if obs.get("policy_running") and obs.get("left", {}).get("applied") is not None else None

    policy_obs = _wait_for(policy_ready)
    assert policy_obs["dagger_state"] == "policy"

    robot.set_intervention(True)

    def intervening():
        obs = robot.get_observation()
        return obs if obs.get("intervention") else None

    intervention_obs = _wait_for(intervening)
    assert intervention_obs["dagger_state"] == "intervention"

    robot.set_intervention(False)

    def resumed():
        obs = robot.get_observation()
        return obs if obs.get("dagger_state") == "policy" and not obs.get("intervention") else None

    resumed_obs = _wait_for(resumed)
    assert bool(resumed_obs["policy_running"])

    robot.set_estop(True)

    def estopped():
        obs = robot.get_observation()
        return obs if obs.get("estop") else None

    _wait_for(estopped)
    bridge_thread.join(timeout=3.0)
    assert not bridge_thread.is_alive()
    assert not bool(robot.get_observation()["policy_running"])
    robot_server.close()
