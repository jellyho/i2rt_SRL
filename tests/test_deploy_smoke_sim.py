"""End-to-end deployment smoke test in sim — robot server + policy server + deploy runner.

Same shape as the other sim smoke tests: a real portal robot server on a free port, a real
websocket policy server on another, and the actual `DeploymentPolicyRunner` in between. No
hardware, no mocks of the parts under test — the point is that the three processes agree
about the wire, which is exactly where this stack has failed before.

The sim robot is a mujoco `DeployController`, so nothing here can move a physical arm.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("portal")
pytest.importorskip("mujoco")  # sim robot
pytest.importorskip("websockets")

from yam_policy import WebsocketPolicyServer

from i2rt.serving.controllers import DeployConfig, DeployController
from i2rt.serving.robot_server import RobotServer
from tests._util import free_port, wait_port
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner

CHUNK = 30  # what the YAM configs actually train with; deliberately != any client default
ACTION_DIM = 14


class ChunkPolicy:
    """Returns a fixed-size chunk and records the observations it was given."""

    def __init__(self, chunk: int = CHUNK) -> None:
        self.chunk = chunk
        self.seen: list[dict] = []
        self._lock = threading.Lock()

    def infer(self, obs):
        with self._lock:
            self.seen.append({k: v for k, v in obs.items()})
        return {"actions": np.zeros((self.chunk, ACTION_DIM), dtype=np.float32)}

    def reset(self):
        pass


@pytest.fixture
def policy_server():
    policy = ChunkPolicy()
    port = free_port()
    srv = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"
    return policy, port


@pytest.fixture
def robot_server():
    ctrl = DeployController(DeployConfig(sim=True))
    port = free_port()
    srv = RobotServer(ctrl, port=port, rate_hz=60.0)
    threading.Thread(target=srv.serve, daemon=True).start()
    assert wait_port(port), "robot server did not start"
    yield ctrl, port
    ctrl.close()


def _runner(robot_port, policy_port, images_fn):
    cfg = BridgeConfig(
        robot_host="127.0.0.1",
        robot_port=robot_port,
        policy_host="127.0.0.1",
        policy_port=policy_port,
        rate_hz=60.0,
        prompt="assemble lego blocks to make yellow taxi",
    )
    return DeploymentPolicyRunner(cfg, RecorderConfig(), images_fn)


def _images():
    return {r: np.zeros((480, 640, 3), np.uint8) for r in ("agentview", "wrist_left", "wrist_right")}


def _wait_until(pred, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_policy_actions_reach_the_sim_robot(policy_server, robot_server):
    """The whole path: robot obs -> openpi observation -> action chunk -> follower command."""
    policy, policy_port = policy_server
    ctrl, robot_port = robot_server
    runner = _runner(robot_port, policy_port, _images)
    runner.start()
    try:
        ctrl.set_policy_running(True)
        assert _wait_until(lambda: runner.get_status().get("streaming")), (
            f"never streamed: {runner.get_status().get('last_error')}"
        )
        assert _wait_until(lambda: len(policy.seen) > 0), "policy was never queried"
    finally:
        runner.shutdown()
        ctrl.set_policy_running(False)

    assert runner.get_status()["last_error"] == ""


def test_the_observation_the_policy_receives_is_openpi_shaped(policy_server, robot_server):
    """Guards the two silent failures: arrays arriving as raw dicts (msgpack mismatch), and
    a state vector that is the right key but the wrong length."""
    policy, policy_port = policy_server
    ctrl, robot_port = robot_server
    runner = _runner(robot_port, policy_port, _images)
    runner.start()
    try:
        ctrl.set_policy_running(True)
        assert _wait_until(lambda: len(policy.seen) > 0), "policy was never queried"
        obs = policy.seen[0]
    finally:
        runner.shutdown()
        ctrl.set_policy_running(False)

    # msgpack survived the round trip as real arrays, not {b'nd': ...} dicts
    assert isinstance(obs["observation/state"], np.ndarray), obs["observation/state"]
    assert obs["observation/state"].shape == (42,)

    for key in ("observation/image", "observation/wrist_image", "observation/image_right"):
        assert isinstance(obs[key], np.ndarray), key
        assert obs[key].shape == (224, 224, 3)
        assert obs[key].dtype == np.uint8

    assert obs["prompt"] == "assemble lego blocks to make yellow taxi"

    # The dotted keys alongside these are the recorder's other columns, and they are sent on
    # purpose: LeRobot's trainer takes every dataset column as a policy input, so a checkpoint
    # trained on this stack's data may require any of them, and one that does could not be
    # deployed at all when they were withheld. openpi never reads them.
    #
    # Pinned as an exact set rather than allowed wholesale -- the original point of this line was
    # that the client does not put arbitrary things on the wire, and that still holds.
    assert {k for k in obs if k.startswith("observation.")} == {
        "observation.leader",
        "observation.eef",
        "observation.control_mode",
    }, f"unexpected keys sent: {sorted(obs)}"


def test_the_chunk_size_comes_from_the_policy(policy_server, robot_server):
    """A 30-step chunk must be executed as 30 steps, with nothing configured to say so —
    the client has no action_horizon setting at all any more."""
    policy, policy_port = policy_server
    ctrl, robot_port = robot_server
    runner = _runner(robot_port, policy_port, _images)
    runner.start()
    try:
        ctrl.set_policy_running(True)
        assert _wait_until(lambda: runner.get_status().get("action_horizon") == CHUNK), (
            f"observed chunk {runner.get_status().get('action_horizon')}, expected {CHUNK}"
        )
        # ~2 chunks' worth of ticks at 60 Hz should be nowhere near 2*CHUNK inferences
        n_before = len(policy.seen)
        time.sleep(2.0 * CHUNK / 60.0)
        n_after = len(policy.seen)
    finally:
        runner.shutdown()
        ctrl.set_policy_running(False)

    queried = n_after - n_before
    assert queried <= 6, f"re-inferred {queried} times over ~2 chunks — chunk not being used"


def test_leader_mirror_rpc_reaches_the_controller(robot_server):
    """The deploy UI toggles this live; it has to actually land on the robot."""
    from i2rt.serving.robot_client import RobotClient

    ctrl, robot_port = robot_server
    client = RobotClient(host="127.0.0.1", port=robot_port, timeout=3.0)
    assert ctrl.snapshot()["leader_mirror"] is True

    client.set_leader_mirror(False)
    assert _wait_until(lambda: ctrl.snapshot()["leader_mirror"] is False)
    client.set_leader_mirror(True)
    assert _wait_until(lambda: ctrl.snapshot()["leader_mirror"] is True)


def test_robot_reports_the_deploy_mode(robot_server):
    """What the workstation's startup check reads to refuse a crossed server."""
    ctrl, _ = robot_server
    assert ctrl.snapshot()["mode"] == "deploy"


def test_no_snapshot_field_is_an_empty_string(robot_server):
    """portal's SendBuffer asserts every buffer is non-empty, so an empty string in the snapshot
    kills the RPC server thread -- the client simply stops receiving, with the real cause buried in
    a server-side traceback. Send None instead, as last_dagger_event already does."""
    ctrl, _port = robot_server
    ctrl.step()
    snap = ctrl.snapshot()

    def empties(obj, path="snapshot"):
        if isinstance(obj, str):
            return [] if obj else [path]
        if isinstance(obj, dict):
            return [p for k, v in obj.items() for p in empties(v, f"{path}.{k}")]
        if isinstance(obj, (list, tuple)):
            return [p for i, v in enumerate(obj) for p in empties(v, f"{path}[{i}]")]
        return []

    assert empties(snap) == []
