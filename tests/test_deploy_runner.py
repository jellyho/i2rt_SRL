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


# --------------------------------------------------------------------------------------- #
# Reporting whether the policy is reachable, before anything is running
# --------------------------------------------------------------------------------------- #
def _idle_runner(policy_port=59998):
    """A runner whose robot is up and idle: connected, but not running a rollout."""
    import types

    cfg = BridgeConfig(robot_host="127.0.0.1", robot_port=59999,
                       policy_host="127.0.0.1", policy_port=policy_port, rate_hz=50)
    rec = RecorderConfig(mock=False)
    runner = DeploymentPolicyRunner(cfg, rec, lambda: {})
    runner._connect_robot = lambda: runner._set(robot_connected=True)
    runner._robot = types.SimpleNamespace(get_observation=lambda: {
        "policy_running": False, "intervention": False, "homing": False, "estop": False})
    return runner


def test_idle_status_says_connecting_not_nothing():
    """A red dot with an empty reason reads as 'fine', when it means 'not tried yet'."""
    assert _idle_runner().get_status()["last_error"] == "connecting…"


def test_policy_state_is_reported_without_starting_a_rollout():
    """The reason a policy is unreachable has to show up while idle.

    The connection used to be attempted only inside the streaming branch, which needs a
    rollout already running. So before starting one the UI showed a red dot next to
    "policy idle" with no message at all -- and since the GUI refuses to start a rollout
    unless the policy is connected, the only path to connected ran through a rollout that
    could not be started.
    """
    import time

    runner = _idle_runner()
    runner.start()
    try:
        time.sleep(2.5)  # one probe period plus slack
        status = runner.get_status()
    finally:
        runner.shutdown()

    assert status["policy_connected"] is False
    assert status["streaming"] is False
    assert "59998" in status["last_error"], status["last_error"]


def test_idle_probe_connects_so_the_gui_guard_can_pass():
    import time

    runner = _idle_runner()
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()),
                                      runner._set(policy_connected=True))
    runner.start()
    try:
        time.sleep(1.0)
        status = runner.get_status()
    finally:
        runner.shutdown()

    assert status["policy_connected"] is True
    assert status["last_error"] == ""


def test_idle_probe_is_throttled():
    """The probe is a real websocket round-trip, and the loop runs at the control rate."""
    import time

    runner = _idle_runner()
    attempts = []

    def fail():
        attempts.append(time.monotonic())
        raise ConnectionError("offline")

    runner._connect_policy = fail
    runner.start()
    try:
        time.sleep(3.0)
    finally:
        runner.shutdown()

    # 50 Hz for 3 s is ~150 ticks; a 2 s probe period should be 1-3 attempts.
    assert 1 <= len(attempts) <= 3, len(attempts)


def _capture(runner, seconds, level="INFO"):
    """Run the loop for a while, returning the deploy_runner log lines it emitted."""
    import logging, time

    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(f"{record.levelname} {record.getMessage()}")

    log = logging.getLogger("workstation.policy_bridge.deploy_runner")
    handler, old = Grab(), log.level
    log.addHandler(handler)
    log.setLevel(level)
    try:
        runner.start()
        time.sleep(seconds)
    finally:
        runner.shutdown()
        log.removeHandler(handler)
        log.setLevel(old)
    return records


def test_client_says_it_cannot_reach_the_policy_at_startup():
    """Silence is the wrong answer to "did I connect?".

    Transitions alone cannot cover this: a policy that was never up never transitions, so a
    client started before its server would print nothing at all about the link.
    """
    lines = [x for x in _capture(_idle_runner(), 3.0) if "policy" in x]
    assert any("NOT CONNECTED" in x and "59998" in x for x in lines), lines
    # ...and does not repeat the same reason on every probe
    assert sum("NOT CONNECTED" in x for x in lines) == 1, lines


def test_client_announces_the_policy_link_coming_up():
    runner = _idle_runner()
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()),
                                      runner._set(policy_connected=True))
    lines = _capture(runner, 1.0)
    assert any("policy CONNECTED" in x and "59998" in x for x in lines), lines


def test_losing_the_policy_is_reported_once_not_twice():
    """The probe and the transition describe the same event; only one should speak."""
    import time

    runner = _idle_runner()
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()),
                                      runner._set(policy_connected=True))

    def drop():
        runner._policy = None
        runner._connect_policy = lambda: (_ for _ in ()).throw(ConnectionError("gone"))

    import threading
    threading.Timer(1.0, drop).start()
    lines = _capture(runner, 4.0)
    assert sum("DISCONNECTED" in x for x in lines) == 1, lines
    assert sum("NOT CONNECTED" in x for x in lines) == 0, lines
