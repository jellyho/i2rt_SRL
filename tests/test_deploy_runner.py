"""Deployment policy runner — the observation it sends must match openpi's contract.

openpi is the standard for this link, and its policies (libero, RoboCasa, YAM) all read the
same slot names: one `observation/state` plus `observation/image` / `observation/wrist_image`
/ `observation/image_right`. Sending anything else is not an error anyone sees — the server
raises deep inside a transform, or normalizes against the wrong statistics.
"""

from __future__ import annotations

import logging

import numpy as np

from workstation.lerobot_recorder.config import CONTROL_MODE, EEF_DIM, LEADER_DIM, RecorderConfig
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


def test_sends_every_recorded_column_and_nothing_else():
    """Leader/eef/control_mode were once withheld as "recorder features, not policy inputs".
    That held for openpi, whose transforms ignore them, and broke for LeRobot: its trainer takes
    every dataset column as an input, so a checkpoint trained on this stack's data can require
    any of them -- and one that did could not be deployed at all while they were withheld.

    27 floats a tick. The set stays pinned, because the original point was that the client puts
    nothing arbitrary on the wire.
    """
    obs = _runner()._build_obs(_robot_obs(), {})
    assert set(obs) == {
        "observation/state",
        "observation.leader",
        "observation.eef",
        "observation.control_mode",
        "prompt",
    }
    assert obs["prompt"] == "pick up the banana cloth"
    assert obs["observation.leader"].shape == (LEADER_DIM,)
    assert obs["observation.eef"].shape == (EEF_DIM,)
    assert float(obs["observation.control_mode"][0]) == CONTROL_MODE["policy"]


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

    cfg = BridgeConfig(
        robot_host="127.0.0.1", robot_port=59999, policy_host="127.0.0.1", policy_port=policy_port, rate_hz=50
    )
    rec = RecorderConfig(mock=False)
    runner = DeploymentPolicyRunner(cfg, rec, lambda: {})
    runner._connect_robot = lambda: runner._set(robot_connected=True)
    runner._robot = types.SimpleNamespace(
        get_observation=lambda: {"policy_running": False, "intervention": False, "homing": False, "estop": False}
    )
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
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()), runner._set(policy_connected=True))
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
    import logging
    import time

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
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()), runner._set(policy_connected=True))
    lines = _capture(runner, 1.0)
    assert any("policy CONNECTED" in x and "59998" in x for x in lines), lines


def test_losing_the_policy_is_reported_once_not_twice():
    """The probe and the transition describe the same event; only one should speak."""

    runner = _idle_runner()
    runner._connect_policy = lambda: (setattr(runner, "_policy", object()), runner._set(policy_connected=True))

    def drop():
        runner._policy = None
        runner._connect_policy = lambda: (_ for _ in ()).throw(ConnectionError("gone"))

    import threading

    threading.Timer(1.0, drop).start()
    lines = _capture(runner, 4.0)
    assert sum("DISCONNECTED" in x for x in lines) == 1, lines
    assert sum("NOT CONNECTED" in x for x in lines) == 0, lines


# --------------------------------------------------------------------------------------- #
# Reachability probe
# --------------------------------------------------------------------------------------- #
def _http_server(status=200):
    """A throwaway HTTP server; returns (port, shutdown)."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"OK\n")

        def log_message(self, *_args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


def test_probe_asks_over_http_so_the_server_logs_no_handshake_failure():
    """The probe must not look like a broken client to a websocket server.

    Opening a bare TCP connection and closing it reports reachability correctly, but the
    policy server is a websocket server: a socket that says nothing is a failed opening
    handshake, and it logs one with a full traceback right before the real connection
    succeeds -- indistinguishable from a genuine fault. openpi answers /healthz for this.
    """
    port, shutdown = _http_server()
    try:
        assert _idle_runner(policy_port=port)._policy_port_open() is True
    finally:
        shutdown()


def test_probe_counts_any_http_answer_as_up():
    """A server without the /healthz route is still a server."""
    port, shutdown = _http_server(status=404)
    try:
        assert _idle_runner(policy_port=port)._policy_port_open() is True
    finally:
        shutdown()


def test_probe_reports_down_when_nothing_is_listening():
    import socket

    with socket.socket() as s:  # bind and close to get a port nothing holds
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert _idle_runner(policy_port=port)._policy_port_open() is False


# --------------------------------------------------------------------------------------- #
# Replay source: in-process dataset replay (no server), driven by the run-page episode pick
# --------------------------------------------------------------------------------------- #
def test_replay_mode_without_an_episode_is_a_benign_not_connected():
    """In replay mode the dataset+episode is chosen on the run page; until one is, connecting is a
    plain 'not connected' (the GUI blocks Start), NOT an attempt to reach a policy server."""
    import pytest

    r = DeploymentPolicyRunner(BridgeConfig(replay_mode=True), RecorderConfig(mock=False), lambda: {})
    with pytest.raises(ConnectionError, match="select an episode"):
        r._connect_policy()


def test_set_replay_source_updates_cfg_and_forces_a_rebuild():
    """Picking an episode points the in-process replay at it and drops the current policy so the
    idle probe rebuilds DatasetPolicy for the new episode -- no server to restart."""
    r = DeploymentPolicyRunner(BridgeConfig(replay_mode=True), RecorderConfig(mock=False), lambda: {})
    r._policy = object()
    r.set_replay_source("yam_cable_tie_v4", 3)
    assert r.cfg.replay_dataset == "yam_cable_tie_v4"
    assert r.cfg.replay_episode == 3
    assert r._policy is None  # dropped -> the probe rebuilds for the new episode


def test_set_replay_source_ignores_a_no_op_reselect():
    """Re-selecting the SAME episode (same dataset+episode, policy already built) does nothing --
    so the overlay auto-selecting the current episode cannot thrash the policy."""
    r = DeploymentPolicyRunner(BridgeConfig(replay_mode=True), RecorderConfig(mock=False), lambda: {})
    r.set_replay_source("d", 1)
    r._policy = object()  # pretend it built
    r.set_replay_source("d", 1)
    assert r._policy is not None  # unchanged: no rebuild forced


def test_replay_pause_is_a_workstation_send_gate():
    """Pause/resume never touches policy_running -- it is a flag the loop reads to stop/continue
    sending actions, so the robot just holds and its start/stop (gripper close) logic never runs."""
    r = DeploymentPolicyRunner(BridgeConfig(replay_mode=True), RecorderConfig(mock=False), lambda: {})
    assert r.replay_paused is False
    r.set_replay_paused(True)
    assert r.replay_paused is True
    r.set_replay_paused(False)
    assert r.replay_paused is False


def test_picking_a_new_episode_clears_pause():
    """A freshly picked episode plays from the start, so selecting one clears any pause."""
    r = DeploymentPolicyRunner(BridgeConfig(replay_mode=True), RecorderConfig(mock=False), lambda: {})
    r.set_replay_paused(True)
    r.set_replay_source("d", 5)
    assert r.replay_paused is False


def test_every_eval_frame_carries_the_provenance_of_the_action_it_recorded():
    """A rollout must say where each action came from, not just what it was.

    An action is the k-th step of a chunk inferred from an observation some ticks earlier, so
    without these columns a dataset cannot distinguish "the policy reacted to this frame" from
    "the policy was still replaying a plan made a second ago" -- and the send cadence is lost to
    the dataset's uniform frame timestamps.
    """
    r = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})

    class _Broker:
        def stats(self):
            return {"chunk_index": 3, "step_in_chunk": 7, "infer_ms": 120.5, "delay_ticks": 4}

    r._policy = _Broker()
    r._note_timing(1234.5)

    features = r.extra_features()
    for name in ("chunk_index", "step_in_chunk", "infer_ms", "delay_ticks", "wall_time"):
        assert features["policy." + name] == (1,)

    extras = r.get_extras()
    assert float(extras["policy.chunk_index"][0]) == 3.0
    assert float(extras["policy.step_in_chunk"][0]) == 7.0
    assert float(extras["policy.delay_ticks"][0]) == 4.0
    assert float(extras["policy.wall_time"][0]) == 1234.5
    st = r.get_status()
    assert st["delay_ticks"] == 4 and st["underruns"] == 0


def test_provenance_is_present_even_when_the_policy_declares_no_extras():
    """The columns come from the runner, not the handshake, so a plain policy still gets them."""
    r = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})
    assert r._extra_features == {}
    assert set(r.extra_features()) == {
        "policy.chunk_index",
        "policy.step_in_chunk",
        "policy.infer_ms",
        "policy.delay_ticks",
        "policy.wall_time",
    }
    assert all(v.shape == (1,) for v in r.get_extras().values())


def test_inference_is_synchronous_unless_asked_for():
    """Eval is the default use, and synchronous is the stricter thing to measure: every chunk is
    computed from the observation just handed over, so a rollout has no delay confound. Async
    (continuous motion, chunk starts `delay_ticks` late) is opt-in per run."""
    assert BridgeConfig().async_inference is False
    assert BridgeConfig(async_inference=True).async_inference is True


def test_the_length_of_every_reply_is_recorded_for_the_plot():
    """The horizon comes from each reply, not from a setting, so it can change per replan -- and
    the reply is the only place that number exists. One entry per NEW chunk, including a repeat of
    the same length; nothing while the same chunk is being consumed."""
    r = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})

    class _Broker:
        chunk_index = -1
        action_horizon = 0

    broker = _Broker()
    r._policy = broker
    r._note_chunk()
    assert r.chunk_lengths() == []  # nothing inferred yet

    for index, horizon in enumerate([30, 30, 12, 12, 7]):
        broker.chunk_index, broker.action_horizon = index, horizon
        r._note_chunk()
        r._note_chunk()  # same chunk consumed over many ticks -> still one entry
    assert r.chunk_lengths() == [30, 30, 12, 12, 7]


def test_chunk_history_is_bounded():
    from workstation.policy_bridge.deploy_runner import CHUNK_HISTORY

    r = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})

    class _Broker:
        chunk_index = -1
        action_horizon = 16

    broker = _Broker()
    r._policy = broker
    for index in range(CHUNK_HISTORY + 25):
        broker.chunk_index = index
        r._note_chunk()
    assert len(r.chunk_lengths()) == CHUNK_HISTORY


def test_critic_select_is_only_sent_when_asked_for():
    """Server-side selection is a per-request opt-in: the RLT critic reads a token that never
    leaves the model and the patch critic needs the base VLA's sampler, so the client cannot do it
    -- but a critic-backed server also does NOTHING unless the request carries the key."""
    plain = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})
    assert "critic_select" not in plain._build_obs(_robot_obs(), {})

    picky = DeploymentPolicyRunner(BridgeConfig(critic_select=True), RecorderConfig(mock=True), lambda: {})
    assert picky._build_obs(_robot_obs(), {})["critic_select"] is True


def test_a_server_without_a_critic_is_reported_not_silently_obeyed(caplog):
    """The failure this closes: critic_select against a plain server is an unknown key, so the run
    looks like a critic run and is not one."""
    r = DeploymentPolicyRunner(BridgeConfig(critic_select=True), RecorderConfig(mock=True), lambda: {})
    r._extra_features = {"action_samples": (8, 14)}  # samples but no critic_scores -> no critic
    with caplog.at_level(logging.WARNING):
        r._warn_critic_mismatch()
    assert "no critic" in caplog.text


def test_asking_for_a_different_n_than_the_server_declared_is_reported(caplog):
    """The columns are fixed at handshake, so a per-request N that disagrees makes every reply the
    wrong shape and the critic columns are dropped frame by frame."""
    r = DeploymentPolicyRunner(BridgeConfig(critic_select=True, num_samples=16), RecorderConfig(mock=True), lambda: {})
    r._extra_features = {"critic_scores": (8,), "action_samples": (8, 14)}
    with caplog.at_level(logging.WARNING):
        r._warn_critic_mismatch()
    assert "num_samples=16" in caplog.text and "N=8" in caplog.text

    quiet = DeploymentPolicyRunner(BridgeConfig(critic_select=True), RecorderConfig(mock=True), lambda: {})
    quiet._extra_features = {"critic_scores": (8,)}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        quiet._warn_critic_mismatch()
    assert caplog.text == ""  # num_samples left at 0 uses the server's own N -- nothing to warn about


def test_no_critic_warnings_when_the_feature_is_off(caplog):
    r = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: {})
    r._extra_features = {}
    with caplog.at_level(logging.WARNING):
        r._warn_critic_mismatch()
    assert caplog.text == ""
