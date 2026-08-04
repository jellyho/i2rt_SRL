"""Robot serving (portal) round-trip, e-stop, and controller-step tests (sim)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("portal")
pytest.importorskip("mujoco")  # sim robot

from i2rt.serving.controllers import (
    DaggerConfig,
    DaggerController,
    TeleopConfig,
    TeleopController,
    WrapperConfig,
    WrapperController,
)
from i2rt.serving.robot_client import RobotClient
from i2rt.serving.robot_server import RobotServer
from tests._util import free_port, wait_port


def test_controllers_step_sim():
    tc = TeleopController(TeleopConfig(sim=True))
    for _ in range(3):
        tc.step()
    snap = tc.snapshot()
    assert snap["teleop_state"] in ("HOMING", "IDLE", "ENGAGED")
    assert len(snap["left"]["pos"]) == 7
    tc.close()

    dc = DaggerController(DaggerConfig(sim=True))
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
    for _ in range(3):
        dc.step()
    ds = dc.snapshot()
    assert ds["intervention"] is False
    assert len(ds["left"]["applied"]) == 7  # policy drives the follower when not intervening
    dc.close()


def test_dagger_policy_intervention_and_home_states():
    dc = DaggerController(DaggerConfig(sim=True, max_joint_speed=10.0, home_speed=10.0))
    dc.step()
    assert dc.snapshot()["dagger_state"] == "stopped"

    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
    dc.step()
    snap = dc.snapshot()
    assert snap["dagger_state"] == "policy"
    assert snap["policy_running"] is True
    assert snap["left"]["applied"] is not None

    dc.set_intervention(True)
    dc.step()
    snap = dc.snapshot()
    assert snap["dagger_state"] == "intervention"
    assert snap["intervention"] is True

    dc.finish_dagger_run("keep")
    for _ in range(60):
        dc.step()
        if dc.snapshot()["dagger_state"] == "stopped":
            break
    snap = dc.snapshot()
    assert snap["dagger_state"] == "stopped"
    assert snap["policy_running"] is False
    assert snap["intervention"] is False
    assert snap["last_dagger_event"]["action"] == "keep"
    dc.close()


def test_dagger_relative_return_masks_zero_joints_and_hands_off_to_human():
    relative = [0.0] * 14
    relative[1] = 0.2
    relative[8] = -0.15
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            max_joint_speed=100.0,
            home_speed=1.0,
            command_timeout=10.0,
            return_mode="relative",
            return_rel_pos=relative,
            return_abs_pos=[0.0] * 14,
        )
    )
    try:
        ctrl.set_policy_running(True)
        ctrl.set_policy_action({"left": np.full(7, 0.2), "right": np.full(7, 0.3)})
        ctrl.step()
        before_left = np.asarray(ctrl.snapshot()["left"]["pos"], dtype=float)
        before_right = np.asarray(ctrl.snapshot()["right"]["pos"], dtype=float)

        ctrl.return_to_dagger_pose()
        assert ctrl.snapshot()["returning"] is False  # RPC only latches the request
        ctrl.step()
        assert ctrl.snapshot()["dagger_state"] == "returning"

        for _ in range(40):
            ctrl.step()
            if ctrl.snapshot()["intervention"]:
                break
        snap = ctrl.snapshot()
        left = np.asarray(snap["left"]["pos"], dtype=float)
        right = np.asarray(snap["right"]["pos"], dtype=float)
        assert snap["returning"] is False
        assert snap["intervention"] is True
        assert snap["policy_running"] is True
        assert snap["dagger_state"] == "intervention"
        assert left[1] == pytest.approx(before_left[1] + 0.2, abs=1e-6)
        assert right[1] == pytest.approx(before_right[1] - 0.15, abs=1e-6)
        assert np.allclose(left[[0, 2, 3, 4, 5, 6]], before_left[[0, 2, 3, 4, 5, 6]])
        assert np.allclose(right[[0, 2, 3, 4, 5, 6]], before_right[[0, 2, 3, 4, 5, 6]])
    finally:
        ctrl.close()


def test_dagger_return_moves_follower_and_leader_at_half_home_speed_without_policy_actions():
    relative = [0.0] * 14
    relative[1] = 0.2
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            home_speed=1.0,
            return_mode="relative",
            return_rel_pos=relative,
        )
    )
    try:
        pair = ctrl.pairs["left"]
        n = pair.leader.num_dofs()
        pair.base_kp = np.ones(n)
        pair.base_kd = np.ones(n)
        pair.leader.update_kp_kd = lambda kp, kd: None
        follower_before = np.asarray(pair.follower.get_joint_pos(), dtype=float)
        leader_before = np.asarray(pair.leader.get_joint_pos(), dtype=float)

        # Rollout state is sufficient; a connected policy server/fresh action is
        # intentionally not required for a recovery return.
        ctrl.set_policy_running(True)
        ctrl.return_to_dagger_pose()
        ctrl.step()

        follower_step = np.asarray(pair.follower.get_joint_pos(), dtype=float)[1] - follower_before[1]
        leader_step = np.asarray(pair.leader.get_joint_pos(), dtype=float)[1] - leader_before[1]
        first_tick_limit = 1.0 * 0.5 / 20.0 * ctrl._ease_vel_scale(0.0)
        assert 0.0 < follower_step <= first_tick_limit + 1e-9
        assert leader_step == pytest.approx(follower_step)
        assert ctrl.snapshot()["returning"] is True
    finally:
        ctrl.close()


def test_dagger_return_hands_off_after_leader_alignment_timeout():
    relative = [0.0] * 14
    relative[1] = 0.2
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            home_speed=1.0,
            return_mode="relative",
            return_rel_pos=relative,
        )
    )
    try:
        for pair in ctrl.pairs.values():
            n = pair.leader.num_dofs()
            pair.base_kp = np.ones(n)
            pair.base_kd = np.ones(n)
            pair.leader.update_kp_kd = lambda kp, kd: None
            pair.leader.command_joint_pos = lambda target: None

        ctrl.set_policy_running(True)
        ctrl.return_to_dagger_pose()
        ctrl.step()
        ctrl._return_deadline = 0.0
        for _ in range(40):
            ctrl.step()
            if ctrl.snapshot()["intervention"]:
                break

        snap = ctrl.snapshot()
        assert snap["returning"] is False
        assert snap["intervention"] is True
        assert snap["dagger_state"] == "intervention"
    finally:
        ctrl.close()


def test_dagger_absolute_return_uses_zero_as_hold_mask():
    absolute = [0.0] * 14
    absolute[1] = 0.1
    absolute[8] = 0.15
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            max_joint_speed=100.0,
            home_speed=2.0,
            command_timeout=10.0,
            return_mode="absolute",
            return_rel_pos=[0.0] * 14,
            return_abs_pos=absolute,
        )
    )
    try:
        ctrl.set_policy_running(True)
        ctrl.set_policy_action({"left": np.full(7, 0.2), "right": np.full(7, 0.3)})
        ctrl.step()
        before_left = np.asarray(ctrl.snapshot()["left"]["pos"], dtype=float)
        before_right = np.asarray(ctrl.snapshot()["right"]["pos"], dtype=float)
        ctrl.return_to_dagger_pose()
        for _ in range(40):
            ctrl.step()
            if ctrl.snapshot()["intervention"]:
                break

        snap = ctrl.snapshot()
        left = np.asarray(snap["left"]["pos"], dtype=float)
        right = np.asarray(snap["right"]["pos"], dtype=float)
        assert left[1] == pytest.approx(0.1, abs=1e-6)
        assert right[1] == pytest.approx(0.15, abs=1e-6)
        assert left[0] == pytest.approx(before_left[0])
        assert right[0] == pytest.approx(before_right[0])
        assert snap["intervention"] is True
    finally:
        ctrl.close()


@pytest.mark.parametrize(
    ("sampling", "radius", "mode", "message"),
    [
        ("random", 0.03, "absolute", "deterministic.*probabilistic"),
        ("deterministic", -0.01, "absolute", "nonnegative"),
        ("probabilistic", 0.0, "absolute", "must be positive"),
        ("probabilistic", 0.03, "relative", "requires.*absolute"),
    ],
)
def test_dagger_return_sampling_config_fails_fast(sampling, radius, mode, message):
    values = [0.2] * 14
    with pytest.raises(ValueError, match=message):
        DaggerController(
            DaggerConfig(
                sim=True,
                return_mode=mode,
                return_rel_pos=values,
                return_abs_pos=values,
                return_sampling=sampling,
                return_radius=radius,
            )
        )


def test_dagger_probabilistic_return_samples_each_arm_and_rejects_joint_limits():
    absolute = [0.2] * 14
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            return_mode="absolute",
            return_abs_pos=absolute,
            return_sampling="probabilistic",
            return_radius=0.03,
        )
    )

    class FakeKinematics:
        def __init__(self, x):
            self.center = np.array([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.sampled = []
            self.seeds = []

        def fk(self, _q):
            return self.center.copy()

        def ik(self, pose, init_q):
            self.sampled.append(np.asarray(pose, dtype=float).copy())
            self.seeds.append(np.asarray(init_q, dtype=float).copy())
            solved = np.asarray(init_q, dtype=float).copy()
            solved[0] = 99.0 if len(self.sampled) == 1 else solved[0] + 0.01
            return solved

    fake = {"left": FakeKinematics(0.4), "right": FakeKinematics(-0.4)}
    ctrl._kin = fake
    ctrl._return_rng = np.random.default_rng(123)
    try:
        ctrl.set_policy_running(True)
        ctrl.return_to_dagger_pose()
        ctrl.step()

        offsets = []
        for side in ("left", "right"):
            kin = fake[side]
            assert len(kin.sampled) == 2
            sampled = kin.sampled[-1]
            offset = sampled[:3] - kin.center[:3]
            offsets.append(offset)
            assert np.linalg.norm(offset) <= 0.03
            assert sampled[3:] == pytest.approx(kin.center[3:])
            assert kin.seeds[-1] == pytest.approx(np.full(7, 0.2))
            assert ctrl._return_target[side][-1] == pytest.approx(0.2)
            assert ctrl._return_target[side][0] == pytest.approx(0.21)
        assert not np.allclose(offsets[0], offsets[1])
    finally:
        ctrl.close()


def test_dagger_probabilistic_return_rejects_arm_zero_masks():
    absolute = [0.2] * 14
    absolute[2] = 0.0
    with pytest.raises(ValueError, match="requires every arm joint.*zero mask"):
        DaggerController(
            DaggerConfig(
                sim=True,
                return_mode="absolute",
                return_abs_pos=absolute,
                return_sampling="probabilistic",
                return_radius=0.03,
            )
        )


def test_dagger_estop_cancels_return_and_invalidates_policy_target():
    relative = [0.0] * 14
    relative[1] = 0.5
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            home_speed=0.2,
            command_timeout=10.0,
            return_mode="relative",
            return_rel_pos=relative,
        )
    )
    try:
        ctrl.set_policy_running(True)
        ctrl.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
        ctrl.step()
        ctrl.return_to_dagger_pose()
        ctrl.step()
        assert ctrl.snapshot()["returning"] is True

        ctrl.set_estop(True)
        ctrl.step()
        snap = ctrl.snapshot()
        assert snap["estop"] is True
        assert snap["returning"] is False
        assert snap["intervention"] is False

        ctrl.set_estop(False)
        ctrl.step()
        snap = ctrl.snapshot()
        assert snap["returning"] is False
        assert snap["left"]["applied"] is None
    finally:
        ctrl.close()


def test_dagger_return_target_is_clamped_to_native_arm_and_gripper_limits(monkeypatch):
    from i2rt.serving import control_config as cc

    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", None)
    absolute = [0.0] * 14
    absolute[3] = 99.0
    absolute[6] = 2.0
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            home_speed=0.2,
            command_timeout=10.0,
            return_mode="absolute",
            return_abs_pos=absolute,
        )
    )
    try:
        ctrl.set_policy_running(True)
        ctrl.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
        ctrl.step()
        native = np.asarray(ctrl.pairs["left"].follower.get_robot_info()["joint_limits"])
        ctrl.return_to_dagger_pose()
        ctrl.step()

        target = ctrl._return_target["left"]
        assert target[3] == pytest.approx(native[3, 1])
        assert target[6] == pytest.approx(1.0)
    finally:
        ctrl.close()


def test_dagger_return_target_uses_narrower_config_limits(monkeypatch):
    from i2rt.serving import control_config as cc

    configured = [None, None, None, (-1.0, 1.0), None, None, (0.2, 0.8)]
    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", configured)
    absolute = [0.0] * 14
    absolute[3] = 99.0
    absolute[6] = 2.0
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            rate=20.0,
            home_speed=0.2,
            command_timeout=10.0,
            return_mode="absolute",
            return_abs_pos=absolute,
        )
    )
    try:
        ctrl.set_policy_running(True)
        ctrl.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
        ctrl.step()
        ctrl.return_to_dagger_pose()
        ctrl.step()

        target = ctrl._return_target["left"]
        assert target[3] == pytest.approx(1.0)
        assert target[6] == pytest.approx(0.8)
    finally:
        ctrl.close()


@pytest.mark.parametrize(
    ("mode", "relative", "absolute", "message"),
    [
        ("cartesian", [0.0] * 14, [0.0] * 14, "relative.*absolute"),
        ("relative", [0.0] * 13, [0.0] * 14, "expects 14 values"),
        ("absolute", [0.0] * 14, [], "required"),
    ],
)
def test_dagger_return_config_fails_fast(mode, relative, absolute, message):
    with pytest.raises(ValueError, match=message):
        DaggerController(
            DaggerConfig(
                sim=True,
                return_mode=mode,
                return_rel_pos=relative,
                return_abs_pos=absolute,
            )
        )


def test_dagger_button_map_toggles_rollout(monkeypatch):
    ctrl = DaggerController(
        DaggerConfig(sim=True, button_map={"left.0": "rollout_toggle", "left.1": "intervention_toggle"})
    )
    current = {"buttons": [1, 0]}

    def fake_read_handle(leader):
        return np.zeros(leader.num_dofs()), None, list(current["buttons"])

    monkeypatch.setattr("i2rt.serving.controllers.read_handle", fake_read_handle)
    ctrl.step()
    assert ctrl.snapshot()["policy_running"] is True
    ctrl.step()  # held button does not retrigger
    assert ctrl.snapshot()["policy_running"] is True
    current["buttons"] = [0, 0]
    ctrl.step()
    current["buttons"] = [1, 0]
    ctrl.step()
    assert ctrl.snapshot()["policy_running"] is False
    ctrl.close()


def test_dagger_button_map_can_request_return(monkeypatch):
    relative = [0.0] * 14
    relative[1] = 0.2
    ctrl = DaggerController(
        DaggerConfig(
            sim=True,
            return_mode="relative",
            return_rel_pos=relative,
            button_map={"left.1": "return"},
        )
    )
    current = {"buttons": [0, 0]}

    def fake_read_handle(leader):
        return np.zeros(leader.num_dofs()), None, list(current["buttons"])

    monkeypatch.setattr("i2rt.serving.controllers.read_handle", fake_read_handle)
    try:
        ctrl.step()  # initialize button edge state
        ctrl.set_policy_running(True)
        current["buttons"] = [0, 1]
        ctrl.step()

        assert ctrl.snapshot()["returning"] is True
        assert ctrl.snapshot()["dagger_state"] == "returning"
    finally:
        ctrl.close()


@pytest.mark.parametrize("policy_running", [False, True])
def test_dagger_intervention_off_relocks_leader_to_held_follower(policy_running):
    ctrl = DaggerController(
        DaggerConfig(sim=True, rate=20.0, home_speed=1.0, command_timeout=0.01)
    )
    try:
        pair = ctrl.pairs["left"]
        n = pair.leader.num_dofs()
        pair.base_kp = np.ones(n)
        pair.base_kd = np.ones(n)
        pair.leader.update_kp_kd = lambda kp, kd: None

        if policy_running:
            ctrl.set_policy_running(True)  # deliberately no policy action/server
        ctrl.set_intervention(True)
        aligned = np.zeros(pair.follower.num_dofs())
        aligned[:n] = 0.2
        pair.follower.command_joint_pos(aligned)
        pair.leader.command_joint_pos(aligned[:n])
        ctrl._smooth["left"].reset(aligned)

        ctrl.set_intervention(False)
        drifted = aligned[:n].copy()
        drifted[1] += 0.2
        pair.leader.command_joint_pos(drifted)
        ctrl.step()

        assert np.asarray(pair.follower.get_joint_pos())[1] == pytest.approx(0.2)
        assert np.asarray(pair.leader.get_joint_pos())[1] == pytest.approx(0.2)

        # Re-entering intervention from the re-locked pose has no follower jump.
        ctrl.set_intervention(True)
        ctrl.step()
        assert np.asarray(pair.follower.get_joint_pos())[1] == pytest.approx(0.2)
    finally:
        ctrl.close()


def test_teleop_bilateral_engage_steps():
    """Engage with bilateral_kp>0 runs through the (fixed) free-until-caught-up path."""
    tc = TeleopController(TeleopConfig(sim=True, bilateral_kp=0.2))
    tc.set_sim_engage(True)
    for _ in range(10):
        tc.step()
    snap = tc.snapshot()
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["left"]["applied"] is not None
    tc.close()


def test_eef_obs_and_safe_osc_roundtrip():
    """EEF FK populates the snapshot; commanding the current EE pose (resolved-rate
    IK -> joint impedance) holds ~the current joints."""
    import numpy as np

    wc = WrapperController(WrapperConfig(sim=True, control="eef", rate=100.0))
    side = "left"
    kin = wc._kin[side]
    assert kin.available  # sim model builds
    cur = np.asarray(wc.followers[side].get_joint_pos(), dtype=float)
    pose = kin.fk(cur)
    assert pose is not None and pose.shape == (7,)

    wc.command({side: pose})  # EEF target = current pose
    wc.step()
    applied = wc.snapshot()[side]["applied"]
    assert applied is not None
    assert np.allclose(np.asarray(applied)[:-1], cur[:-1], atol=1e-2)  # arm holds
    wc.close()


def test_command_staleness_watchdog():
    """A wrapper follower holds (applied=None) once external commands go stale."""
    ctrl = WrapperController(WrapperConfig(sim=True, rate=100.0, command_timeout=0.2))
    ctrl.command({"left": np.zeros(7), "right": np.zeros(7)})
    ctrl.step()
    assert ctrl.snapshot()["left"]["applied"] is not None  # fresh command -> applied
    time.sleep(0.3)  # let the command go stale (link loss)
    ctrl.step()
    assert ctrl.snapshot()["left"]["applied"] is None  # stale -> hold
    ctrl.close()


def test_portal_roundtrip_and_estop():
    port = free_port()
    srv = RobotServer(WrapperController(WrapperConfig(sim=True, rate=100.0)), port=port, rate_hz=100.0)
    threading.Thread(target=srv.serve, daemon=True).start()
    assert wait_port(port), "robot server did not start"

    client = RobotClient(host="127.0.0.1", port=port)
    assert client.metadata["mode"] == "wrapper"

    client.command({"left": np.zeros(7), "right": np.zeros(7)})
    time.sleep(0.3)
    assert client.get_observation()["left"]["applied"] is not None

    # e-stop: commands are ignored while engaged
    client.set_estop(True)
    time.sleep(0.2)
    client.command({"left": np.ones(7), "right": np.ones(7)})
    time.sleep(0.3)
    obs = client.get_observation()
    assert bool(obs["estop"])  # portal serializes bool -> numpy scalar; check truthiness
    assert obs["left"]["applied"] is None

    client.set_estop(False)
    srv.close()
