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


def test_dagger_intervention_tracks_gripper_directly_but_limits_arm(monkeypatch):
    ctrl = DaggerController(DaggerConfig(sim=True, rate=120.0, max_joint_speed=0.12))

    def fake_read_handle(leader):
        return np.ones(leader.num_dofs()), 0.0, []

    monkeypatch.setattr("i2rt.serving.controllers.read_handle", fake_read_handle)
    try:
        ctrl.set_policy_running(True)
        ctrl.set_intervention(True)
        for side, smoother in ctrl._smooth.items():
            smoother.reset(np.array([0.0] * 6 + [1.0]))
            ctrl._fine_mapper[side].map(np.zeros(6), np.zeros(6), enabled=False)

        ctrl.step()

        applied = np.asarray(ctrl.snapshot()["left"]["applied"], dtype=float)
        assert applied[:6] == pytest.approx(np.full(6, 0.001))
        assert applied[-1] == pytest.approx(0.0)
        assert ctrl._smooth["left"].cur[-1] == pytest.approx(0.0)
    finally:
        ctrl.close()


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
