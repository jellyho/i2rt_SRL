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


def test_homing_puts_the_object_down_before_travelling():
    """finish -> open (arm still) -> travel home with the gripper open -> close at home.

    Homing used to ramp straight to the shut home pose, so an episode that ended mid-grasp
    carried the object across the workspace and kept holding it at home. Releasing first
    drops it where it was being worked on -- which only holds if the arm does not move
    during the release, or "put it down here" becomes "put it down somewhere on the way".
    """
    dc = DaggerController(DaggerConfig(sim=True, max_joint_speed=10.0, home_speed=1.0))
    dc.step()
    if not dc._has_grip:
        pytest.skip("rig has no gripper")

    # The travel target keeps the gripper open; only the tail after it shuts.
    assert dc.home_full_released[-1] == dc.release_grip
    assert dc.release_grip != dc.home_grip

    # End a rollout away from home, mid-grasp.
    for pair in dc.pairs.values():
        pos = np.asarray(pair.follower.get_joint_pos(), dtype=float).copy()
        pos[: dc.home_arm.size] += 0.4
        pos[-1] = (dc.home_grip + dc.release_grip) / 2.0  # half open == still gripping
        pair.follower.command_joint_pos(pos)
    for _ in range(40):
        dc.step()
    arm_at_finish = {
        side: np.asarray(p.follower.get_joint_pos(), dtype=float)[: dc.home_arm.size].copy()
        for side, p in dc.pairs.items()
    }

    dc.finish_dagger_run("keep")
    dc.step()  # the snapshot is rebuilt by step(), not by the command
    assert dc.snapshot()["dagger_state"] == "releasing_gripper"
    assert dc.snapshot()["homing"] is True, "the whole routine must read as homing"

    seen = []
    arm_drift_while_releasing = 0.0
    grip_while_homing = []
    for _ in range(2000):
        state = dc.snapshot()["dagger_state"]
        if not seen or seen[-1] != state:
            seen.append(state)
        for side, pair in dc.pairs.items():
            q = np.asarray(pair.follower.get_joint_pos(), dtype=float)
            if state == "releasing_gripper":
                arm_drift_while_releasing = max(
                    arm_drift_while_releasing,
                    float(np.max(np.abs(q[: dc.home_arm.size] - arm_at_finish[side]))),
                )
            elif state == "homing":
                grip_while_homing.append(float(q[-1]))
        dc.step()
        if state == "stopped" and len(seen) > 1:
            break

    assert seen[0] == "releasing_gripper", seen
    assert seen[-1] == "stopped", seen
    assert "closing_gripper" in seen, seen
    assert seen.index("closing_gripper") > seen.index("releasing_gripper"), seen
    # The arm holds position while the gripper opens (a little PD sag is fine).
    assert arm_drift_while_releasing < 0.05, arm_drift_while_releasing
    # ...and it travels home OPEN, so the object is not re-gripped on the way.
    assert grip_while_homing, "expected a travel phase after moving the arm off home"
    assert min(grip_while_homing) > (dc.home_grip + dc.release_grip) / 2.0, min(grip_while_homing)


def test_nothing_can_cut_into_the_homing_routine():
    """Release and the closing tail are as protected as the travel itself."""
    dc = DaggerController(DaggerConfig(sim=True, max_joint_speed=10.0, home_speed=10.0))
    dc.step()
    if not dc._has_grip:
        pytest.skip("rig has no gripper")
    for pair in dc.pairs.values():
        pos = np.asarray(pair.follower.get_joint_pos(), dtype=float).copy()
        pos[-1] = (dc.home_grip + dc.release_grip) / 2.0
        pair.follower.command_joint_pos(pos)
    for _ in range(30):
        dc.step()

    dc.finish_dagger_run("keep")
    dc.step()
    assert dc.snapshot()["dagger_state"] == "releasing_gripper"
    dc.set_policy_running(True)
    dc.set_intervention(True)
    assert dc.snapshot()["policy_running"] is False
    assert dc.snapshot()["intervention"] is False
