"""DAgger leader damping is phase-dependent: grav_comp_kd ON while the leader
mirrors the policy, OFF (fully free) while the human intervenes with
feedback_kp = 0. Uses stub arm pairs so no CAN/mujoco is needed.
"""

from __future__ import annotations

import threading

import numpy as np

GRAV_COMP_KD = np.array([0.1, 0.1, 0.1, 0.3, 0.05, 0.05])


class StubLeader:
    """6-dof teaching-handle leader that records every command it receives."""

    def __init__(self, n: int = 6):
        self.n = n
        self.kp_kd_calls = []  # (kp, kd) from update_kp_kd
        self.cmd_calls = []  # targets from command_joint_pos
        self.idle_calls = []  # kd (or None) from enter_gravity_comp_idle

    def num_dofs(self):
        return self.n

    def get_joint_pos(self):
        return np.zeros(self.n)

    def get_observations(self):
        z = np.zeros(self.n)
        return {"joint_pos": z, "joint_vel": z, "joint_eff": z}

    def get_robot_info(self):
        return {"kp": np.full(self.n, 10.0), "kd": np.full(self.n, 1.0), "grav_comp_kd": GRAV_COMP_KD.copy()}

    def update_kp_kd(self, kp, kd):
        self.kp_kd_calls.append((np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)))

    def command_joint_pos(self, pos):
        self.cmd_calls.append(np.asarray(pos, dtype=float))

    def enter_gravity_comp_idle(self, kd=None):
        self.idle_calls.append(None if kd is None else np.asarray(kd, dtype=float))


class StubFollower:
    """7-dof follower (6 arm + gripper)."""

    def __init__(self, n: int = 7):
        self.n = n
        self.cmds = []

    def num_dofs(self):
        return self.n

    def get_joint_pos(self):
        return np.zeros(self.n)

    def get_observations(self):
        z = np.zeros(self.n - 1)
        g = np.zeros(1)
        return {
            "joint_pos": z, "joint_vel": z, "joint_eff": z,
            "gripper_pos": g, "gripper_vel": g, "gripper_eff": g,
        }

    def command_joint_pos(self, pos):
        self.cmds.append(np.asarray(pos, dtype=float))


def _make_dagger(monkeypatch, feedback_kp: float):
    from i2rt.serving import controllers as ctl

    leader, follower = StubLeader(), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    dc = ctl.DaggerController(ctl.DaggerConfig(feedback_kp=feedback_kp))
    return dc, leader, follower


def test_mirror_applies_leader_grav_comp_kd(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    kp, kd = leader.kp_kd_calls[-1]
    assert np.allclose(kp, np.full(6, 10.0) * dc.mirror_kp)
    assert np.allclose(kd, GRAV_COMP_KD)  # damping ON while mirroring the policy


def test_intervention_with_zero_feedback_kp_frees_leader(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.0)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()  # mirror phase leaves a PD command on the leader
    n_cmds = len(leader.cmd_calls)

    dc.set_intervention(True)
    dc.step()
    assert leader.idle_calls, "feedback_kp=0 intervention must enter grav-comp idle"
    assert np.allclose(leader.idle_calls[-1], np.zeros(6))  # damping OFF while intervening
    assert len(leader.cmd_calls) == n_cmds  # the stale mirror target is NOT re-commanded


def test_intervention_with_feedback_kp_keeps_pd_force_feel(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    dc.set_policy_running(True)
    dc.set_intervention(True)
    dc.step()
    assert not leader.idle_calls  # PD force feel, not free mode
    kp, kd = leader.kp_kd_calls[-1]
    assert np.allclose(kp, np.full(6, 10.0) * 0.1)
    assert np.allclose(kd, np.zeros(6))  # unchanged legacy behavior


def test_enter_gravity_comp_idle_kd_override():
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    r = MotorChainRobot.__new__(MotorChainRobot)  # skip hardware __init__
    r._command_lock = threading.Lock()
    r.motor_chain = [None] * 6
    r._grav_comp_kd = GRAV_COMP_KD.copy()

    r.enter_gravity_comp_idle()
    assert np.allclose(r._commands.kd, GRAV_COMP_KD)  # default: configured damping
    r.enter_gravity_comp_idle(kd=np.zeros(6))
    assert np.allclose(r._commands.kd, np.zeros(6))  # per-call override
