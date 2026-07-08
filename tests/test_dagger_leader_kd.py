"""Leader grav-comp feel is phase-dependent (teleop AND DAgger): the config.yaml
leader_* overrides (kd / coulomb / gravity factor — the "human-held feel") apply
only while the human actually holds the arm (teleop ENGAGED, DAgger intervention);
in every other phase (homing, idle, policy mirroring, stopped) they revert to the
yam.yml originals. Uses stub arm pairs so no CAN/mujoco is needed.
"""

from __future__ import annotations

import threading

import numpy as np

# yam.yml originals (what mirroring must revert to)
GRAV_COMP_KD = np.array([0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
ORIG_FACTOR = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
ORIG_COULOMB = np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06])
# pretend config.yaml leader_* overrides the leader was BUILT with (human-held feel)
FREE_FACTOR = np.array([1.0, 1.2, 1.2, 1.3, 1.0, 1.0])
FREE_COULOMB = np.array([0.5, 0.5, 0.5, 0.06, 0.06, 0.06])


class StubLeader:
    """6-dof teaching-handle leader that records every command it receives."""

    def __init__(self, n: int = 6, grav_comp_kd: np.ndarray = None):
        self.n = n
        self.pos = np.zeros(n)  # physical joint positions (settable by tests)
        self.grav_comp_kd = GRAV_COMP_KD.copy() if grav_comp_kd is None else np.asarray(grav_comp_kd, dtype=float)
        self.kp_kd_calls = []  # (kp, kd) from update_kp_kd
        self.cmd_calls = []  # targets from command_joint_pos
        self.idle_calls = []  # kd (or None) from enter_gravity_comp_idle
        self.factor_calls = []  # from set_gravity_comp_factor
        self.coulomb_calls = []  # from set_coulomb_friction

    def num_dofs(self):
        return self.n

    def get_joint_pos(self):
        return self.pos.copy()

    def get_observations(self):
        z = np.zeros(self.n)
        return {"joint_pos": self.pos.copy(), "joint_vel": z, "joint_eff": z}

    def get_robot_info(self):
        return {
            "kp": np.full(self.n, 10.0),
            "kd": np.full(self.n, 1.0),
            "grav_comp_kd": self.grav_comp_kd.copy(),
            "gravity_comp_factor": FREE_FACTOR.copy(),
            "coulomb_friction": FREE_COULOMB.copy(),
        }

    def update_kp_kd(self, kp, kd):
        self.kp_kd_calls.append((np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)))

    def command_joint_pos(self, pos):
        self.cmd_calls.append(np.asarray(pos, dtype=float))

    def enter_gravity_comp_idle(self, kd=None):
        self.idle_calls.append(None if kd is None else np.asarray(kd, dtype=float))

    def set_gravity_comp_factor(self, factor):
        self.factor_calls.append(np.asarray(factor, dtype=float))

    def set_coulomb_friction(self, coulomb):
        self.coulomb_calls.append(np.asarray(coulomb, dtype=float))


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


def _make_dagger(monkeypatch, feedback_kp: float, leader_kd: np.ndarray = None):
    from i2rt.serving import controllers as ctl

    leader, follower = StubLeader(grav_comp_kd=leader_kd), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    dc = ctl.DaggerController(ctl.DaggerConfig(feedback_kp=feedback_kp))
    return dc, leader, follower


def test_mirror_applies_arm_default_grav_comp_kd(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    kp, kd = leader.kp_kd_calls[-1]
    assert np.allclose(kp, np.full(6, 10.0) * dc.mirror_kp)
    assert np.allclose(kd, GRAV_COMP_KD)  # damping ON while mirroring the policy


def test_mirror_damping_ignores_leader_free_mode_override(monkeypatch):
    # leader built with leader_grav_comp_kd: 0 (free-mode feel) — the mirror phase
    # must still use the arm's ORIGINAL yam.yml damping, not the override.
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1, leader_kd=np.zeros(6))
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    _, kd = leader.kp_kd_calls[-1]
    assert np.allclose(kd, GRAV_COMP_KD)  # yam.yml value, damping stays ON


def test_intervention_with_zero_feedback_kp_frees_leader(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.0)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()  # mirror phase leaves a PD command on the leader
    n_cmds = len(leader.cmd_calls)

    dc.set_intervention(True)
    dc.step()
    assert leader.idle_calls, "feedback_kp=0 intervention must enter grav-comp idle"
    # kd=None -> the leader's own configured kd, i.e. the leader_grav_comp_kd
    # override — intervention feels exactly like teleop free mode
    assert leader.idle_calls[-1] is None
    assert len(leader.cmd_calls) == n_cmds  # the stale mirror target is NOT re-commanded


def test_mirror_reverts_grav_ff_to_yam_defaults(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.0)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    assert np.allclose(leader.factor_calls[-1], ORIG_FACTOR)  # yam.yml, not the override
    assert np.allclose(leader.coulomb_calls[-1], ORIG_COULOMB)


def test_intervention_restores_leader_override_grav_ff(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.0)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()  # mirror -> originals
    dc.set_intervention(True)
    dc.step()  # human takes over -> back to the human-held feel
    assert np.allclose(leader.factor_calls[-1], FREE_FACTOR)
    assert np.allclose(leader.coulomb_calls[-1], FREE_COULOMB)


def test_dagger_stopped_uses_original_grav_ff(monkeypatch):
    # nobody driving, nobody holding -> yam.yml feel (human-held only during intervention)
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.0)
    dc.step()
    assert np.allclose(leader.factor_calls[-1], ORIG_FACTOR)
    assert np.allclose(leader.coulomb_calls[-1], ORIG_COULOMB)


# ---------------------------------------------------------------------------
# Teleop: overrides only while ENGAGED; HOMING / IDLE use the yam.yml originals
# ---------------------------------------------------------------------------
def _make_teleop(monkeypatch, **cfg_kw):
    from i2rt.serving import controllers as ctl

    leader, follower = StubLeader(), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    tc = ctl.TeleopController(ctl.TeleopConfig(**cfg_kw))
    return tc, leader, follower


def test_teleop_not_engaged_uses_original_grav_ff(monkeypatch):
    tc, leader, _ = _make_teleop(monkeypatch)
    tc.step()
    tc.step()  # HOMING -> IDLE; either way the human is not holding the arm
    assert np.allclose(leader.factor_calls[-1], ORIG_FACTOR)
    assert np.allclose(leader.coulomb_calls[-1], ORIG_COULOMB)
    # IDLE frees the leader with the ORIGINAL damping, not the override
    assert leader.idle_calls and np.allclose(leader.idle_calls[-1], GRAV_COMP_KD)


def test_teleop_engaged_applies_override_feel(monkeypatch):
    tc, leader, _ = _make_teleop(monkeypatch)
    tc.step()
    tc.set_sim_engage(True)
    tc.step()
    assert np.allclose(leader.factor_calls[-1], FREE_FACTOR)
    assert np.allclose(leader.coulomb_calls[-1], FREE_COULOMB)
    assert leader.idle_calls[-1] is None  # free with the leader's own override kd


def test_homing_done_requires_physical_leader_at_home(monkeypatch):
    tc, leader, _ = _make_teleop(monkeypatch)
    # virtual ramp targets (smoothers) already sit at home after construction,
    # but the physical leader lags behind the weak home_kp pull
    leader.pos = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert not tc._homing_done()  # must NOT flip to IDLE and drop the leader here
    leader.pos = np.zeros(6)
    assert tc._homing_done()


def test_motor_chain_robot_grav_ff_setters():
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    r = MotorChainRobot.__new__(MotorChainRobot)  # skip hardware __init__
    r._state_lock = threading.Lock()
    r.gravity_comp_factor = np.ones(6)
    r._coulomb_friction = np.zeros(6)
    r.set_gravity_comp_factor(np.full(6, 1.1))
    r.set_coulomb_friction(np.full(6, 0.3))
    assert np.allclose(r.gravity_comp_factor, 1.1)
    assert np.allclose(r._coulomb_friction, 0.3)


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
