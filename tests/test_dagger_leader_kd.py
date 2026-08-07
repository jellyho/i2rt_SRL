"""Leader grav-comp feel is phase-dependent (teleop AND DAgger): the config.yaml
leader_* overrides (kd / coulomb / gravity factor — the "human-held feel") apply
only while the human actually holds the arm (teleop ENGAGED, DAgger intervention);
in every other phase (homing, idle, policy mirroring, stopped) they revert to the
yam.yml originals. Uses stub arm pairs so no CAN/mujoco is needed.
"""

from __future__ import annotations

import threading

import numpy as np

# yam.yml originals (construction state, and what every not-held phase uses)
GRAV_COMP_KD = np.array([0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
ORIG_FACTOR = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
ORIG_COULOMB = np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06])
# config.yaml leader_* overrides (human-held feel, applied at runtime only)
FREE_FACTOR = [1.0, 1.2, 1.2, 1.3, 1.0, 1.0]
FREE_COULOMB = [0.5, 0.5, 0.5, 0.06, 0.06, 0.06]
FREE_KD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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


def _set_free_feel(monkeypatch):
    """Simulate config.yaml leader_* overrides (the human-held feel)."""
    from i2rt.serving import control_config as cc

    monkeypatch.setattr(cc, "LEADER_GRAVITY_COMP_FACTOR", list(FREE_FACTOR), raising=False)
    monkeypatch.setattr(cc, "LEADER_COULOMB_FRICTION", list(FREE_COULOMB), raising=False)
    monkeypatch.setattr(cc, "LEADER_GRAV_COMP_KD", list(FREE_KD), raising=False)


def _make_dagger(monkeypatch, feedback_kp: float, leader_kd: np.ndarray = None):
    from i2rt.serving import controllers as ctl

    _set_free_feel(monkeypatch)
    leader, follower = StubLeader(grav_comp_kd=leader_kd), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    dc = ctl.DeployController(ctl.DeployConfig(feedback_kp=feedback_kp))
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
    # the leader_grav_comp_kd override kd is passed explicitly (the leader itself
    # is BUILT with the yam.yml original) — intervention feels like teleop engaged
    assert np.allclose(leader.idle_calls[-1], FREE_KD)
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

    _set_free_feel(monkeypatch)
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
    # IDLE frees the leader with its BUILT-IN (original) damping, kd untouched
    assert leader.idle_calls and leader.idle_calls[-1] is None


def test_teleop_engaged_applies_override_feel(monkeypatch):
    tc, leader, _ = _make_teleop(monkeypatch)
    tc.step()
    tc.set_sim_engage(True)
    tc.step()
    assert np.allclose(leader.factor_calls[-1], FREE_FACTOR)
    assert np.allclose(leader.coulomb_calls[-1], FREE_COULOMB)
    assert np.allclose(leader.idle_calls[-1], FREE_KD)  # override kd passed explicitly


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


# ------------------------------------------------------- leader mirroring on/off
def test_mirroring_off_frees_the_leader_instead_of_driving_it(monkeypatch):
    """Mirroring off must FREE the leader, not merely skip the drive.

    Skipping alone would leave whatever PD gains were last commanded, so the handles
    would keep stiffly holding a stale target — worse than mirroring."""
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    dc.set_policy_running(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    assert leader.cmd_calls, "sanity: mirroring on should command the leader"

    dc.set_leader_mirror(False)
    n_cmds, n_idle = len(leader.cmd_calls), len(leader.idle_calls)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    assert len(leader.cmd_calls) == n_cmds  # no new PD target
    assert len(leader.idle_calls) > n_idle  # explicitly freed instead


def test_mirroring_can_be_turned_back_on_mid_rollout(monkeypatch):
    dc, leader, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    dc.set_policy_running(True)
    dc.set_leader_mirror(False)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    n_cmds = len(leader.cmd_calls)

    dc.set_leader_mirror(True)
    dc.set_policy_action({"left": np.zeros(7)})
    dc.step()
    assert len(leader.cmd_calls) > n_cmds  # driving the leader again
    kp, _ = leader.kp_kd_calls[-1]
    assert np.allclose(kp, np.full(6, 10.0) * dc.mirror_kp)


def test_mirror_state_defaults_from_mirror_kp_and_is_reported(monkeypatch):
    from i2rt.serving import controllers as ctl

    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    assert dc._leader_mirror is True  # default mirror_kp > 0
    dc.step()
    assert dc.snapshot()["leader_mirror"] is True
    dc.set_leader_mirror(False)
    dc.step()
    assert dc.snapshot()["leader_mirror"] is False

    # launching with --mirror-kp 0 means "do not mirror" rather than a silent no-op
    _set_free_feel(monkeypatch)
    leader, follower = StubLeader(), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    assert ctl.DeployController(ctl.DeployConfig(mirror_kp=0.0))._leader_mirror is False


# --------------------------------------------------- gripper closed before a rollout
def _stateful(dc, gripper=0.6):
    """Make the stub followers remember what they were commanded.

    The default StubFollower always reports zeros, i.e. a gripper that is already shut --
    which would make every assertion below pass for the wrong reason.
    """
    for pair in dc.pairs.values():
        f = pair.follower
        f._q = np.zeros(f.n)
        f._q[-1] = gripper
        f.get_joint_pos = lambda _f=f: _f._q.copy()
        f.command_joint_pos = lambda pos, _f=f: _f._q.__setitem__(slice(None), np.asarray(pos, float))
    return dc


def test_rollout_waits_for_the_gripper_to_close(monkeypatch):
    """Every recorded episode starts with the gripper shut, so a rollout that begins on an
    open one hands the policy a first observation it never saw in training."""
    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.6)

    dc.set_policy_running(True)
    assert dc._closing_grip is True
    assert dc._policy_running is False, "the policy must not drive until the gripper is shut"

    for _ in range(500):
        dc.step()
        if not dc._closing_grip:
            break
    assert dc._closing_grip is False
    assert dc._policy_running is True
    for pair in dc.pairs.values():
        assert abs(float(pair.follower.get_joint_pos()[-1]) - dc.home_grip) < 0.05


def test_already_closed_gripper_starts_immediately(monkeypatch):
    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.0)
    dc.set_policy_running(True)
    assert dc._closing_grip is False
    assert dc._policy_running is True


def test_closing_only_moves_the_gripper(monkeypatch):
    """The arm must not swing while the gripper shuts."""
    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.6)
    for pair in dc.pairs.values():  # put the arm somewhere non-zero
        pair.follower._q[:-1] = np.linspace(0.1, 0.6, pair.follower.n - 1)
    before = {s: p.follower.get_joint_pos()[:-1].copy() for s, p in dc.pairs.items()}

    dc.set_policy_running(True)
    for _ in range(500):
        dc.step()
        if not dc._closing_grip:
            break
    for side, pair in dc.pairs.items():
        np.testing.assert_allclose(pair.follower.get_joint_pos()[:-1], before[side], atol=1e-6)


def test_reported_state_says_it_is_closing(monkeypatch):
    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.6)
    dc.set_policy_running(True)
    dc.step()
    assert dc.snapshot()["dagger_state"] == "closing_gripper"


def test_stop_cancels_a_pending_close(monkeypatch):
    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.6)
    dc.set_policy_running(True)
    assert dc._closing_grip is True
    dc.set_policy_running(False)
    assert dc._closing_grip is False
    assert dc._policy_running is False


def test_blocked_gripper_starts_anyway_after_the_timeout(monkeypatch):
    """A gripper on an object never reaches home; refusing to start would leave the
    operator pressing a button that does nothing."""
    from i2rt.serving import controllers as ctl

    dc, _, _ = _make_dagger(monkeypatch, feedback_kp=0.1)
    _stateful(dc, gripper=0.6)
    dc.set_policy_running(True)
    assert dc._closing_grip is True

    monkeypatch.setattr(ctl, "_GRIP_CLOSE_TIMEOUT", 0.0)
    for pair in dc.pairs.values():  # gripper physically cannot move
        pair.follower.command_joint_pos = lambda pos, _f=pair.follower: None
    dc.step()
    assert dc._closing_grip is False
    assert dc._policy_running is True
