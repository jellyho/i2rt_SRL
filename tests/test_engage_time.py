"""Fixed-time engage catch-up: with control.engage_time set, the follower meets
the LIVE leader pose in a fixed duration via a cosine blend (velocity-matched at
handoff), stretched only if the implied joint speed would exceed ramp_speed.
engage_time omitted/0 keeps the legacy speed-based catch-up.
"""

from __future__ import annotations

import time as _time

import numpy as np


class StubLeader:
    def __init__(self, n: int = 6):
        self.n = n
        self.pos = np.zeros(n)

    def num_dofs(self):
        return self.n

    def get_joint_pos(self):
        return self.pos.copy()

    def get_observations(self):
        z = np.zeros(self.n)
        return {"joint_pos": self.pos.copy(), "joint_vel": z, "joint_eff": z}

    def get_robot_info(self):
        return {"kp": np.full(self.n, 10.0), "kd": np.full(self.n, 1.0)}

    def update_kp_kd(self, kp, kd):
        pass

    def command_joint_pos(self, pos):
        pass

    def enter_gravity_comp_idle(self, kd=None):
        pass


class StubFollower:
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


def _make_teleop(monkeypatch, **cfg_kw):
    from i2rt.serving import controllers as ctl

    leader, follower = StubLeader(), StubFollower()
    pair = ctl.ArmPair(side="left", leader=leader, follower=follower,
                       base_kp=np.full(6, 10.0), base_kd=np.full(6, 1.0))
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    tc = ctl.TeleopController(ctl.TeleopConfig(**cfg_kw))
    return tc, leader, follower


def _fake_clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(_time, "monotonic", lambda: t["now"])
    return t


def test_engage_time_blend_reaches_leader_in_fixed_time(monkeypatch):
    t = _fake_clock(monkeypatch)
    # huge ramp_speed so the ceiling never stretches the 2 s duration
    tc, leader, follower = _make_teleop(monkeypatch, engage_time=2.0, ramp_speed=100.0)
    tc.step()  # settle in HOMING/IDLE at home

    leader.pos = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    tc.set_sim_engage(True)
    tc.step()  # ENGAGED transition: blend starts now (tau ~ 0)
    assert abs(follower.cmds[-1][1]) < 0.05  # still ~at home

    t["now"] = 1.0  # halfway: cosine blend s(0.5) = 0.5
    tc.step()
    assert abs(follower.cmds[-1][1] - 0.2) < 0.02

    t["now"] = 2.1  # past T: caught up, direct tracking of the live leader
    tc.step()
    assert abs(follower.cmds[-1][1] - 0.4) < 1e-6
    leader.pos = np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0])
    tc.step()
    assert abs(follower.cmds[-1][1] - 0.6) < 1e-6  # 1:1 after handoff


def test_engage_time_stretched_by_speed_ceiling(monkeypatch):
    t = _fake_clock(monkeypatch)
    # 1.0 rad gap, ramp_speed 0.5 -> cosine peak speed cap gives T_min = pi*1.0/(2*0.5) = pi
    tc, leader, follower = _make_teleop(monkeypatch, engage_time=1.0, ramp_speed=0.5)
    tc.step()
    leader.pos = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    tc.set_sim_engage(True)
    tc.step()
    assert tc._engage_T["left"] >= np.pi - 1e-6  # stretched beyond the 1 s request

    t["now"] = 1.0  # the requested duration — must NOT have arrived yet
    tc.step()
    assert follower.cmds[-1][1] < 0.8


def test_config_engage_time_reaches_teleop_config():
    """Regression: config.yaml `control.engage_time` must actually reach the controller.

    TeleopConfig binds its default `engage_time=cc.ENGAGE_TIME` at import time (0.0), so
    run_robot_server must read the value AFTER apply_control_overrides (argparse default)
    and pass it into TeleopConfig — otherwise the fixed-time engage is silently off and the
    follower falls back to speed-based catch-up that can chase a moving leader forever.
    """
    import argparse

    from i2rt.serving import control_config as cc
    from i2rt.serving.controllers import TeleopConfig
    from i2rt.serving.rig_config import apply_control_overrides

    prev = cc.ENGAGE_TIME
    try:
        assert TeleopConfig().engage_time == prev  # dataclass default is the import-time value
        apply_control_overrides({"control": {"engage_time": 2.0}})
        assert cc.ENGAGE_TIME == 2.0
        # run_robot_server builds this arg AFTER the override, then passes it through.
        p = argparse.ArgumentParser()
        p.add_argument("--engage-time", type=float, default=cc.ENGAGE_TIME)
        args = p.parse_args([])
        assert TeleopConfig(engage_time=args.engage_time).engage_time == 2.0
    finally:
        cc.ENGAGE_TIME = prev
