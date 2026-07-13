"""Fine-grained teleop/DAgger scaling, offsets, reset, and safe handoff."""

from __future__ import annotations

import numpy as np


class StubLeader:
    def __init__(self, n: int = 6):
        self.n = n
        self.pos = np.zeros(n)
        self.cmds = []

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
        self.cmds.append(np.asarray(pos, dtype=float))

    def enter_gravity_comp_idle(self, kd=None):
        pass


class StubFollower:
    def __init__(self, n: int = 7):
        self.n = n
        self.pos = np.zeros(n)
        self.cmds = []

    def num_dofs(self):
        return self.n

    def get_joint_pos(self):
        return self.pos.copy()

    def get_observations(self):
        z = np.zeros(self.n - 1)
        g = np.zeros(1)
        return {
            "joint_pos": self.pos[:-1].copy(),
            "joint_vel": z,
            "joint_eff": z,
            "gripper_pos": g,
            "gripper_vel": g,
            "gripper_eff": g,
        }

    def command_joint_pos(self, pos):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.cmds.append(self.pos.copy())


def _pair(ctl):
    leader, follower = StubLeader(), StubFollower()
    pair = ctl.ArmPair(
        side="left",
        leader=leader,
        follower=follower,
        base_kp=np.full(6, 10.0),
        base_kd=np.full(6, 1.0),
    )
    return pair, leader, follower


def test_mapper_toggle_off_preserves_offset_without_snap():
    from i2rt.serving.teleop_common import FineGrainedMapper

    mapper = FineGrainedMapper(0.2)
    assert np.allclose(mapper.map([0.0], [0.0], enabled=False), [0.0])
    assert np.allclose(mapper.map([1.0], [0.0], enabled=False), [1.0])

    # Toggle on/off ticks retain the exact previous command.
    assert np.allclose(mapper.map([1.0], [1.0], enabled=True), [1.0])
    assert np.allclose(mapper.map([2.0], [1.0], enabled=True), [1.2])
    assert np.allclose(mapper.map([2.0], [1.2], enabled=False), [1.2])

    # Normal motion is 1:1 again, retaining the -0.8 rad accumulated offset.
    assert np.allclose(mapper.map([3.0], [1.2], enabled=False), [2.2])


def test_teleop_left_upper_scales_then_resumes_one_to_one_and_home_resets(monkeypatch):
    from i2rt.serving import controllers as ctl

    pair, leader, follower = _pair(ctl)
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    current = {"buttons": [0, 0]}
    monkeypatch.setattr(
        ctl,
        "read_handle",
        lambda _leader: (leader.pos.copy(), 0.0, list(current["buttons"])),
    )
    ctrl = ctl.TeleopController(ctl.TeleopConfig(fine_grained_scale=0.2, ramp_speed=100.0, home_speed=1.0))
    ctrl.set_sim_engage(True)
    ctrl.step()  # zero-gap catch-up completes

    current["buttons"] = [1, 0]
    ctrl.step()
    assert ctrl.snapshot()["teleop_state"] == "ENGAGED"  # left.0 is not home
    assert ctrl.snapshot()["fine_grained"] is True

    current["buttons"] = [0, 0]
    leader.pos[0] = 1.0
    ctrl.step()
    assert abs(follower.cmds[-1][0] - 0.2) < 1e-9

    current["buttons"] = [1, 0]
    ctrl.step()  # toggle off: no target jump
    assert ctrl.snapshot()["fine_grained"] is False
    assert abs(follower.cmds[-1][0] - 0.2) < 1e-9

    current["buttons"] = [0, 0]
    leader.pos[0] = 2.0
    ctrl.step()
    assert abs(follower.cmds[-1][0] - 1.2) < 1e-9  # 1:1 with constant -0.8 offset

    current["buttons"] = [0, 1]  # left.1 outcome/home
    ctrl.step()
    assert ctrl.snapshot()["teleop_state"] == "HOMING"
    assert ctrl.snapshot()["fine_grained"] is False


def test_dagger_context_button_and_policy_handoff_are_safe(monkeypatch):
    from i2rt.serving import controllers as ctl

    pair, leader, follower = _pair(ctl)
    monkeypatch.setattr(ctl, "build_bimanual", lambda specs, sim: {"left": pair})
    current = {"buttons": [0, 0]}
    monkeypatch.setattr(
        ctl,
        "read_handle",
        lambda _leader: (leader.pos.copy(), 0.0, list(current["buttons"])),
    )
    ctrl = ctl.DaggerController(
        ctl.DaggerConfig(feedback_kp=0.0, fine_grained_scale=0.2, max_joint_speed=1.2, rate=120.0)
    )

    # Outside intervention left.0 retains rollout-toggle behavior.
    current["buttons"] = [1, 0]
    ctrl.step()
    assert ctrl.snapshot()["policy_running"] is True
    current["buttons"] = [0, 0]
    ctrl.step()

    ctrl.set_intervention(True)
    ctrl.step()  # establish the normal relative anchor
    current["buttons"] = [1, 0]
    ctrl.step()
    assert ctrl.snapshot()["fine_grained"] is True

    current["buttons"] = [0, 0]
    leader.pos[0] = 1.0
    ctrl.step()
    assert abs(ctrl.snapshot()["left"]["human"][0] - 0.2) < 1e-9

    # Hand back at a large leader/follower offset. Fine mode resets, and the first
    # mirrored leader setpoint moves by at most max_joint_speed/rate.
    ctrl.set_intervention(False)
    ctrl.set_policy_action({"left": np.ones(7)})
    ctrl.step()
    assert ctrl.snapshot()["fine_grained"] is False
    assert leader.cmds
    assert np.max(np.abs(leader.cmds[-1] - leader.pos)) <= ctrl._run_step + 1e-9

    ctrl.set_intervention(True)
    ctrl.step()  # next takeover starts anchored in normal mode, without a snap
    assert ctrl.snapshot()["fine_grained"] is False
    assert np.allclose(ctrl.snapshot()["left"]["human"][:6], follower.pos[:6])
