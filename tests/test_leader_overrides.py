"""Leader-specific gravity-comp overrides: config.yaml -> leader construction.

The leader (human-held) arm's free-mode feel is set by three per-joint vectors
that normally come from the shared arm yml (yam.yml): gravity_comp_factor,
grav_comp_kd, coulomb_friction. These tests cover the leader-only override path
so the leader can be tuned lighter without touching the follower.
"""

from __future__ import annotations

import numpy as np
import pytest

from i2rt.serving import control_config as cc


def _reset_leader_overrides(monkeypatch):
    for attr in ("LEADER_GRAVITY_COMP_FACTOR", "LEADER_GRAV_COMP_KD", "LEADER_COULOMB_FRICTION"):
        monkeypatch.setattr(cc, attr, None, raising=False)


def test_leader_arm_overrides_empty_by_default(monkeypatch):
    _reset_leader_overrides(monkeypatch)
    assert cc.leader_arm_overrides() == {}


def test_leader_arm_overrides_only_set_values(monkeypatch):
    _reset_leader_overrides(monkeypatch)
    monkeypatch.setattr(cc, "LEADER_GRAVITY_COMP_FACTOR", 1.1, raising=False)
    monkeypatch.setattr(cc, "LEADER_GRAV_COMP_KD", [0.05] * 6, raising=False)
    kw = cc.leader_arm_overrides()
    assert kw == {"gravity_comp_factor": 1.1, "grav_comp_kd": [0.05] * 6}


def test_override_vec_broadcast():
    from i2rt.robots.get_robot import _override_vec

    base = np.array([0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
    assert np.allclose(_override_vec(base, None), base)  # None -> keep defaults
    assert np.allclose(_override_vec(base, 0.05), np.full(6, 0.05))  # scalar broadcast
    assert np.allclose(_override_vec(base, [1, 2, 3, 4, 5, 6]), [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        _override_vec(base, [1.0, 2.0])  # wrong per-joint length


def test_build_pair_builds_both_arms_with_yam_defaults(monkeypatch):
    # The leader_* overrides are a RUNTIME feel (applied by the controllers only
    # while a human holds the arm). Construction must stay on the yam.yml
    # originals so the leader is properly damped from power-on — even during the
    # window before the control loop starts stepping.
    import i2rt.serving.teleop_common as tc

    _reset_leader_overrides(monkeypatch)
    monkeypatch.setattr(cc, "LEADER_COULOMB_FRICTION", 0.45, raising=False)
    monkeypatch.setattr(cc, "LEADER_GRAV_COMP_KD", [0.0] * 6, raising=False)

    calls = []

    class _Stub:
        def get_robot_info(self):
            return {}

    def fake_get_yam_robot(**kw):
        calls.append(kw)
        return _Stub()

    monkeypatch.setattr(tc, "get_yam_robot", fake_get_yam_robot)
    tc.build_pair(tc.PairSpec(side="left", leader_channel="can_l", follower_channel="can_f"), sim=False)

    leader_kw, follower_kw = calls
    for kw in (leader_kw, follower_kw):
        for key in ("coulomb_friction", "grav_comp_kd", "gravity_comp_factor"):
            assert key not in kw
