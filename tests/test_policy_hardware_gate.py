"""Real-policy hardware startup gate tests."""

from __future__ import annotations

import pytest

from i2rt.serving import control_config as cc
from i2rt.serving.run_robot_server import validate_real_dagger_safety


def test_real_dagger_requires_joint_and_effort_limits(monkeypatch):
    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", None)
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", None)

    with pytest.raises(ValueError, match="follower_joint_limits"):
        validate_real_dagger_safety(0.2, 0.25)


def test_real_dagger_accepts_complete_safety_configuration(monkeypatch):
    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", [(-3.0, 3.0)] * 6 + [(0.0, 1.0)])
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", 10.0)

    validate_real_dagger_safety(0.2, 0.25)
