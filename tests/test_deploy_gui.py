"""Keyboard shortcut behavior for the DAgger deployment UI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from workstation.lerobot_recorder.deploy_gui import DeployGUI


class _Button:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def isEnabled(self) -> bool:
        return self.enabled


def test_r_shortcut_requests_return_only_when_enabled():
    calls = []
    gui = SimpleNamespace(
        recorder=SimpleNamespace(),
        return_btn=_Button(enabled=True),
        _on_return_to_pose=lambda: calls.append("return"),
    )

    DeployGUI._on_return_shortcut(gui)
    gui.return_btn.enabled = False
    DeployGUI._on_return_shortcut(gui)

    assert calls == ["return"]


def test_joint_position_display_is_absolute_config_ready():
    values = [index / 10 for index in range(14)]

    text = DeployGUI._format_joint_positions(values)

    assert text == (
        "[0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, "
        "0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000]"
    )
    assert DeployGUI._format_joint_positions(values[:13]) is None
