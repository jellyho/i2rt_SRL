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
