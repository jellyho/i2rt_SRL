"""Keyboard shortcut behavior for the DAgger deployment UI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from workstation.lerobot_recorder.deploy_gui import DeployGUI


class _Button:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.checked = []

    def isEnabled(self) -> bool:
        return self.enabled

    def setChecked(self, value: bool) -> None:
        self.checked.append(bool(value))


def test_space_shortcut_only_engages_estop():
    gui = SimpleNamespace()
    gui.recorder = SimpleNamespace()
    gui.estop_btn = _Button()

    DeployGUI._on_estop_shortcut(gui)
    DeployGUI._on_estop_shortcut(gui)

    assert gui.estop_btn.checked == [True, True]


def test_r_shortcut_requests_human_handoff_only_when_enabled():
    gui = SimpleNamespace()
    gui.recorder = SimpleNamespace()
    gui.rewind_btn = _Button(enabled=True)
    calls = []
    gui._on_rewind = lambda *, resume_policy: calls.append(resume_policy)

    DeployGUI._on_rewind_shortcut(gui)
    gui.rewind_btn.enabled = False
    DeployGUI._on_rewind_shortcut(gui)

    assert calls == [False]
