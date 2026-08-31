"""The deploy page's on-screen instructions match what the buttons actually do."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_legend_on_screen_matches_the_buttons(qapp):
    """The legend is set twice -- once at construction and once in _sync_run_mode, which runs on
    every mode change and therefore WINS. When the mapping moved to success/fail only the
    constructor's copy was updated, so the screen told the operator that right upper discards while
    the robot recorded it as a success, and said nothing about left upper having become discard.

    Wrong instructions on screen are worse than none here: the button that used to mean "stop" now
    throws the episode away, and there is no undo."""
    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.deploy_gui import DeployGUI
    from workstation.policy_bridge.deploy_runner import BridgeConfig

    gui = DeployGUI(RecorderConfig(mock=True), BridgeConfig(), mode="dagger", record=True)
    try:
        legend = gui.button_legend.text().lower()
        assert "discard" in legend and "left upper" in legend, "left upper discards; say so"
        assert "success" in legend and "failure" in legend, "the right pair is the verdict"
        # the mapping this replaced, which must not survive anywhere in the text
        assert "keep + home" not in legend, "keep/discard was the OLD pair"
        assert gui.success_home_btn.text() == "Success + Home"
        assert gui.fail_home_btn.text() == "Failure + Home"
    finally:
        gui.close()
