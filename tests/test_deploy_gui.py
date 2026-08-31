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


def test_the_overlay_root_is_editable_after_the_run_starts(qapp, tmp_path):
    """The setup page is gone once a session starts, and which demonstrations are worth ghosting is
    something the operator works out mid-run -- the same reasoning that put the mirror checkbox on
    the run page. So the overlay root lives in both places."""
    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.deploy_gui import DeployGUI
    from workstation.policy_bridge.deploy_runner import BridgeConfig

    gui = DeployGUI(RecorderConfig(mock=True, root=str(tmp_path)), BridgeConfig(), mode="dagger", record=True)
    try:
        assert gui.reference_root_run_edit.isVisibleTo(gui.reference_box)
    finally:
        gui.close()


def test_the_two_overlay_root_boxes_are_one_value(qapp, tmp_path):
    """Two boxes, one setting. A second source that drifts is how the run page ends up listing from
    a folder the setup page does not name -- and `_reference_root` reads only one of them, so the
    drift would be silent in exactly one direction."""
    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.deploy_gui import DeployGUI
    from workstation.policy_bridge.deploy_runner import BridgeConfig

    gui = DeployGUI(RecorderConfig(mock=True, root=str(tmp_path)), BridgeConfig(), mode="dagger", record=True)
    try:
        gui.reference_root_run_edit.setText("/demos/a")
        gui._on_reference_root_typed("/demos/a")
        assert gui.reference_root_edit.text() == "/demos/a"

        gui.reference_root_edit.setText("/demos/b")
        gui._on_reference_root_typed("/demos/b")
        assert gui.reference_root_run_edit.text() == "/demos/b"
        assert gui._reference_root() == "/demos/b"

        # blank still means "the recording root" -- whatever that box holds, which is a persisted
        # setting rather than the config default, so the test asks the widget.
        gui.reference_root_run_edit.setText("")
        gui._on_reference_root_typed("")
        assert gui._reference_root() == gui.root_edit.text().strip()
    finally:
        gui.close()
