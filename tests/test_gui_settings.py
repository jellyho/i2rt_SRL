"""Persistence tests for the recorder and DAgger setup pages."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

from workstation.lerobot_recorder import gui as gui_module
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.gui import RecorderGUI


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def no_camera_scan(monkeypatch):
    monkeypatch.setattr(
        gui_module,
        "detect_cameras",
        lambda _cfg: {"found": 0, "total": 0, "missing": []},
    )


def test_setup_values_are_restored_after_window_closes(tmp_path, qapp, no_camera_scan):
    settings = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.IniFormat)
    first = RecorderGUI(RecorderConfig(mock=True), settings=settings)
    first.repo_combo.setCurrentText("operator/last_dataset")
    first.root_edit.setText(str(tmp_path / "datasets"))
    first.task_combo.setCurrentText("pick up the blue block")
    first.source_combo.setCurrentText("eval")
    first.resume_check.setChecked(True)
    first.rl_check.setChecked(True)
    first.reward_combo.setCurrentText("step")
    first.discount_spin.setValue(0.875)
    first.close()

    restored = RecorderGUI(RecorderConfig(mock=True), settings=settings)
    try:
        assert restored.repo_combo.currentText() == "operator/last_dataset"
        assert restored.root_edit.text() == str(tmp_path / "datasets")
        assert restored.task_combo.currentText() == "pick up the blue block"
        assert restored.source_combo.currentText() == "eval"
        assert restored.resume_check.isChecked() is True
        assert restored.rl_check.isChecked() is True
        assert restored.reward_combo.currentText() == "step"
        assert restored.discount_spin.value() == pytest.approx(0.875)

        # Restoring the form must not advance to collection or start hardware.
        assert restored.stack.currentWidget() is restored.setup_page
        assert restored.recorder is None
    finally:
        restored.close()


def test_record_and_deploy_use_separate_setting_scopes(tmp_path, qapp, no_camera_scan):
    settings = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.IniFormat)
    settings.setValue("setup/record/task", "recording task")
    settings.setValue("setup/deploy/task", "dagger task")

    class DeploySettingsGUI(RecorderGUI):
        SETTINGS_SCOPE = "deploy"

    record = RecorderGUI(RecorderConfig(mock=True), settings=settings)
    deploy = DeploySettingsGUI(RecorderConfig(mock=True), settings=settings)
    try:
        assert record.task_combo.currentText() == "recording task"
        assert deploy.task_combo.currentText() == "dagger task"
    finally:
        record.close()
        deploy.close()


# ------------------------------------------------------------ deploy-only (no recording)
def _deploy_gui(deploy_only, tmp_path, mode="dagger", leader_mirror=None):
    from workstation.lerobot_recorder.deploy_gui import DeployGUI
    from workstation.policy_bridge.config import BridgeConfig

    cfg = RecorderConfig(mock=True, repo_id="test/deploy", root=str(tmp_path))
    return DeployGUI(cfg, BridgeConfig(), mode=mode, record=not deploy_only,
                     leader_mirror=leader_mirror)


def test_deploy_only_selects_the_non_recording_source(tmp_path, qapp, no_camera_scan):
    gui = _deploy_gui(True, tmp_path)
    try:
        assert gui.cfg.record_source == "deploy"
        assert gui.source_combo.currentText() == "deploy"
        assert gui.source_combo.isEnabled() is False  # not switchable mid-session
        assert "not recording" in gui.dagger_box.title().lower()
        assert "nothing is recorded" in gui.hint.text().lower()
    finally:
        gui.close()


def test_deploy_only_hides_every_dataset_control(tmp_path, qapp, no_camera_scan):
    """Nothing that only means something when writing episodes should be on screen.

    Asserts isHidden(), not isVisible(): the window is never show()n in a headless test, so
    isVisible() is False for every widget and would pass no matter what."""
    gui = _deploy_gui(True, tmp_path)
    try:
        for widget in (gui.repo_combo, gui.root_edit, gui.resume_check, gui.rl_check,
                       gui.reward_combo, gui.discount_spin, gui.collect_btn, gui.save_btn,
                       gui.review_box, gui.keep_home_btn):
            assert widget.isHidden() is True, f"{widget} should be hidden in deploy-only"
        # the task field stays: it doubles as the policy prompt
        assert gui.task_combo.isHidden() is False
        # ...and the rollout controls that make deployment usable stay too
        for widget in (gui.policy_btn, gui.intervention_btn, gui.discard_home_btn, gui.estop_btn):
            assert widget.isHidden() is False
    finally:
        gui.close()


def test_dagger_mode_keeps_the_dataset_controls(tmp_path, qapp, no_camera_scan):
    gui = _deploy_gui(False, tmp_path)
    try:
        assert gui.cfg.record_source == "dagger"
        assert gui.source_combo.currentText() == "dagger"
        assert gui.repo_combo.isHidden() is False
        assert gui.keep_home_btn.isHidden() is False
        assert "not recording" not in gui.dagger_box.title().lower()
    finally:
        gui.close()


def test_deploy_only_stats_line_reports_control_not_episodes(tmp_path, qapp, no_camera_scan):
    gui = _deploy_gui(True, tmp_path)
    try:
        gui._update_stats({"dagger_state": "policy", "policy_running": True, "intervention": False})
        assert "not recording" in gui.stats.text()
        assert "policy" in gui.stats.text()
        gui._update_stats({"dagger_state": "intervention", "policy_running": True, "intervention": True})
        assert "human" in gui.stats.text()
    finally:
        gui.close()


def test_record_checkbox_switches_source_and_controls_live(tmp_path, qapp, no_camera_scan):
    """Recording is a setup-page choice, so toggling it must re-shape the page both ways."""
    gui = _deploy_gui(False, tmp_path)  # dagger + record
    try:
        assert gui.cfg.record_source == "dagger"
        assert gui.repo_combo.isHidden() is False

        gui.record_check.setChecked(False)
        assert gui.cfg.record_source == "deploy"
        assert gui.deploy_only is True
        assert gui.repo_combo.isHidden() is True
        assert gui.keep_home_btn.isHidden() is True
        assert gui.discard_home_btn.text() == "Stop + Home"

        gui.record_check.setChecked(True)  # and back again
        assert gui.cfg.record_source == "dagger"
        assert gui.repo_combo.isHidden() is False
        assert gui.keep_home_btn.isHidden() is False
        assert gui.discard_home_btn.text() == "Discard + Home"
    finally:
        gui.close()


def test_leader_mirror_checkbox_is_on_the_collect_page(tmp_path, qapp, no_camera_scan):
    """It has to be reachable DURING a rollout — the setup page is gone by then."""
    gui = _deploy_gui(True, tmp_path)
    try:
        assert gui.mirror_check.isHidden() is False
        assert gui.mirror_check.parent() is gui.dagger_box
        assert gui.mirror_check.isChecked() is True
    finally:
        gui.close()


def test_leader_mirror_follows_the_run_mode(tmp_path, qapp, no_camera_scan):
    """Picking a mode must set mirroring by itself: on for dagger (you mean to take over),
    off for deploy (you are only watching)."""
    gui = _deploy_gui(True, tmp_path, mode="dagger")
    try:
        assert gui.mirror_check.isChecked() is True
        gui.mode_combo.setCurrentText("deploy")
        assert gui.mirror_check.isChecked() is False
        gui.mode_combo.setCurrentText("dagger")
        assert gui.mirror_check.isChecked() is True
    finally:
        gui.close()


def test_mode_and_record_are_independent_axes(tmp_path, qapp, no_camera_scan):
    """All four cells must be reachable — including "dagger but do not save" and
    "plain deploy but do save", which a single picker could not express."""
    gui = _deploy_gui(True, tmp_path, mode="deploy")
    try:
        expected = {
            ("deploy", False): "deploy",
            ("deploy", True): "eval",
            ("dagger", True): "dagger",
            ("dagger", False): "deploy",
        }
        for (mode, record), source in expected.items():
            gui.mode_combo.setCurrentText(mode)
            gui.record_check.setChecked(record)
            assert gui.cfg.record_source == source, (mode, record)
            # mirroring tracks the MODE, not whether we are recording
            assert gui.mirror_check.isChecked() is (mode == "dagger"), (mode, record)
    finally:
        gui.close()


def test_explicit_mirror_flag_overrides_the_mode_default(tmp_path, qapp, no_camera_scan):
    gui = _deploy_gui(True, tmp_path, mode="dagger", leader_mirror=False)
    try:
        assert gui.mirror_check.isChecked() is False  # CLI flag wins at startup
    finally:
        gui.close()


def test_leader_mirror_toggle_is_forwarded_to_the_robot(tmp_path, qapp, no_camera_scan):
    """Toggling mid-rollout must reach the robot; it owns the leader gains."""
    sent = []
    gui = _deploy_gui(True, tmp_path)
    try:
        class FakeRecorder:
            def set_leader_mirror(self, flag):
                sent.append(bool(flag))

        gui.recorder = FakeRecorder()
        gui.mirror_check.setChecked(False)
        gui.mirror_check.setChecked(True)
        assert sent == [False, True]
    finally:
        gui.recorder = None
        gui.close()


def test_runner_status_reports_the_robots_own_mirror_state(tmp_path, qapp, no_camera_scan):
    gui = _deploy_gui(True, tmp_path)
    try:
        gui._update_dagger_controls({"dagger_state": "policy", "leader_mirror": False})
        assert "free (not mirroring)" in gui.runner_status.text()
        gui._update_dagger_controls({"dagger_state": "policy", "leader_mirror": True})
        assert "mirroring the policy" in gui.runner_status.text()
    finally:
        gui.close()
