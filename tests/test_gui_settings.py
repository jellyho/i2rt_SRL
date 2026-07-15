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
