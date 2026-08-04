"""Persistence tests for the recorder and DAgger setup pages."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

from workstation.lerobot_recorder import gui as gui_module
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.gui import RecorderGUI
from workstation.lerobot_recorder.reference_video import ReferenceEpisode


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
    first.reference_opacity.setValue(35)
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
        assert restored.reference_opacity.value() == 35

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


def test_reference_list_defaults_to_first_demonstration_without_off_row(tmp_path, qapp, no_camera_scan, monkeypatch):
    (tmp_path / "data").mkdir()
    episodes = [
        ReferenceEpisode(
            episode=index,
            paths={key: Path(f"{key}-{index}.mp4") for key in ("wrist_left", "agentview", "wrist_right")},
            fps=30,
        )
        for index in (3, 4)
    ]
    monkeypatch.setattr(gui_module, "discover_reference_episodes", lambda *_args: episodes)
    gui = RecorderGUI(
        RecorderConfig(root=str(tmp_path), repo_id="test/data", mock=True),
        settings=QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.IniFormat),
    )
    played = []
    monkeypatch.setattr(
        gui._reference_player,
        "play",
        lambda episode, *, start_paused: played.append((episode.episode, start_paused)),
    )

    try:
        gui._refresh_reference_datasets()

        assert gui.reference_list.count() == 2
        assert gui.reference_list.currentRow() == 0
        assert all("Off" not in gui.reference_list.item(row).text() for row in range(2))
        assert gui.reference_list.item(0).text().startswith("demonstration 0003")
        assert played == [(3, True)]
        assert gui.reference_pause_btn.text() == "Resume reference"
    finally:
        gui.close()


def test_reference_dataset_picker_switches_between_root_subfolders(
    tmp_path, qapp, no_camera_scan, monkeypatch
):
    for name in ("current", "older_runs"):
        (tmp_path / name).mkdir()

    episodes = {
        "current": [
            ReferenceEpisode(
                episode=2,
                paths={key: Path(f"current-{key}.mp4") for key in ("wrist_left", "agentview", "wrist_right")},
                fps=30,
            )
        ],
        "older_runs": [
            ReferenceEpisode(
                episode=9,
                paths={key: Path(f"older-{key}.mp4") for key in ("wrist_left", "agentview", "wrist_right")},
                fps=30,
            )
        ],
    }
    discovered = []

    def discover(root, _camera_keys):
        name = Path(root).name
        discovered.append(name)
        return episodes[name]

    monkeypatch.setattr(gui_module, "discover_reference_episodes", discover)
    gui = RecorderGUI(
        RecorderConfig(root=str(tmp_path), repo_id="operator/current", mock=True),
        settings=QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.IniFormat),
    )
    played = []
    monkeypatch.setattr(
        gui._reference_player,
        "play",
        lambda episode, *, start_paused: played.append((episode.episode, start_paused)),
    )

    try:
        gui._refresh_reference_datasets(preferred=gui._active_dataset_name())

        assert [gui.reference_dataset_combo.itemText(i) for i in range(gui.reference_dataset_combo.count())] == [
            "current",
            "older_runs",
        ]
        assert gui.reference_dataset_combo.currentText() == "current"
        assert gui.reference_list.item(0).text().startswith("demonstration 0002")

        gui.reference_dataset_combo.setCurrentText("older_runs")

        assert discovered[-1] == "older_runs"
        assert gui.reference_list.item(0).text().startswith("demonstration 0009")
        assert played[-1] == (9, True)
    finally:
        gui.close()
