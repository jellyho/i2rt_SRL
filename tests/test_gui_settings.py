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
        gui.runner = None
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


# ------------------------------------------------ no policy -> no rollout, no recording
def _quiet_gui(tmp_path):
    """A deploy GUI with its refresh timers stopped.

    The 10 Hz timer calls into `self.recorder`, and a stand-in that only carries the few
    keys a test cares about crashes the interpreter when the timer reaches the banner/health
    code. Stopping the timers keeps the test to the method under test.
    """
    gui = _deploy_gui(False, tmp_path)
    gui._timer.stop()
    gui._review_timer.stop()
    return gui


class _FakeRunner:
    def __init__(self, connected=False, err=""):
        self._st = {"policy_connected": connected, "last_error": err, "streaming": connected}

    def get_status(self):
        return dict(self._st)

    def shutdown(self):  # closeEvent calls this
        pass


class _FakeRecorder:
    def __init__(self, running=False):
        self.st = {"policy_running": running, "dagger_state": "policy" if running else "stopped"}
        self.calls = []

    def get_status(self):
        return dict(self.st)

    def set_policy_running(self, flag):
        self.calls.append(bool(flag))
        self.st["policy_running"] = bool(flag)

    def shutdown(self):  # closeEvent calls this
        pass


def test_start_is_refused_without_a_policy(tmp_path, qapp, no_camera_scan, monkeypatch):
    """Starting a rollout opens an episode, so with no policy the arms sit still and the
    recording fills up with rollouts nothing ever drove."""
    gui = _quiet_gui(tmp_path)
    try:
        warned = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a))
        gui.runner = _FakeRunner(connected=False, err="policy server offline")
        gui.recorder = _FakeRecorder()

        gui._on_policy_toggle()
        assert gui.recorder.calls == [], "must not start a rollout without a policy"
        assert warned, "the operator has to be told why"
    finally:
        gui.recorder = None
        gui.runner = None
        gui.close()


def test_start_works_once_the_policy_is_up(tmp_path, qapp, no_camera_scan):
    gui = _quiet_gui(tmp_path)
    try:
        gui.runner = _FakeRunner(connected=True)
        gui.recorder = _FakeRecorder()
        gui._on_policy_toggle()
        assert gui.recorder.calls == [True]
    finally:
        gui.recorder = None
        gui.runner = None
        gui.close()


def test_stopping_never_needs_a_policy(tmp_path, qapp, no_camera_scan):
    """A running rollout must always be stoppable, connected or not."""
    gui = _quiet_gui(tmp_path)
    try:
        gui.runner = _FakeRunner(connected=False)
        gui.recorder = _FakeRecorder(running=True)
        gui._on_policy_toggle()
        assert gui.recorder.calls == [False]
    finally:
        gui.recorder = None
        gui.runner = None
        gui.close()


def test_rollout_started_from_the_handle_button_is_stopped(tmp_path, qapp, no_camera_scan):
    """The handle button starts rollouts without going through this UI, so the guard on
    the button alone would not cover it."""
    gui = _quiet_gui(tmp_path)
    try:
        gui.runner = _FakeRunner(connected=False)
        gui.recorder = _FakeRecorder(running=True)
        gui._update_dagger_controls({"policy_running": True, "dagger_state": "policy"})
        assert gui.recorder.calls == [False]
    finally:
        gui.recorder = None
        gui.runner = None
        gui.close()


def test_start_button_is_disabled_while_no_policy(tmp_path, qapp, no_camera_scan):
    gui = _quiet_gui(tmp_path)
    try:
        gui.runner = _FakeRunner(connected=False)
        gui.recorder = _FakeRecorder()
        gui._update_dagger_controls({"policy_running": False, "dagger_state": "stopped"})
        assert gui.policy_btn.isEnabled() is False
        assert "no policy" in gui.policy_btn.toolTip()

        gui.runner = _FakeRunner(connected=True)
        gui._update_dagger_controls({"policy_running": False, "dagger_state": "stopped"})
        assert gui.policy_btn.isEnabled() is True
    finally:
        gui.recorder = None
        gui.runner = None
        gui.close()


def _samples_gui(tmp_path, num_samples=0):
    """A deploy window plus the bridge config it was built from, which is the object the
    runner is later constructed with."""
    from workstation.lerobot_recorder.deploy_gui import DeployGUI
    from workstation.policy_bridge.config import BridgeConfig

    bridge_cfg = BridgeConfig()
    bridge_cfg.num_samples = num_samples
    cfg = RecorderConfig(mock=True, repo_id="test/deploy", root=str(tmp_path))
    return DeployGUI(cfg, bridge_cfg, mode="dagger", record=True), bridge_cfg


def test_no_count_is_shown_while_sampling_is_off(tmp_path, qapp, no_camera_scan):
    """A greyed-out spinner still shows a number, and a number on screen reads as "this many
    samples are being taken" however pale it is. Nothing visible, nothing sampled."""
    gui, bridge_cfg = _samples_gui(tmp_path, num_samples=0)
    try:
        assert gui.samples_check.isChecked() is False
        assert gui.samples_spin.isHidden() is True
        assert bridge_cfg.num_samples == 0
    finally:
        gui.close()


def test_turning_it_on_before_the_rig_starts_is_not_lost(tmp_path, qapp, no_camera_scan):
    """The handler used to write to self.runner.cfg, and the runner is only built when the rig
    starts — so ticking this on the setup page silently did nothing."""
    gui, bridge_cfg = _samples_gui(tmp_path, num_samples=0)
    try:
        assert gui.runner is None, "this test is about the window before the runner exists"
        gui.samples_check.setChecked(True)
        assert bridge_cfg.num_samples == gui.samples_spin.value()
        assert gui.samples_spin.isHidden() is False
    finally:
        gui.close()


def test_turning_it_off_restores_a_plain_request(tmp_path, qapp, no_camera_scan):
    gui, bridge_cfg = _samples_gui(tmp_path, num_samples=8)
    try:
        assert bridge_cfg.num_samples == 8
        gui.samples_check.setChecked(False)
        assert bridge_cfg.num_samples == 0, "0, not 1: the key must leave the wire entirely"
        assert gui.samples_spin.isHidden() is True
    finally:
        gui.close()


def test_the_launch_flag_and_the_checkbox_agree(tmp_path, qapp, no_camera_scan):
    gui, bridge_cfg = _samples_gui(tmp_path, num_samples=6)
    try:
        assert gui.samples_check.isChecked() is True
        assert gui.samples_spin.value() == 6
        assert bridge_cfg.num_samples == 6
    finally:
        gui.close()
