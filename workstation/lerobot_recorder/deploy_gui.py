"""DAgger deployment UI.

This reuses the expert recorder GUI and adds the policy rollout controls needed
for deployment / DAgger collection.
"""

from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.gui import RecorderGUI
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner


class DeployGUI(RecorderGUI):
    SETTINGS_SCOPE = "deploy"

    def __init__(self, cfg: RecorderConfig, bridge_cfg: BridgeConfig) -> None:
        self.bridge_cfg = bridge_cfg
        self.runner: DeploymentPolicyRunner | None = None
        super().__init__(cfg)
        self.setWindowTitle("YAM · DAgger Deployment")
        idx = self.source_combo.findText("dagger")
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.source_combo.setEnabled(False)
        self.hint.setText(
            "R returns to the configured recovery pose, then hands control to the human · "
            "other rollout controls use UI or handle buttons"
        )
        self._return_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("R"), self)
        self._return_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._return_shortcut.setAutoRepeat(False)
        self._return_shortcut.activated.connect(self._on_return_shortcut)

    def _build_collect_page(self) -> QtWidgets.QWidget:
        page = super()._build_collect_page()

        self.dagger_box = QtWidgets.QGroupBox("DAgger rollout")
        grid = QtWidgets.QGridLayout(self.dagger_box)
        self.dagger_state = QtWidgets.QLabel("state: stopped")
        self.dagger_state.setStyleSheet(f"color:{theme.MUTED};font-size:24px;")
        self.policy_btn = QtWidgets.QPushButton("Start Policy")
        self.policy_btn.clicked.connect(self._on_policy_toggle)
        self.intervention_btn = QtWidgets.QPushButton("Human Intervention")
        self.intervention_btn.setCheckable(True)
        self.intervention_btn.clicked.connect(self._on_intervention_toggle)
        self.return_btn = QtWidgets.QPushButton("Return + Human  [R]")
        self.return_btn.setEnabled(False)
        self.return_btn.clicked.connect(self._on_return_to_pose)
        self.keep_home_btn = QtWidgets.QPushButton("Keep + Home")
        self.keep_home_btn.clicked.connect(lambda: self._on_finish("keep"))
        self.discard_home_btn = QtWidgets.QPushButton("Discard + Home")
        self.discard_home_btn.clicked.connect(lambda: self._on_finish("discard"))
        self.runner_status = QtWidgets.QLabel("policy: not connected")
        self.runner_status.setStyleSheet(f"color:{theme.MUTED};")
        self.joint_pos = QtWidgets.QLineEdit()
        self.joint_pos.setReadOnly(True)
        self.joint_pos.setPlaceholderText("waiting for robot joint state…")
        self.joint_pos.setStyleSheet("font-family:monospace;")
        self.joint_pos.setToolTip(
            "Current follower positions: left 7 values, then right 7. "
            "Paste this array into control.dagger_return_abs_pos."
        )
        self.copy_joint_pos_btn = QtWidgets.QPushButton("Copy joint array")
        self.copy_joint_pos_btn.setEnabled(False)
        self.copy_joint_pos_btn.clicked.connect(self._copy_joint_positions)
        self.button_legend = QtWidgets.QLabel(
            "Handle buttons: left upper = start/stop policy rollout, or fine-grained toggle during intervention; "
            "left lower = human intervention on/off, "
            "right upper = discard + home, right lower = keep + home."
        )
        self.button_legend.setWordWrap(True)
        self.button_legend.setStyleSheet(
            f"background:#2d230b;color:{theme.TEXT};border:1px solid {theme.WARN};"
            "border-radius:8px;padding:10px 12px;font-weight:600;"
        )

        grid.addWidget(self.dagger_state, 0, 0, 1, 5)
        grid.addWidget(self.policy_btn, 1, 0)
        grid.addWidget(self.intervention_btn, 1, 1)
        grid.addWidget(self.return_btn, 1, 2)
        grid.addWidget(self.keep_home_btn, 1, 3)
        grid.addWidget(self.discard_home_btn, 1, 4)
        grid.addWidget(QtWidgets.QLabel("Current follower joints — left 7, then right 7:"), 2, 0, 1, 5)
        grid.addWidget(self.joint_pos, 3, 0, 1, 4)
        grid.addWidget(self.copy_joint_pos_btn, 3, 4)
        grid.addWidget(self.button_legend, 4, 0, 1, 5)
        grid.addWidget(self.runner_status, 5, 0, 1, 5)

        lay = page.layout()
        if isinstance(lay, QtWidgets.QVBoxLayout):
            lay.insertWidget(2, self.dagger_box)
        return page

    def _on_start(self) -> None:
        self.source_combo.setCurrentText("dagger")
        self.bridge_cfg.prompt = self.task_combo.currentText().strip()
        super()._on_start()
        if self.recorder is not None and self.runner is None:
            self.runner = DeploymentPolicyRunner(self.bridge_cfg, self.cfg, self.recorder.get_last_images)
            self.runner.start()

    def _on_policy_toggle(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        self.recorder.set_policy_running(not bool(st.get("policy_running")))

    def _on_intervention_toggle(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        self.recorder.set_intervention(not bool(st.get("intervention")))

    def _on_finish(self, action: str) -> None:
        if self.recorder is not None:
            self.recorder.finish_dagger_run(action)

    def _on_return_to_pose(self) -> None:
        if self.recorder is not None:
            self.recorder.return_to_dagger_pose()

    @staticmethod
    def _format_joint_positions(values: object) -> str | None:
        if values is None:
            return None
        try:
            joints = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        if len(joints) != 14 or not all(math.isfinite(value) for value in joints):
            return None
        return "[" + ", ".join(f"{value:.6f}" for value in joints) + "]"

    def _copy_joint_positions(self) -> None:
        if self.joint_pos.text():
            QtWidgets.QApplication.clipboard().setText(self.joint_pos.text())

    def _on_return_shortcut(self) -> None:
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(
            focus,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
                QtWidgets.QComboBox,
            ),
        ):
            return
        if self.recorder is not None and self.return_btn.isEnabled():
            self._on_return_to_pose()

    def _refresh(self) -> None:
        super()._refresh()
        if self.recorder is not None:
            self._update_dagger_controls(self.recorder.get_status())

    def _update_banner(self, st: dict) -> None:
        if self.estop_btn.isChecked() or st.get("estop"):
            text, color = "■ E-STOP ENGAGED", theme.STATE_COLORS["ERROR"]
        elif not st.get("disk_ok", True):
            text, color = "LOW DISK — not saving", theme.STATE_COLORS["ERROR"]
        elif not st.get("writer_ok", True):
            text, color = "DATASET SAVE FAILED — RESTART REQUIRED", theme.STATE_COLORS["ERROR"]
        elif not (st["cam_ok"] and st.get("robot_ok", True)):
            text, color = "DEVICE FAULT", theme.STATE_COLORS["ERROR"]
        elif st.get("returning"):
            text, color = "RETURNING TO RECOVERY POSE — HUMAN CONTROL NEXT", theme.STATE_COLORS["REVIEW"]
        elif st.get("homing"):
            text, color = "HOMING", theme.STATE_COLORS["REVIEW"]
        elif st.get("recenter_fault"):
            text, color = "LEADER ALIGNMENT TIMED OUT — FOLLOWER HELD", theme.STATE_COLORS["ERROR"]
        elif st.get("leader_recentering"):
            text, color = "HUMAN INTERVENTION — ALIGNING LEADER (RECORDING PAUSED)", theme.STATE_COLORS["REVIEW"]
        elif st.get("intervention"):
            text = "HUMAN INTERVENTION (FINE-GRAINED)" if st.get("fine_grained") else "HUMAN INTERVENTION"
            color = theme.STATE_COLORS["REC"]
        elif st.get("policy_running"):
            text, color = "POLICY RUNNING", theme.STATE_COLORS["ARMED"]
        elif st["armed"]:
            text, color = "COLLECTION ARMED · POLICY STOPPED", theme.STATE_COLORS["ARMED"]
        else:
            text, color = "POLICY STOPPED", theme.STATE_COLORS["IDLE"]
        self.banner.setText(text)
        self.banner.setStyleSheet(theme.banner_style(color))

    def _update_health(self, st: dict) -> None:
        super()._update_health(st)
        runner = self.runner.get_status() if self.runner is not None else {}
        pol = theme.dot(bool(runner.get("policy_connected")))
        stream = "streaming" if runner.get("streaming") else "idle"
        err = runner.get("last_error") or ""
        extra = f" &nbsp;&nbsp; {pol} policy {stream}"
        if err:
            extra += f' <span style="color:{theme.WARN};">({err})</span>'
        self.health.setText(self.health.text() + extra)

    def _update_stats(self, st: dict) -> None:
        self.stats.setText(
            f"episodes {st['episodes']} · interventions {st.get('interventions', 0)} · "
            f"kept {st['kept']} · discarded {st['discarded']} · frames {st['frames']}"
        )

    def _update_dagger_controls(self, st: dict) -> None:
        running = bool(st.get("policy_running"))
        intervention = bool(st.get("intervention"))
        homing = bool(st.get("homing"))
        returning = bool(st.get("returning"))
        return_configured = bool(st.get("dagger_return_configured"))
        return_mode = st.get("dagger_return_mode") or "unconfigured"
        blocked = homing or returning or bool(st.get("estop"))
        self.policy_btn.setText("Stop Policy" if running else "Start Policy")
        self.intervention_btn.setChecked(intervention)
        self.intervention_btn.setText("Human Control" if intervention else "Human Intervention")
        self.return_btn.setText(f"Return ({return_mode}) + Human  [R]")
        self.dagger_state.setText(f"state: {st.get('dagger_state', 'stopped')}")
        joint_text = self._format_joint_positions(st.get("joint_pos"))
        self.joint_pos.setText(joint_text or "")
        self.copy_joint_pos_btn.setEnabled(joint_text is not None)
        for btn in (self.policy_btn, self.intervention_btn, self.keep_home_btn, self.discard_home_btn):
            btn.setEnabled(not blocked)
        self.return_btn.setEnabled(not blocked and return_configured and running and not intervention)
        runner = self.runner.get_status() if self.runner is not None else {}
        horizon = runner.get("action_horizon", self.bridge_cfg.action_horizon)
        img = runner.get("image_size", self.bridge_cfg.image_size)
        self.runner_status.setText(f"policy horizon {horizon} · image {img}px")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.runner is not None:
            self.runner.shutdown()
        super().closeEvent(event)
