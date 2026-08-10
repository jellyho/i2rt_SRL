"""Policy deployment UI.

The policy streams actions to the robot, a human can take over with a handle button, and
the live cameras / e-stop / health strip come from the expert recorder GUI.

A run is described by **two independent choices** on the setup page, because recording a
deployment is not the same thing as doing DAgger:

* **mode** — what you are here to do, which is what decides *leader mirroring*:

  - ``Deploy`` — watch the policy work. The leader hangs free, so the handles do not fly
    around while the arm moves.
  - ``DAgger`` — correct the policy. The leader mirrors the follower, so grabbing a handle
    to take over starts from the arm's current pose instead of yanking it somewhere else.

* **record** — whether this run lands in a dataset. Independent of mode: you may want the
  data from a plain deployment (an eval run) and you may want to practise takeovers
  without saving anything.

The two combine into the recorder's ``record_source``; all four cells are ordinary,
already-supported behaviour:

===========  ==================  =========================================
mode         record              source
===========  ==================  =========================================
Deploy       off                 ``deploy``  — nothing written
Deploy       on                  ``eval``    — Start/Stop collection bounds one episode
DAgger       on                  ``dagger``  — one episode per rollout, keep/discard
DAgger       off                 ``deploy``  — mirroring on, still nothing written
===========  ==================  =========================================

Mirroring follows the mode automatically but stays overridable from the collect page,
since it is the one setting an operator may want to change mid-rollout.

Everything safety-related is unaffected: e-stop, human takeover, and homing are owned by
the robot-side controller, not by this UI.
"""

from __future__ import annotations

import logging

from PyQt5 import QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.gui import PickerComboBox, RecorderGUI
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner

logger = logging.getLogger(__name__)


class DeployGUI(RecorderGUI):
    SETTINGS_SCOPE = "deploy"

    #: (mode, record) -> recorder record_source. Deployment never uses "teleop": no engage
    #: gate is involved when a policy is driving the followers.
    SOURCES = {
        ("deploy", False): "deploy",
        ("deploy", True): "eval",
        ("dagger", False): "deploy",
        ("dagger", True): "dagger",
    }
    #: Mirroring is a property of the MODE, not of recording — it exists so a takeover
    #: starts from the arm's pose, which is the point of DAgger and pointless when you are
    #: only watching. Picking a mode re-applies this; the collect-page checkbox overrides
    #: it until the mode changes again.
    MIRROR_BY_MODE = {"deploy": False, "dagger": True}

    def __init__(
        self,
        cfg: RecorderConfig,
        bridge_cfg: BridgeConfig,
        *,
        mode: str = "dagger",
        record: bool = True,
        leader_mirror: bool | None = None,
    ) -> None:
        self.bridge_cfg = bridge_cfg
        self.runner: DeploymentPolicyRunner | None = None
        mode = mode if mode in self.MIRROR_BY_MODE else "dagger"
        cfg.record_source = self.SOURCES[(mode, bool(record))]
        super().__init__(cfg)
        self.setWindowTitle("YAM · Policy Deployment")

        for source in set(self.SOURCES.values()):
            if self.source_combo.findText(source) < 0:
                self.source_combo.addItem(source)
        # Fully derived from mode + record now, so it would only be dead UI to stare at.
        self.source_combo.setEnabled(False)
        self._set_form_row_visible(self.source_combo, False)

        self.mode_combo = PickerComboBox()
        self.mode_combo.addItems(["deploy", "dagger"])
        self.mode_combo.setToolTip(
            "deploy: watch the policy work — the leader hangs free.\n"
            "dagger: correct the policy — the leader mirrors the follower so a takeover "
            "starts from the arm's current pose."
        )
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.currentTextChanged.connect(lambda *_: self._sync_run_mode())

        self.record_check = QtWidgets.QCheckBox("Record this run to a dataset")
        self.record_check.setToolTip(
            "Off: no dataset is created, opened, or resumed — nothing is written.\n"
            "On (deploy): Start/Stop collection bounds one episode — an eval log.\n"
            "On (dagger): one episode per rollout, ended with keep/discard."
        )
        self.record_check.setChecked(bool(record))
        self.record_check.toggled.connect(lambda *_: self._sync_run_mode())

        form = self._setup_form()
        if form is not None:
            form.addRow("mode", self.mode_combo)
            form.addRow("", self.record_check)

        self._sync_run_mode()  # also applies the mode's mirroring default
        if leader_mirror is not None:  # explicit CLI flag wins over the mode default
            self.mirror_check.setChecked(leader_mirror)

    # ------------------------------------------------------------- record on/off
    def _setup_form(self) -> QtWidgets.QFormLayout | None:
        for box in self.setup_page.findChildren(QtWidgets.QGroupBox):
            if isinstance(box.layout(), QtWidgets.QFormLayout):
                return box.layout()
        return None

    @property
    def run_mode(self) -> str:
        """"deploy" (watch) or "dagger" (correct) — the axis that decides mirroring."""
        return self.mode_combo.currentText()

    def _sync_run_mode(self) -> None:
        """Re-derive the record source and re-shape the page for (mode, record).

        The two axes genuinely do not move together: whether a dataset exists is the
        `record` checkbox, while the per-rollout keep/discard verdict only applies in
        dagger (in eval the episode is accepted from the review panel after Stop
        collection, not from a handle button)."""
        mode = self.run_mode
        recording = self.record_check.isChecked()
        source = self.SOURCES[(mode, recording)]
        self.cfg.record_source = source
        self.source_combo.setCurrentText(source)
        per_rollout_verdict = source == "dagger"
        # Mirroring follows the MODE. setChecked fires the toggle handler, which pushes it
        # to the robot, so switching mode mid-session re-applies it there too.
        self.mirror_check.setChecked(self.MIRROR_BY_MODE[mode])

        for widget in (self.repo_combo, self.root_edit, self.resume_check, self.rl_check,
                       self.reward_combo, self.discount_spin):
            self._set_form_row_visible(widget, recording)
        self.review_box.setVisible(recording and self.cfg.review_before_save)
        self.collect_btn.setVisible(recording)
        self.save_btn.setVisible(recording)
        self.keep_home_btn.setVisible(per_rollout_verdict)
        self.discard_home_btn.setText("Discard + Home" if per_rollout_verdict else "Stop + Home")
        self.dagger_box.setTitle(
            {"deploy": "Policy rollout (not recording)",
             "eval": "Policy rollout (logging to the dataset)",
             "dagger": "DAgger rollout"}[source]
        )
        self.hint.setText(
            {"deploy": "policy start/stop and human intervention can use the UI or the handle "
                       "buttons · NOTHING is recorded in this mode",
             "eval": "Start collection begins ONE episode; Stop collection ends it · "
                     "policy/intervention can use the UI or the handle buttons",
             "dagger": "space toggles collection · policy/intervention/keep/discard can use UI "
                       "or handle buttons"}[source]
        )
        self.button_legend.setText(
            "Handle buttons: left upper = start/stop policy rollout, or fine-grained toggle during "
            "intervention; left lower = human intervention on/off, "
            + ("right upper = discard + home, right lower = keep + home."
               if per_rollout_verdict else
               "right upper or lower = stop the rollout and home."
               + ("" if recording else " Nothing is recorded."))
        )

    def _on_mirror_toggled(self, flag: bool) -> None:
        """Apply live: the operator may want the handles to stop moving mid-session."""
        if self.recorder is not None:
            self.recorder.set_leader_mirror(bool(flag))

    def _set_form_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        """Show/hide a QFormLayout row *including its label*.

        Qt has no row-level visibility, so the label is looked up by matching the field
        widget; hiding the field alone would leave a dangling "repo_id" caption."""
        form = self._setup_form()
        if form is not None:
            for row in range(form.rowCount()):
                item = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
                if item is not None and item.widget() is widget:
                    label = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
                    if label is not None and label.widget() is not None:
                        label.widget().setVisible(visible)
                    break
        widget.setVisible(visible)

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
        self.keep_home_btn = QtWidgets.QPushButton("Keep + Home")
        self.keep_home_btn.clicked.connect(lambda: self._on_finish("keep"))
        self.discard_home_btn = QtWidgets.QPushButton("Discard + Home")
        self.discard_home_btn.clicked.connect(lambda: self._on_finish("discard"))
        self.runner_status = QtWidgets.QLabel("policy: not connected")
        self.runner_status.setStyleSheet(f"color:{theme.MUTED};")
        # Lives on the COLLECT page, not the setup page: the operator decides mid-session
        # that the handles thrashing around is distracting, and the setup page is gone by
        # then. Applying it live is safe — it only changes what the leader does next tick.
        self.mirror_check = QtWidgets.QCheckBox("Leader mirrors the policy while it drives")
        self.mirror_check.setToolTip(
            "On: the leader handles track the follower, so a takeover starts from the arm's "
            "current pose.\nOff: the handles hang free — but on takeover the follower travels "
            "(rate-limited) to wherever the leader is."
        )
        self.mirror_check.toggled.connect(self._on_mirror_toggled)
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

        grid.addWidget(self.dagger_state, 0, 0, 1, 4)
        grid.addWidget(self.policy_btn, 1, 0)
        grid.addWidget(self.intervention_btn, 1, 1)
        grid.addWidget(self.keep_home_btn, 1, 2)
        grid.addWidget(self.discard_home_btn, 1, 3)
        grid.addWidget(self.mirror_check, 2, 0, 1, 4)
        grid.addWidget(self.button_legend, 3, 0, 1, 4)
        grid.addWidget(self.runner_status, 4, 0, 1, 4)

        lay = page.layout()
        if isinstance(lay, QtWidgets.QVBoxLayout):
            lay.insertWidget(2, self.dagger_box)
        return page

    @property
    def deploy_only(self) -> bool:
        """True when this run records nothing — regardless of which mode it is in."""
        return not self.record_check.isChecked()

    def _on_start(self) -> None:
        self.source_combo.setCurrentText("deploy" if self.deploy_only else "dagger")
        self.bridge_cfg.prompt = self.task_combo.currentText().strip()
        super()._on_start()
        if self.recorder is not None:
            # Push the mirror choice once the link exists; the bridge latches it and
            # re-applies on reconnect, so the robot never silently reverts to its default.
            self.recorder.set_leader_mirror(self.mirror_check.isChecked())
        if self.recorder is not None and self.runner is None:
            self.runner = DeploymentPolicyRunner(self.bridge_cfg, self.cfg, self.recorder.get_last_images)
            # The sampled chunks are logged beside the dataset, so a rendered video can show what
            # the policy predicted AT THE TIME rather than what the current checkpoint would.
            self.recorder.set_samples_source(self.runner.get_samples)
            self.runner.start()

    @property
    def policy_ready(self) -> bool:
        """Whether the policy server is actually reachable.

        Starting a rollout is what opens a DAgger episode, so without this the operator can
        start one against a policy that is not there: the arms sit still, and a recording
        session quietly fills up with episodes in which nothing ever acted."""
        return bool(self.runner is not None and self.runner.get_status().get("policy_connected"))

    def _on_policy_toggle(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        running = bool(st.get("policy_running"))
        if not running and not self.policy_ready:
            err = (self.runner.get_status().get("last_error") if self.runner else "") or "not connected yet"
            QtWidgets.QMessageBox.warning(
                self,
                "No policy",
                f"The policy server at {self.bridge_cfg.policy_host}:{self.bridge_cfg.policy_port} "
                f"is not connected ({err}).\n\n"
                "Starting now would open an episode that no policy ever drives — the arms "
                "would sit still and the rollout would be recorded anyway.\n\n"
                "Start the policy server first; this reconnects on its own.",
            )
            return
        self.recorder.set_policy_running(not running)

    def _on_intervention_toggle(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        self.recorder.set_intervention(not bool(st.get("intervention")))

    def _on_finish(self, action: str) -> None:
        if self.recorder is not None:
            self.recorder.finish_dagger_run(action)

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
        if self.deploy_only:
            # The base strip is all about the dataset writer (workers, saved, queue), none
            # of which exists here — build a link-oriented one instead.
            self.health.setText(
                f"{theme.dot(bool(st.get('robot_ok')))} robot &nbsp;&nbsp; "
                f"{theme.dot(bool(st.get('cam_ok')))} cameras"
            )
        else:
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
        if self.deploy_only:
            # No episode/kept/discarded counters exist here — report what a watch-only run
            # actually has: whether the policy is driving, and whether a human has taken over.
            state = st.get("dagger_state", "stopped")
            who = "human" if st.get("intervention") else ("policy" if st.get("policy_running") else "—")
            self.stats.setText(f"not recording · rollout {state} · in control: {who}")
            return
        self.stats.setText(
            f"episodes {st['episodes']} · interventions {st.get('interventions', 0)} · "
            f"kept {st['kept']} · discarded {st['discarded']} · frames {st['frames']}"
        )

    def _update_dagger_controls(self, st: dict) -> None:
        running = bool(st.get("policy_running"))
        intervention = bool(st.get("intervention"))
        homing = bool(st.get("homing"))
        blocked = homing or bool(st.get("estop"))
        self.policy_btn.setText("Stop Policy" if running else "Start Policy")
        self.intervention_btn.setChecked(intervention)
        self.intervention_btn.setText("Human Control" if intervention else "Human Intervention")
        self.dagger_state.setText(f"state: {st.get('dagger_state', 'stopped')}")
        buttons = [self.policy_btn, self.intervention_btn, self.discard_home_btn]
        if not self.deploy_only:
            buttons.append(self.keep_home_btn)
        for btn in buttons:
            btn.setEnabled(not blocked)

        ready = self.policy_ready
        # Starting a rollout is what opens an episode, so without a policy it would record
        # one in which nothing ever acted.
        self.policy_btn.setEnabled(not blocked and (running or ready))
        self.policy_btn.setToolTip(
            "" if ready else f"no policy at {self.bridge_cfg.policy_host}:{self.bridge_cfg.policy_port}"
        )
        if running and not ready:
            # The handle button starts rollouts too, and it does not go through this UI.
            # Close it here rather than let the recording keep going: the operator asked
            # for a policy rollout and there is no policy.
            logger.warning(
                "rollout is running with no policy connected (%s) — stopping it",
                (self.runner.get_status().get("last_error") if self.runner else "") or "not connected",
            )
            self.recorder.set_policy_running(False)
        runner = self.runner.get_status() if self.runner is not None else {}
        # 0 until the first chunk arrives — the policy decides, we do not configure it.
        horizon = runner.get("action_horizon", 0)
        img = runner.get("image_size", self.bridge_cfg.image_size)
        # Report the robot's OWN mirror state, not the checkbox: they can differ for a tick
        # after a toggle or a reconnect, and the robot is the one actually driving the leader.
        leader = "mirroring the policy" if st.get("leader_mirror", True) else "free (not mirroring)"
        chunk = f"chunk {horizon}" if horizon else "chunk —"
        self.runner_status.setText(f"policy {chunk} · image {img}px · leader {leader}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.runner is not None:
            self.runner.shutdown()
        super().closeEvent(event)
