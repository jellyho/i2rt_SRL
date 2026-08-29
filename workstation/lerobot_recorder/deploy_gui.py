"""Policy deployment UI.

The policy streams actions to the robot, a human can take over with a handle button, and
the live cameras / e-stop / health strip come from the expert recorder GUI.

A run is described by **two independent choices** on the setup page, because recording a
deployment is not the same thing as doing DAgger:

* **mode** — the action source, which also decides *leader mirroring*:

  - ``policy`` (default) — watch a live policy work. The leader hangs free, so the handles do
    not fly around while the arm moves.
  - ``dagger`` — correct the policy. The leader mirrors the follower, so grabbing a handle to
    take over starts from the arm's current pose instead of yanking it somewhere else.
  - ``dataset`` — replay a recorded dataset (actions from a chosen episode instead of a live
    policy). Pick the episode on the run page's past-demonstration panel; watch-only.

* **record** — whether this run lands in a dataset. Independent of mode (except ``dataset``,
  which is watch-only): you may want the data from a plain deployment (an eval run) and you
  may want to practise takeovers without saving anything.

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
from workstation.lerobot_recorder.chunk_plot import ChunkLengthPlot
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.gui import PickerComboBox, RecorderGUI
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner

logger = logging.getLogger(__name__)


class DeployGUI(RecorderGUI):
    SETTINGS_SCOPE = "deploy"

    #: Action source (mode) x record -> the recorder's record_source. The VALUES ("deploy"/"eval"/
    #: "dagger") are recorder concepts and stay; only the mode KEYS name the action source:
    #: "policy" (watch a live policy), "dagger" (correct it), "dataset" (replay a recording).
    SOURCES = {
        ("policy", False): "deploy",
        ("policy", True): "eval",
        ("dagger", False): "deploy",
        ("dagger", True): "dagger",
        # dataset replay is watch-only and never records, so its record source is moot ("deploy").
        ("dataset", False): "deploy",
        ("dataset", True): "deploy",
    }
    #: Mirroring is a property of the MODE, not of recording — it exists so a takeover
    #: starts from the arm's pose, which is the point of DAgger and pointless when you are
    #: only watching. Picking a mode re-applies this; the collect-page checkbox overrides
    #: it until the mode changes again. policy/dataset are watch-only: leader free.
    MIRROR_BY_MODE = {"policy": False, "dagger": True, "dataset": False}

    def __init__(
        self,
        cfg: RecorderConfig,
        bridge_cfg: BridgeConfig,
        *,
        mode: str = "policy",
        record: bool = True,
        leader_mirror: bool | None = None,
    ) -> None:
        self.bridge_cfg = bridge_cfg
        self.runner: DeploymentPolicyRunner | None = None
        mode = mode if mode in self.MIRROR_BY_MODE else "policy"
        cfg.record_source = self.SOURCES[(mode, bool(record))]
        # Eval recording is send-driven: one frame per action the runner pushes to the robot, at the
        # bridge control rate. The dataset fps must be that send rate, NOT the 60 Hz camera loop --
        # otherwise ~30 Hz frames get 60 fps timestamps and the recording plays back at ~2x. (The
        # record loop drains the whole send queue each tick, so 1 frame == 1 action holds regardless
        # of the loop rate; this only fixes the timestamps.)
        if cfg.record_source == "eval":
            cfg.fps = max(1, round(bridge_cfg.rate_hz))
        super().__init__(cfg)
        self.setWindowTitle("YAM · Policy Deployment")

        for source in set(self.SOURCES.values()):
            if self.source_combo.findText(source) < 0:
                self.source_combo.addItem(source)
        # Fully derived from mode + record now, so it would only be dead UI to stare at.
        self.source_combo.setEnabled(False)
        self._set_form_row_visible(self.source_combo, False)

        self.mode_combo = PickerComboBox()
        # "policy" is the default; "dataset" sources the actions from a recorded dataset instead of
        # a live policy -- the episode is picked on the run page's reference panel.
        self.mode_combo.addItems(["policy", "dagger", "dataset"])
        self.mode_combo.setToolTip(
            "policy: watch a live policy work — the leader hangs free.\n"
            "dagger: correct the policy — the leader mirrors the follower so a takeover "
            "starts from the arm's current pose.\n"
            "dataset: drive the robot from a recorded dataset (pick the episode on the run page); "
            "watch-only, nothing recorded."
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
        """ "policy" (watch), "dagger" (correct), or "dataset" (replay a recording)."""
        return self.mode_combo.currentText()

    def _sync_run_mode(self) -> None:
        """Re-derive the record source and re-shape the page for (mode, record).

        The two axes genuinely do not move together: whether a dataset exists is the
        `record` checkbox, while the per-rollout keep/discard verdict only applies in
        dagger (in eval the episode is accepted from the review panel after Stop
        collection, not from a handle button)."""
        mode = self.run_mode
        # Replay is watch-only: no recording is possible, so the record checkbox is forced off and
        # disabled. `replay_mode` tells the runner to source actions from a dataset (the specific
        # episode is chosen on the run page's reference panel -> set_replay_source).
        replaying = mode == "dataset"
        self.bridge_cfg.replay_mode = replaying
        self.record_check.setEnabled(not replaying)
        if replaying and self.record_check.isChecked():
            self.record_check.setChecked(False)  # re-enters here with recording False
            return
        recording = self.record_check.isChecked()
        source = self.SOURCES[(mode, recording)]
        self.cfg.record_source = source
        self.source_combo.setCurrentText(source)
        per_rollout_verdict = source == "dagger"
        # Mirroring follows the MODE. setChecked fires the toggle handler, which pushes it
        # to the robot, so switching mode mid-session re-applies it there too.
        self.mirror_check.setChecked(self.MIRROR_BY_MODE[mode])

        for widget in (
            self.repo_combo,
            self.resume_check,
            self.rl_check,
            self.reward_combo,
            self.discount_spin,
        ):
            self._set_form_row_visible(widget, recording)
        # `root` is the datasets folder, and it is needed in EVERY deploy mode -- not just when
        # recording. It is where the run page's reference panel lists past demonstrations from, and
        # that panel is the overlay in policy/dagger (and the action source in dataset replay). A
        # watch-only policy run still overlays a demonstration, so hiding the field there left no
        # way to point the overlay at the datasets folder holding them.
        self._set_form_row_visible(self.root_edit, True)
        # Same reasoning, one level further: the overlay's datasets need not sit beside the ones
        # being recorded. Blank keeps the old behaviour (overlay reads the recording root).
        self._set_form_row_visible(self.reference_root_edit, True)
        self.root_edit.setToolTip(
            "Folder holding one subfolder per dataset.\n"
            + (
                "New episodes are written here, and the run page's past-demonstration "
                "overlay lists demonstrations from here."
                if recording
                else "The run page's past-demonstration panel lists episodes from here — "
                + ("the one to replay." if replaying else "the one to overlay on the live view.")
            )
        )
        self.review_box.setVisible(recording and self.cfg.review_before_save)
        self.collect_btn.setVisible(recording)
        self.save_btn.setVisible(recording)
        # The past-demonstration panel's playback ("Resume reference") and "Refresh demonstrations"
        # buttons are not useful in deployment: in policy/dagger the panel is just a first-frame
        # alignment overlay, and in replay the episode is chosen from the LIST while the ghost
        # auto-follows the rollout (_sync_replay_overlay) and the dataset list is refreshed on Start.
        # So hide both in every deploy mode.
        for btn in (getattr(self, "reference_pause_btn", None), getattr(self, "reference_refresh_btn", None)):
            if btn is not None:
                btn.setVisible(False)
        self.success_home_btn.setVisible(per_rollout_verdict)
        self.fail_home_btn.setVisible(per_rollout_verdict)
        self.discard_home_btn.setText("Discard + Home" if per_rollout_verdict else "Stop + Home")
        self.dagger_box.setTitle(
            {
                "deploy": "Policy rollout (not recording)",
                "eval": "Policy rollout (logging to the dataset)",
                "dagger": "DAgger rollout",
            }[source]
        )
        self.hint.setText(
            {
                "deploy": "policy start/stop and human intervention can use the UI or the handle "
                "buttons · NOTHING is recorded in this mode",
                "eval": "Start collection begins ONE episode; Stop collection ends it · "
                "policy/intervention can use the UI or the handle buttons",
                "dagger": "space toggles collection · policy/intervention/keep/discard can use UI or handle buttons",
            }[source]
        )
        self.button_legend.setText(
            "Handle buttons: left upper = start/stop policy rollout, or fine-grained toggle during "
            "intervention; left lower = human intervention on/off, "
            + (
                "right upper = discard + home, right lower = keep + home."
                if per_rollout_verdict
                else "right upper or lower = stop the rollout and home."
                + ("" if recording else " Nothing is recorded.")
            )
        )
        # Recording just changed -> refresh the dataset line so a watch-only run stops warning
        # about overwriting (see _will_record).
        if hasattr(self, "setup_status"):
            self._update_setup_status()

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
        # A rollout ends with a verdict, not with keep-or-throw-away. Both outcomes are data: a
        # critic is fitted on successes AND failures, so a run that fails is worth recording, and
        # only a botched one (robot fault, wrong scene) is worth discarding.
        self.success_home_btn = QtWidgets.QPushButton("Success + Home")
        self.success_home_btn.clicked.connect(lambda: self._on_finish("success"))
        self.fail_home_btn = QtWidgets.QPushButton("Failure + Home")
        self.fail_home_btn.clicked.connect(lambda: self._on_finish("fail"))
        self.discard_home_btn = QtWidgets.QPushButton("Discard + Home")
        self.discard_home_btn.clicked.connect(lambda: self._on_finish("discard"))
        # Kept as an alias so older call sites (and tests) still resolve.
        self.keep_home_btn = self.success_home_btn
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
            "Handle buttons: left upper = start the rollout, then DISCARD it (fine-grained toggle "
            "during intervention); left lower = human intervention on/off; "
            "right upper = success + home, right lower = failure + home."
        )
        self.button_legend.setWordWrap(True)
        self.button_legend.setStyleSheet(
            f"background:#2d230b;color:{theme.TEXT};border:1px solid {theme.WARN};"
            "border-radius:8px;padding:10px 12px;font-weight:600;"
        )

        # How many steps each reply carried. The horizon is read off every reply rather than
        # configured, so it can change per replan -- and it decides how often the policy is
        # re-queried (and, running synchronously, how often the arm stalls).
        self.chunk_plot = ChunkLengthPlot()

        grid.addWidget(self.dagger_state, 0, 0, 1, 4)
        grid.addWidget(self.policy_btn, 1, 0)
        grid.addWidget(self.intervention_btn, 1, 1)
        grid.addWidget(self.success_home_btn, 1, 2)
        grid.addWidget(self.fail_home_btn, 1, 3)
        grid.addWidget(self.discard_home_btn, 1, 4)
        grid.addWidget(self.mirror_check, 2, 0, 1, 4)
        grid.addWidget(self.chunk_plot, 3, 0, 1, 4)
        grid.addWidget(self.button_legend, 4, 0, 1, 4)
        grid.addWidget(self.runner_status, 5, 0, 1, 4)

        lay = page.layout()
        if isinstance(lay, QtWidgets.QVBoxLayout):
            lay.insertWidget(2, self.dagger_box)
        return page

    @property
    def deploy_only(self) -> bool:
        """True when this run records nothing — regardless of which mode it is in."""
        return not self.record_check.isChecked()

    def _will_record(self) -> bool:
        """A deploy run writes a dataset only when 'Record this run' is on (dataset-replay mode
        forces it off). Drives the setup status so it never warns about overwriting in a
        watch-only run -- START skips the whole dataset negotiation there anyway."""
        return bool(getattr(self, "record_check", None) is not None and self.record_check.isChecked())

    def _on_start(self) -> None:
        # The source is derived from mode + record by _sync_run_mode, which has already put it in
        # the combo that RecorderGUI._on_start reads. Re-deriving it here (this used to force
        # "dagger" for any recording run) silently overrode the mode: a `policy` + record run was
        # started as `dagger`, so it recorded from the teleop gate at the loop rate instead of one
        # frame per action sent -- which put every inference stall into the dataset as a frozen
        # command, and left the whole send-driven path dead.
        self.bridge_cfg.prompt = self.task_combo.currentText().strip()
        super()._on_start()
        # Re-list the past-demonstration datasets from the (now-applied) root, so replay can pick an
        # episode without a manual "Refresh demonstrations" button (which is hidden -- see _sync_run_mode).
        if self.recorder is not None:
            self._refresh_reference_datasets()
        if self.recorder is not None:
            # Push the mirror choice once the link exists; the bridge latches it and
            # re-applies on reconnect, so the robot never silently reverts to its default.
            self.recorder.set_leader_mirror(self.mirror_check.isChecked())
        if self.recorder is not None and self.runner is None:
            self.runner = DeploymentPolicyRunner(self.bridge_cfg, self.cfg, self.recorder.get_last_images)
            # Whatever per-step data the policy declares at handshake becomes dataset columns.
            # Wired before the first frame, which is when the dataset schema is fixed.
            self.runner.on_connected = lambda: self.recorder.set_extra_features(
                self.runner.extra_features(), self.runner.get_extras
            )
            # Record exactly one eval frame per action the runner sends (frames == executed actions).
            # A no-op unless armed in eval mode, so it is harmless in policy-watch / dagger / dataset.
            self.runner.on_action_sent = self.recorder.note_action_sent
            # One eval episode per rollout: the runner knows when it stopped driving.
            self.runner.on_rollout_end = self.recorder.end_rollout
            self.runner.start()

    @property
    def policy_ready(self) -> bool:
        """Whether the policy server is actually reachable.

        Starting a rollout is what opens a DAgger episode, so without this the operator can
        start one against a policy that is not there: the arms sit still, and a recording
        session quietly fills up with episodes in which nothing ever acted."""
        return bool(self.runner is not None and self.runner.get_status().get("policy_connected"))

    def _on_reference_selected(self, row: int) -> None:
        """Selecting an episode in the past-demonstration panel also points the REPLAY source at it
        (when in replay mode) -- the overlay dataset+episode IS the action source, so there is one
        place to pick, not two. In deploy/dagger this just drives the overlay, as before."""
        super()._on_reference_selected(row)
        if self.run_mode == "dataset" and self.runner is not None and 0 <= row < len(self._reference_episodes):
            episode = self._reference_episodes[row]
            self.runner.set_replay_source(self.reference_dataset_combo.currentText().strip(), episode.episode)

    def _on_policy_toggle(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        running = bool(st.get("policy_running"))
        # In replay, once it is running the toggle is PAUSE/RESUME -- a pure send-gate on the runner
        # (the robot holds while paused). policy_running is left on, so the robot side never runs its
        # start/stop logic (no gripper close on resume). Starting/stopping the run is the other
        # branch below (the first press starts it; homing/e-stop stops it).
        if self.run_mode == "dataset" and running and self.runner is not None:
            self.runner.set_replay_paused(not self.runner.replay_paused)
            return
        if not running and not self.policy_ready:
            err = (self.runner.get_status().get("last_error") if self.runner else "") or "not connected yet"
            if self.run_mode == "dataset":
                title, body = (
                    "No episode to replay",
                    "Pick an episode in the past-demonstration panel first — that dataset episode "
                    "is what drives the robot in replay mode. If the list is empty, there is no "
                    "dataset under the session root to replay.",
                )
            else:
                title, body = (
                    "No policy",
                    f"The policy server at {self.bridge_cfg.policy_host}:{self.bridge_cfg.policy_port} "
                    f"is not connected ({err}).\n\n"
                    "Starting now would open an episode that no policy ever drives — the arms "
                    "would sit still and the rollout would be recorded anyway.\n\n"
                    "Start the policy server first; this reconnects on its own.",
                )
            QtWidgets.QMessageBox.warning(self, title, body)
            return
        if self.run_mode == "dataset" and self.runner is not None:
            self.runner.set_replay_paused(False)  # (re)starting a replay plays, not paused
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
        if self.runner is not None:
            self.chunk_plot.set_values(self.runner.chunk_lengths())
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
                f"{theme.dot(bool(st.get('robot_ok')))} robot &nbsp;&nbsp; {theme.dot(bool(st.get('cam_ok')))} cameras"
            )
        else:
            super()._update_health(st)
        runner = self.runner.get_status() if self.runner is not None else {}
        self._sync_replay_overlay(runner, st)
        pol = theme.dot(bool(runner.get("policy_connected")))
        stream = "streaming" if runner.get("streaming") else "idle"
        err = runner.get("last_error") or ""
        # Name what answered on the policy port. openpi, ACRFT and a LeRobot checkpoint all speak
        # this wire but read different observations, and the wrong one still streams a chunk of
        # the right shape -- so "policy idle" alone cannot tell an operator they are about to run
        # a rollout against the wrong server.
        who = runner.get("policy_name") or ""
        extra = f" &nbsp;&nbsp; {pol} policy {stream}"
        if who:
            extra += f' <span style="color:{theme.MUTED};">· {who}</span>'
        if err:
            extra += f' <span style="color:{theme.WARN};">({err})</span>'
        self.health.setText(self.health.text() + extra)

    # ------------------------------------------------------------- replay overlay
    def _sync_replay_overlay(self, runner: dict, st: dict) -> None:
        """Play the past-demonstration overlay in lock-step with the replayed rollout.

        The ghost plays at the ARM's frame rate -- one recorded frame per control tick
        (``bridge_cfg.rate_hz``), NOT the recording's raw fps, which would run ahead of the arm.
        It follows the rollout: playing while streaming, frozen while paused (matching the arm's
        hold), and rewound to the first frame when the rollout is stopped/homed instead of left
        frozen mid-trajectory. The decoder can't seek, so this is a rate match, not a per-frame
        lock; if the control loop can't sustain rate_hz the two can still drift over a long episode.
        Only in replay -- for a live policy there is nothing to line the ghost up to.
        """
        if runner.get("policy_framework") != "dataset-replay":
            self._replay_overlay_key = None
            self._replay_overlay_running = False
            return

        dataset, episode = runner.get("replay_dataset") or "", int(runner.get("replay_episode", -1))
        key = (dataset, episode)
        if dataset and episode >= 0 and key != getattr(self, "_replay_overlay_key", None):
            self._replay_overlay_key = key
            self._reference_player.set_rate(max(1.0, float(self.bridge_cfg.rate_hz)))
            self._select_reference_episode(dataset, episode)  # (re)start at the first frame, paused
            self._replay_overlay_running = False
        if self._reference_player.episode is None:
            return

        if bool(st.get("policy_running")):
            # Playing or send-gate-paused: follow the arm -- run while streaming, freeze otherwise.
            self._reference_player.set_paused(not bool(runner.get("streaming")))
            self._replay_overlay_running = True
        elif getattr(self, "_replay_overlay_running", False):
            # Stopped/homed: rewind the ghost to the first frame rather than leave it mid-trajectory.
            self._replay_overlay_running = False
            self._select_reference_episode(dataset, episode)

    def _select_reference_episode(self, dataset: str, episode: int) -> None:
        """Choose `dataset`/`episode` in the overlay panel, if it is there to choose."""
        if self.reference_dataset_combo.currentText().strip() != dataset:
            if self.reference_dataset_combo.findText(dataset) < 0:
                self._refresh_reference_datasets(preferred=dataset)
            if self.reference_dataset_combo.findText(dataset) < 0:
                self.reference_status.setText(
                    f"Replaying {dataset} episode {episode}, which is not under this session's root — "
                    "no overlay for it."
                )
                return
            self.reference_dataset_combo.setCurrentText(dataset)
            self._refresh_reference_episodes()

        row = next((i for i, ep in enumerate(self._reference_episodes) if ep.episode == episode), None)
        if row is None:
            self.reference_status.setText(f"Replaying episode {episode}; it has no completed video to overlay.")
            return
        self.reference_list.setCurrentIndex(row)

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
        if self.run_mode == "dataset":
            # In replay the button starts the run, then pauses/resumes it (a send-gate).
            paused = bool(self.runner is not None and self.runner.replay_paused)
            self.policy_btn.setText("Start Replay" if not running else ("Resume" if paused else "Pause"))
        else:
            self.policy_btn.setText("Stop Policy" if running else "Start Policy")
        self.intervention_btn.setChecked(intervention)
        self.intervention_btn.setText("Human Control" if intervention else "Human Intervention")
        self.dagger_state.setText(f"state: {st.get('dagger_state', 'stopped')}")
        buttons = [self.policy_btn, self.intervention_btn, self.discard_home_btn, self.fail_home_btn]
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
