"""Modern PyQt control panel for the LeRobot recorder.

Two stages in one window:

* a **Setup page** (landing): dataset name/dir/task, record **source** (teleop /
  dagger / eval), a **Continue collecting** (resume) toggle, and a status line
  (cameras detected, whether the dataset already exists). No robot connection yet.
* a **Collect page**: the live agentview (with wrist insets), a color status banner,
  a health strip (robot / cameras / disk / queue), live stats, and a review panel.

Pressing **START** resolves the dataset (create / resume / confirmed-overwrite),
opens the cameras + robot + dataset, switches to the Collect page, and (when
``auto_arm``) arms immediately so the next teleop engage records.

A **log panel** at the bottom is shared by both pages and shows the recorder's
important messages (link up/down, camera fallback, dataset saved, faults).
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections import deque
from typing import IO, Any, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import detect_cameras
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import (
    dataset_dir,
    dataset_info,
    dataset_tasks,
    list_datasets,
    remove_dataset_root,
)
from workstation.lerobot_recorder.recorder import Recorder
from workstation.lerobot_recorder.reference_video import (
    ReferenceEpisode,
    ReferenceVideoPlayer,
    discover_reference_episodes,
)
from workstation.lerobot_recorder.sound import Cues
from workstation.lerobot_recorder.views import compose_camera_strip, overlay_camera_views

logger = logging.getLogger(__name__)


_LOCK_PATH = os.path.join(tempfile.gettempdir(), "yam_recorder.lock")


def _acquire_singleton_lock() -> Optional[IO]:
    """Hold an exclusive lock so a second recorder can't fight over the cameras.

    Returns the open file (keep a reference to hold the lock for the process), or
    None if another live recorder already holds it."""
    f = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.write(str(os.getpid()))
    f.flush()
    return f


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class _LogBuffer(logging.Handler):
    """Collect recorder log records (from any thread) for the GUI to drain.

    The recorder's components log from background threads, so we only buffer here
    (a deque is safe to append/popleft across threads) and let the GUI thread render
    them on its timer — never touching Qt widgets off the GUI thread.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__()
        self.records: "deque" = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.levelno, self.format(record)))
        except Exception:
            pass


class PickerComboBox(QtWidgets.QComboBox):
    """Combo box with a large, obvious popup target on the right."""

    DROPDOWN_HITBOX = 86

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, editable: bool = False) -> None:
        super().__init__(parent)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.setEditable(editable)
        self.setProperty("pickerCombo", True)

        if editable:
            edit = self.lineEdit()
            edit.setFrame(False)
            edit.setTextMargins(0, 0, self.DROPDOWN_HITBOX, 0)

        self._popup_btn = QtWidgets.QToolButton(self)
        self._popup_btn.setObjectName("comboDropButton")
        self._popup_btn.setText("▼")
        self._popup_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._popup_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._popup_btn.setToolTip("Open list")
        self._popup_btn.clicked.connect(self.showPopup)
        self._sync_popup_button_geometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_popup_button_geometry()

    def _sync_popup_button_geometry(self) -> None:
        width = min(self.DROPDOWN_HITBOX, max(56, self.width() // 3))
        if self.isEditable() and self.lineEdit() is not None:
            self.lineEdit().setTextMargins(0, 0, width, 0)
        self._popup_btn.setGeometry(self.width() - width, 1, width, max(0, self.height() - 2))
        self._popup_btn.raise_()


class RecorderGUI(QtWidgets.QWidget):
    SETTINGS_SCOPE = "record"

    def __init__(self, cfg: RecorderConfig, settings: Optional[QtCore.QSettings] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._settings = settings or QtCore.QSettings("i2rt", "yam-workstation")
        self.recorder: "Recorder | None" = None  # created on START (so the source picker applies)
        self.cues = Cues(enabled=True)
        self._review_idx = 0
        self._prev: dict = {}
        configured_keys = {camera.key for camera in cfg.cameras}
        preferred_order = ("wrist_left", "agentview", "wrist_right")
        self._reference_camera_keys = tuple(key for key in preferred_order if key in configured_keys)
        self._reference_episodes: list[ReferenceEpisode] = []
        self._reference_player = ReferenceVideoPlayer(self._reference_camera_keys)

        # Surface the recorder's important log messages in-window. Attach to the
        # package logger so cameras / robot link / dataset writer all flow in.
        self._logbuf = _LogBuffer()
        self._logbuf.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        pkg_logger = logging.getLogger("workstation.lerobot_recorder")
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.propagate = False  # don't also emit via the root logger (double lines)
        if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
            pkg_logger.addHandler(logging.StreamHandler())  # keep console output too
        pkg_logger.addHandler(self._logbuf)

        self.setWindowTitle("YAM · LeRobot Recorder")
        self.setStyleSheet(theme.QSS)
        self._build()
        self._restore_setup_settings()
        self._update_setup_status(rescan=True)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)  # 10 Hz UI
        self._review_timer = QtCore.QTimer(self)
        self._review_timer.timeout.connect(self._advance_review)
        self._review_timer.start(50)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.banner = QtWidgets.QLabel("SETUP")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setStyleSheet(theme.banner_style(theme.IDLE))

        self.stack = QtWidgets.QStackedWidget()
        self.setup_page = self._build_setup_page()
        self.collect_page = self._build_collect_page()
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.collect_page)

        # log panel (shared by both pages): recorder's important messages
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)  # cap memory; old lines scroll off
        self.log_view.setMinimumHeight(220)
        self.log_view.setStyleSheet(
            f"background:#0d1117;border:1px solid #30363d;border-radius:6px;color:{theme.MUTED};"
            "font-family:monospace;font-size:23px;"
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(self.banner)
        root.addWidget(self.stack, 2)
        root.addWidget(self.log_view, 1)

    def _build_setup_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        box = QtWidgets.QGroupBox("Session setup")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)

        # Dataset picker: existing datasets under root are selectable; the combo is
        # editable so typing a fresh name creates a new dataset.
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        self.repo_combo = PickerComboBox(editable=True)
        self._refresh_dataset_choices()
        self.repo_combo.setCurrentText(self.cfg.repo_id)

        # Task picker: tasks already used by the selected dataset + config tasks;
        # editable so a new instruction can be typed in.
        self.task_combo = PickerComboBox(editable=True)
        self._refresh_task_choices()
        self.task_combo.setCurrentText(self.cfg.task)

        self.root_edit.textChanged.connect(self._on_root_changed)
        self.repo_combo.editTextChanged.connect(self._on_dataset_changed)

        self.source_combo = PickerComboBox()
        self.source_combo.addItems(["teleop", "dagger", "eval"])
        idx = self.source_combo.findText(self.cfg.record_source)
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.resume_check = QtWidgets.QCheckBox("Continue collecting (append to the existing dataset)")
        self.resume_check.setChecked(self.cfg.resume)
        self.resume_check.toggled.connect(lambda *_: self._update_setup_status())
        self.resume_check.toggled.connect(self._sync_rl_enabled)

        # Per-frame RL signals (success / reward / mc_return). On resume these are
        # inherited from the existing dataset's rl_config.json, so the controls are
        # disabled to make clear the original scheme is preserved.
        self.rl_check = QtWidgets.QCheckBox("Per-frame RL signals (success / reward / mc_return)")
        self.rl_check.setChecked(bool(getattr(self.cfg, "rl_features", False)))
        self.reward_combo = PickerComboBox()
        self.reward_combo.addItems(["sparse", "step"])
        ridx = self.reward_combo.findText(getattr(self.cfg, "reward_mode", "sparse"))
        self.reward_combo.setCurrentIndex(ridx if ridx >= 0 else 0)
        self.discount_spin = QtWidgets.QDoubleSpinBox()
        self.discount_spin.setRange(0.0, 1.0)
        self.discount_spin.setSingleStep(0.01)
        self.discount_spin.setDecimals(3)
        self.discount_spin.setValue(float(getattr(self.cfg, "discount_factor", 0.99)))

        form.addRow("repo_id", self.repo_combo)
        form.addRow("root", self.root_edit)
        form.addRow("task", self.task_combo)
        form.addRow("source", self.source_combo)
        form.addRow("", self.resume_check)
        form.addRow("", self.rl_check)
        form.addRow("reward", self.reward_combo)
        form.addRow("discount γ", self.discount_spin)  # noqa: RUF001
        self._sync_rl_enabled()

        self.setup_status = QtWidgets.QLabel()
        self.setup_status.setTextFormat(QtCore.Qt.RichText)
        self.setup_status.setWordWrap(True)

        self.rescan_btn = QtWidgets.QPushButton("Re-scan cameras")
        self.rescan_btn.clicked.connect(lambda: self._update_setup_status(rescan=True))
        self.start_btn = QtWidgets.QPushButton("START")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self._on_start)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.rescan_btn)
        row.addStretch(1)
        row.addWidget(self.start_btn)

        lay = QtWidgets.QVBoxLayout(page)
        lay.setSpacing(12)
        lay.addWidget(box)
        lay.addWidget(self.setup_status)
        lay.addStretch(1)
        lay.addLayout(row)
        return page

    def _build_collect_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()

        self.health = QtWidgets.QLabel()
        self.health.setTextFormat(QtCore.Qt.RichText)
        self.health.setStyleSheet("font-size:28px;")  # bigger health/queue strip
        self.stats = QtWidgets.QLabel("dataset total 0")
        self.stats.setStyleSheet(f"color:{theme.MUTED};font-size:26px;")
        self.estop_btn = QtWidgets.QPushButton("■ E-STOP")
        self.estop_btn.setObjectName("estop")
        self.estop_btn.setStyleSheet(f"background:{theme.BAD};color:white;font-weight:700;")
        self.estop_btn.setCheckable(True)
        self.estop_btn.toggled.connect(self._on_estop)
        strip = QtWidgets.QHBoxLayout()
        strip.addWidget(self.health)
        strip.addStretch(1)
        strip.addWidget(self.stats)
        strip.addStretch(1)
        strip.addWidget(self.estop_btn)

        self.collect_btn = QtWidgets.QPushButton("Start collection")
        self.collect_btn.setCheckable(True)
        self.collect_btn.setEnabled(False)
        self.collect_btn.clicked.connect(self._on_collect)
        self.save_btn = QtWidgets.QPushButton("Save dataset")
        self.save_btn.setObjectName("save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_dataset)

        # primary operator view: left wrist, agentview, right wrist with no overlap
        self.live_lbl = QtWidgets.QLabel("camera view")
        self.live_lbl.setMinimumHeight(360)
        self.live_lbl.setAlignment(QtCore.Qt.AlignCenter)
        # Ignored size policy so the scaled pixmap never feeds back into the layout
        # (otherwise the label keeps growing each frame).
        self.live_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.live_lbl.setStyleSheet(f"background:#000;border:1px solid #30363d;border-radius:8px;color:{theme.MUTED};")

        # Past demonstrations from the selected dataset can be overlaid on the three
        # corresponding live views.  This panel sits entirely in the display
        # path: Recorder and DeploymentPolicyRunner continue to receive raw images.
        self.reference_box = QtWidgets.QGroupBox("Past demonstration overlay · preview only")
        reference_layout = QtWidgets.QHBoxLayout(self.reference_box)
        self.reference_list = QtWidgets.QListWidget()
        self.reference_list.setMaximumHeight(115)
        self.reference_list.setMinimumWidth(260)
        self.reference_list.currentRowChanged.connect(self._on_reference_selected)
        reference_layout.addWidget(self.reference_list, 1)

        reference_controls = QtWidgets.QVBoxLayout()
        self.reference_status = QtWidgets.QLabel("Select a saved demonstration to compare all three camera views.")
        self.reference_status.setWordWrap(True)
        self.reference_status.setStyleSheet(f"color:{theme.MUTED};")
        self.reference_opacity_label = QtWidgets.QLabel()
        self.reference_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.reference_opacity.setRange(0, 100)
        self.reference_opacity.setValue(round(100 * self.cfg.reference_live_alpha))
        self.reference_opacity.valueChanged.connect(self._on_reference_opacity)
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(self.reference_opacity_label)
        opacity_row.addWidget(self.reference_opacity, 1)
        self.reference_pause_btn = QtWidgets.QPushButton("Resume reference")
        self.reference_pause_btn.setEnabled(False)
        self.reference_pause_btn.clicked.connect(self._on_reference_pause)
        self.reference_refresh_btn = QtWidgets.QPushButton("Refresh demonstrations")
        self.reference_refresh_btn.clicked.connect(self._refresh_reference_episodes)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.reference_pause_btn)
        button_row.addWidget(self.reference_refresh_btn)
        reference_controls.addWidget(self.reference_status)
        reference_controls.addLayout(opacity_row)
        reference_controls.addLayout(button_row)
        reference_layout.addLayout(reference_controls, 2)
        self._on_reference_opacity(self.reference_opacity.value())

        # review panel
        self.review_box = QtWidgets.QGroupBox("Episode review")
        rv = QtWidgets.QVBoxLayout(self.review_box)
        self.review_lbl = QtWidgets.QLabel("(no episode pending)")
        self.review_lbl.setMinimumHeight(180)
        self.review_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.review_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.review_lbl.setStyleSheet(
            f"background:#000;border:1px solid #30363d;border-radius:6px;color:{theme.MUTED};"
        )
        self.review_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.review_slider.setEnabled(False)
        self.review_slider.sliderMoved.connect(self._scrub)
        self.keep_btn = QtWidgets.QPushButton("✓ Keep success  [S]")
        self.keepfail_btn = QtWidgets.QPushButton("Keep fail  [F]")
        self.del_btn = QtWidgets.QPushButton("✗ Delete  [D]")
        self.keep_btn.clicked.connect(lambda: self._on_keep("success"))
        self.keepfail_btn.clicked.connect(lambda: self._on_keep("fail"))
        self.del_btn.clicked.connect(self._on_delete)
        rb = QtWidgets.QHBoxLayout()
        for b in (self.keep_btn, self.keepfail_btn, self.del_btn):
            b.setEnabled(False)
            rb.addWidget(b)
        rv.addWidget(self.review_lbl)
        rv.addWidget(self.review_slider)
        rv.addLayout(rb)
        # Auto-save sessions (review_before_save: false) never use this panel — hide it
        # and give the room to the log. It stays for review-mode sessions.
        self.review_box.setVisible(self.cfg.review_before_save)

        self.hint = QtWidgets.QLabel("space toggles collection · S/F keep · D delete")
        self.hint.setStyleSheet(f"color:{theme.MUTED};")

        lay = QtWidgets.QVBoxLayout(page)
        lay.setSpacing(12)
        lay.addLayout(strip)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.collect_btn, 1)
        controls.addWidget(self.save_btn)
        lay.addLayout(controls)
        lay.addWidget(self.live_lbl, 1)
        lay.addWidget(self.reference_box)
        lay.addWidget(self.review_box)
        lay.addWidget(self.hint)
        return page

    # ------------------------------------------------------------------ setup pickers
    def _settings_key(self, name: str) -> str:
        return f"setup/{self.SETTINGS_SCOPE}/{name}"

    def _restore_setup_settings(self) -> None:
        """Fill the setup page from the last values used by this UI.

        Only widgets are restored here. The recorder configuration is still
        applied exclusively by START, preserving the existing setup workflow.
        """

        def saved(name: str, fallback: Any, value_type: type) -> Any:
            key = self._settings_key(name)
            if not self._settings.contains(key):
                return fallback
            return self._settings.value(key, fallback, type=value_type)

        # Set root/repo/task in this order so their existing change handlers
        # repopulate the dependent dataset and task choices correctly.
        self.root_edit.setText(saved("root", self.root_edit.text(), str))
        self.repo_combo.setCurrentText(saved("repo_id", self.repo_combo.currentText(), str))
        self.task_combo.setCurrentText(saved("task", self.task_combo.currentText(), str))

        source = saved("source", self.source_combo.currentText(), str)
        source_idx = self.source_combo.findText(source)
        if source_idx >= 0:
            self.source_combo.setCurrentIndex(source_idx)

        self.resume_check.setChecked(saved("resume", self.resume_check.isChecked(), bool))
        self.rl_check.setChecked(saved("rl_features", self.rl_check.isChecked(), bool))

        reward = saved("reward_mode", self.reward_combo.currentText(), str)
        reward_idx = self.reward_combo.findText(reward)
        if reward_idx >= 0:
            self.reward_combo.setCurrentIndex(reward_idx)
        self.discount_spin.setValue(saved("discount_factor", self.discount_spin.value(), float))
        live_alpha = min(max(saved("reference_live_alpha", self.cfg.reference_live_alpha, float), 0.0), 1.0)
        self.reference_opacity.setValue(round(100 * live_alpha))
        self._sync_rl_enabled()

    def _save_setup_settings(self) -> None:
        """Persist the setup page as currently filled, even if START was never pressed."""
        values = {
            "repo_id": self.repo_combo.currentText().strip(),
            "root": self.root_edit.text().strip(),
            "task": self.task_combo.currentText().strip(),
            "source": self.source_combo.currentText().strip(),
            "resume": self.resume_check.isChecked(),
            "rl_features": self.rl_check.isChecked(),
            "reward_mode": self.reward_combo.currentText().strip(),
            "discount_factor": float(self.discount_spin.value()),
            "reference_live_alpha": self.reference_opacity.value() / 100.0,
        }
        for name, value in values.items():
            self._settings.setValue(self._settings_key(name), value)
        self._settings.sync()

    def _refresh_dataset_choices(self) -> None:
        """Repopulate the dataset picker from the folders under root, keeping
        whatever is currently typed (a new name stays a new name)."""
        current = self.repo_combo.currentText()
        root = self.root_edit.text().strip() if hasattr(self, "root_edit") else self.cfg.root
        self.repo_combo.blockSignals(True)
        self.repo_combo.clear()
        self.repo_combo.addItems(list_datasets(root))
        self.repo_combo.setCurrentText(current)
        self.repo_combo.blockSignals(False)

    def _refresh_task_choices(self) -> None:
        """Task choices = tasks already in the selected dataset (kept consistent on
        resume) + the config.yaml tasks; the current/typed text is preserved."""
        current = self.task_combo.currentText()
        root = self.root_edit.text().strip()
        repo = self.repo_combo.currentText().strip()
        ds_tasks = dataset_tasks(dataset_dir(root, repo)) if repo else []
        seen = []
        for t in [*ds_tasks, self.cfg.task, *self.cfg.tasks]:
            if t and t not in seen:
                seen.append(t)
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        self.task_combo.addItems(seen)
        self.task_combo.setCurrentText(current if current else (seen[0] if seen else ""))
        self.task_combo.blockSignals(False)

    def _on_root_changed(self) -> None:
        self._refresh_dataset_choices()
        self._refresh_task_choices()
        self._update_setup_status()

    def _on_dataset_changed(self) -> None:
        self._refresh_task_choices()
        self._update_setup_status()

    # ------------------------------------------------------------------ setup status
    def _update_setup_status(self, rescan: bool = False) -> None:
        """Refresh the camera-detected + dataset-exists line on the setup page."""
        if rescan or not hasattr(self, "_cam_detect"):
            self._cam_detect = detect_cameras(self.cfg)
        cam = self._cam_detect
        cam_ok = cam["found"] == cam["total"]
        cam_col = theme.OK if cam_ok else theme.BAD
        cam_txt = f'<span style="color:{cam_col};">●</span> cameras {cam["found"]}/{cam["total"]}'
        if cam.get("missing"):
            cam_txt += f" (missing: {', '.join(cam['missing'])})"

        root = self.root_edit.text().strip() if hasattr(self, "root_edit") else self.cfg.root
        repo = self.repo_combo.currentText().strip() if hasattr(self, "repo_combo") else self.cfg.repo_id
        ds_dir = dataset_dir(root, repo)
        info = dataset_info(ds_dir)
        where = f' <span style="color:{theme.MUTED};">→ {ds_dir}</span>'
        if not info["exists"]:
            ds_txt = f'<span style="color:{theme.OK};">●</span> dataset: new (will be created){where}'
        else:
            n = info["episodes"]
            ntxt = f"{n} episode(s)" if n is not None else "existing data"
            if self.resume_check.isChecked():
                append = f'<b><span style="color:{theme.OK};">will append</span></b>'
                ds_txt = f'<span style="color:{theme.ACCENT};">●</span> dataset: exists ({ntxt}) — {append}{where}'
            else:
                overwrite = f'<b><span style="color:{theme.BAD};">START will offer to overwrite</span></b>'
                ds_txt = f'<span style="color:{theme.WARN};">●</span> dataset: exists ({ntxt}) — {overwrite}{where}'
        self.setup_status.setText(cam_txt + "<br>" + ds_txt)

    # ------------------------------------------------------------------ actions
    def _sync_rl_enabled(self) -> None:
        """On resume the RL scheme is inherited from the dataset, so grey out the
        controls to signal they won't change the existing dataset."""
        resuming = self.resume_check.isChecked()
        for w in (self.rl_check, self.reward_combo, self.discount_spin):
            w.setEnabled(not resuming)

    def _on_start(self) -> None:
        cfg = self.cfg
        cfg.repo_id = self.repo_combo.currentText().strip()
        cfg.root = self.root_edit.text().strip()
        cfg.task = self.task_combo.currentText().strip()
        cfg.record_source = self.source_combo.currentText().strip()
        cfg.resume = self.resume_check.isChecked()
        cfg.rl_features = self.rl_check.isChecked()
        cfg.reward_mode = self.reward_combo.currentText().strip()
        cfg.discount_factor = float(self.discount_spin.value())
        cfg.reference_live_alpha = self.reference_opacity.value() / 100.0

        # Single-instance guard: two recorders fighting over the cameras causes the
        # flapping/freeze, so refuse to start a second one with a clear message.
        if not cfg.mock and getattr(self, "_lock", None) is None:
            self._lock = _acquire_singleton_lock()
            if self._lock is None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Recorder already running",
                    "Another YAM recorder appears to be running and holds the cameras.\n"
                    "Close it first, then press START again.",
                )
                return

        ds_dir = dataset_dir(cfg.root, cfg.repo_id)
        info = dataset_info(ds_dir)
        if info["exists"] and not cfg.resume:
            if not self._confirm_overwrite(ds_dir, info["episodes"]):
                return
            try:
                remove_dataset_root(ds_dir)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Delete failed", str(e))
                return
        elif not info["exists"]:
            cfg.resume = False  # nothing to resume

        self.recorder = Recorder(cfg)
        self.start_btn.setEnabled(False)
        try:
            self.recorder.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start failed", str(e))
            self.recorder = None
            self.start_btn.setEnabled(True)
            return

        self._prev = self.recorder.get_status()
        self.stack.setCurrentWidget(self.collect_page)
        self.collect_btn.setEnabled(True)
        self._refresh_reference_episodes()
        if cfg.auto_arm:  # arm immediately so the next teleop engage records
            self.collect_btn.setChecked(True)
            self._on_collect()

    def _refresh_reference_episodes(self) -> None:
        """Rescan completed three-camera demonstrations in the active dataset."""
        selected = self._reference_player.episode
        selected_number = selected.episode if selected is not None else None
        root = dataset_dir(self.cfg.root, self.cfg.repo_id)
        try:
            episodes = discover_reference_episodes(root, self._reference_camera_keys)
        except Exception as exc:
            logger.warning("could not read past demonstration metadata: %s", exc)
            self.reference_status.setText(f"Could not read past demonstrations: {exc}")
            return
        self._reference_episodes = episodes

        self.reference_list.blockSignals(True)
        self.reference_list.clear()
        selected_row = 0 if episodes else -1
        kept_selection = False
        for index, episode in enumerate(episodes):
            duration = episode.length / episode.fps if episode.length and episode.fps else 0.0
            duration_text = f" · {duration:.1f}s" if duration else ""
            self.reference_list.addItem(f"{episode.label}{duration_text} · 3 synchronized views")
            if episode.episode == selected_number:
                selected_row = index
                kept_selection = True
        self.reference_list.setCurrentRow(selected_row)
        self.reference_list.blockSignals(False)

        if not episodes:
            self._reference_player.stop()
            self.reference_pause_btn.setEnabled(False)
            self.reference_pause_btn.setText("Resume reference")
            self.reference_status.setText(
                "No completed three-camera demonstrations in this dataset yet. "
                "Use Continue collecting to retain past demonstrations, then refresh after a save."
            )
        elif not kept_selection:
            # START and a refresh with no surviving selection both default to
            # the dataset's first completed episode, held on its first frame.
            self._on_reference_selected(selected_row)
        else:
            self.reference_status.setText(
                f"{len(episodes)} saved demonstration(s) available. "
                "Set live camera opacity to 100% to hide the reference."
            )

    def _on_reference_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._reference_episodes):
            self._reference_player.stop()
            self.reference_pause_btn.setEnabled(False)
            self.reference_pause_btn.setText("Resume reference")
            return
        episode = self._reference_episodes[row]
        try:
            self._reference_player.play(episode, start_paused=True)
        except Exception as exc:
            logger.warning("could not start past episode %s overlay: %s", episode.episode, exc)
            self.reference_status.setText(f"Could not play {episode.label}: {exc}")
            self.reference_pause_btn.setEnabled(False)
            return
        self.reference_pause_btn.setEnabled(True)
        self.reference_pause_btn.setText("Resume reference")
        self.reference_status.setText(
            f"Paused {episode.label} at its first frame over wrist-left · agentview · wrist-right (preview only)."
        )

    def _on_reference_opacity(self, value: int) -> None:
        self.cfg.reference_live_alpha = min(max(value / 100.0, 0.0), 1.0)
        self.reference_opacity_label.setText(f"Live camera opacity: {value}%")

    def _on_reference_pause(self) -> None:
        paused = not self._reference_player.paused
        self._reference_player.set_paused(paused)
        self.reference_pause_btn.setText("Resume reference" if paused else "Pause reference")
        episode = self._reference_player.episode
        if episode is not None:
            action = "Frozen" if paused else "Looping"
            self.reference_status.setText(f"{action} {episode.label} (preview only).")

    def _confirm_overwrite(self, root: str, episodes: Optional[int]) -> bool:
        """Two sequential confirms before deleting an existing dataset."""
        ntxt = f"{episodes} episode(s)" if episodes is not None else "existing data"
        first = QtWidgets.QMessageBox.warning(
            self,
            "Folder already exists",
            f"{os.path.expanduser(root)}\nalready contains {ntxt}.\n\n"
            "Overwrite it and start a fresh dataset?\n(Check 'Continue collecting' instead to append.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if first != QtWidgets.QMessageBox.Yes:
            return False
        second = QtWidgets.QMessageBox.critical(
            self,
            "Confirm overwrite",
            "This permanently DELETES the existing dataset. This cannot be undone.\n\nAre you sure?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return second == QtWidgets.QMessageBox.Yes

    def _on_collect(self) -> None:
        if self.recorder is None:
            return
        if self.collect_btn.isChecked():
            self.cfg.task = self.task_combo.currentText().strip()
            self.recorder.arm()
            self.collect_btn.setText("Stop collection")
        else:
            self.recorder.disarm()
            self.collect_btn.setText("Start collection")

    def _on_save_dataset(self) -> None:
        if self.recorder is None:
            return
        st = self.recorder.get_status()
        if st["recording"] or st.get("leader_recentering"):
            QtWidgets.QMessageBox.information(
                self, "Recording in progress", "Finish the current episode before saving."
            )
            return
        if st["pending"]:
            QtWidgets.QMessageBox.information(
                self, "Review pending", "Keep or delete the pending episode before saving."
            )
            return

        self.save_btn.setEnabled(False)
        self.save_btn.setText("Saving...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            self.recorder.save_dataset()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._refresh()

    def _on_keep(self, outcome: str) -> None:
        if self.recorder is not None:
            self.recorder.keep_episode(outcome=outcome)

    def _on_delete(self) -> None:
        if self.recorder is not None:
            self.recorder.delete_episode()

    def _on_estop(self, engaged: bool) -> None:
        if self.recorder is not None:
            self.recorder.set_estop(engaged)
        self.estop_btn.setText("■ E-STOP (engaged)" if engaged else "■ E-STOP")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if self.recorder is not None and self.stack.currentWidget() is self.collect_page:
            if key == QtCore.Qt.Key_Space and self.collect_btn.isEnabled():
                self.collect_btn.toggle()
                self._on_collect()
            elif self.recorder.get_status()["pending"]:
                if key == QtCore.Qt.Key_S:
                    self._on_keep("success")
                elif key == QtCore.Qt.Key_F:
                    self._on_keep("fail")
                elif key == QtCore.Qt.Key_D:
                    self._on_delete()
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ refresh
    def _refresh(self) -> None:
        self._drain_log()
        if self.recorder is None:
            return  # setup page: status updated on field-change / re-scan
        st = self.recorder.get_status()
        self._update_banner(st)
        self._update_health(st)
        self._update_stats(st)
        self._update_save_button(st)
        self._cue_transitions(self._prev, st)

        for b in (self.keep_btn, self.keepfail_btn, self.del_btn):
            b.setEnabled(st["pending"])
        if not st["pending"]:
            self.review_lbl.setText("(no episode pending)")
            self.review_slider.setEnabled(False)

        images = self.recorder.get_last_images()
        reference_frames = self._reference_player.get_frames()
        preview_images = (
            overlay_camera_views(images, reference_frames, live_alpha=self.cfg.reference_live_alpha)
            if reference_frames
            else images
        )
        composite = compose_camera_strip(preview_images, agent_key=self.cfg.review_cam)
        if isinstance(composite, np.ndarray) and composite.ndim == 3:
            self.live_lbl.setPixmap(_np_to_pixmap(composite).scaled(self.live_lbl.size(), QtCore.Qt.KeepAspectRatio))

        reference_error = self._reference_player.error
        if reference_error:
            self.reference_status.setText(f"Reference overlay stopped: {reference_error}")
        elif self._reference_player.finished:
            self.reference_pause_btn.setText("Resume reference")
            episode = self._reference_player.episode
            if episode is not None:
                self.reference_status.setText(f"Finished {episode.label}; Resume reference replays it from the start.")

        self._prev = st

    def _drain_log(self) -> None:
        """Render any buffered log records (called on the GUI thread)."""
        colors = {logging.ERROR: theme.BAD, logging.WARNING: theme.WARN}
        appended = False
        while self._logbuf.records:
            try:
                level, msg = self._logbuf.records.popleft()
            except IndexError:
                break
            color = colors.get(level, theme.MUTED)
            self.log_view.appendHtml(f'<span style="color:{color};">{msg}</span>')
            appended = True
        if appended:
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _update_banner(self, st: dict) -> None:
        if not st.get("disk_ok", True):
            text, color = "⚠ LOW DISK — not saving", theme.STATE_COLORS["ERROR"]
        elif not (st["cam_ok"] and st.get("robot_ok", True)):
            text, color = "⚠ DEVICE FAULT", theme.STATE_COLORS["ERROR"]
        elif st["pending"]:
            text, color = "REVIEW — keep [S/F] or delete [D]", theme.STATE_COLORS["REVIEW"]
        elif st.get("recenter_fault"):
            text, color = "⚠ LEADER ALIGNMENT TIMED OUT — FOLLOWER HELD", theme.STATE_COLORS["ERROR"]
        elif st.get("leader_recentering"):
            text, color = "ALIGNING LEADER — RECORDING PAUSED", theme.STATE_COLORS["REVIEW"]
        elif st["recording"]:
            text = "● REC (FINE-GRAINED)" if st.get("fine_grained") else "● REC"
            color = theme.STATE_COLORS["REC"]
        elif st["armed"]:
            text, color = f"ARMED · {st['teleop']}", theme.STATE_COLORS["ARMED"]
        else:
            text, color = "IDLE" if st["running"] else "not started", theme.STATE_COLORS["IDLE"]
        self.banner.setText(text)
        self.banner.setStyleSheet(theme.banner_style(color))

    def _update_health(self, st: dict) -> None:
        cam = theme.dot(st["cam_ok"])
        rob = theme.dot(st.get("robot_ok", False))
        disk = theme.dot(st.get("disk_ok", True))
        w = st.get("writer") or {}
        workers = w.get("workers", 1)
        saved = w.get("saved", 0)
        queued = w.get("queued", st.get("queue", 0))
        # idle (green) vs encoding-one-now (amber ⟳, with episode + frame count) vs backlog.
        if w.get("saving"):
            idx, fr = w.get("saving_index"), w.get("saving_frames", 0)
            save_txt = f'<span style="color:{theme.WARN};">⟳ encoding #{idx} ({fr} frames)</span> · queued {queued}'
        elif queued:
            save_txt = f'<span style="color:{theme.WARN};">● queued {queued}</span>'
        else:
            save_txt = f'<span style="color:{theme.OK};">● idle</span>'
        self.health.setText(
            f"{rob} robot &nbsp;&nbsp; {cam} cameras &nbsp;&nbsp; {disk} disk "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; writer: {workers} worker · saved {saved} · {save_txt}"
        )

    def _update_stats(self, st: dict) -> None:
        kept, suc, fail, disc = st["kept"], st["success"], st["fail"], st["discarded"]
        rate = f"{100 * suc / max(suc + fail, 1):.0f}%" if (suc + fail) else "—"
        total = st.get("episodes_total", st["episodes"])
        tsuc, tfail = st.get("success_total", 0), st.get("fail_total", 0)
        self.stats.setText(
            f"dataset {total} (✓{tsuc} ✗{tfail}) · session kept {kept} (✓{suc} ✗{fail}) · "
            f"discarded {disc} · success {rate} · frames {st['frames']}"
        )

    def _update_save_button(self, st: dict) -> None:
        w = st.get("writer") or {}
        submitted = int(w.get("submitted", 0) or 0)
        finalized = bool(w.get("finalized"))
        if finalized and submitted:
            self.save_btn.setText("Saved")
        else:
            self.save_btn.setText("Save dataset")
        self.save_btn.setEnabled(bool(st.get("saveable") and not st.get("recording") and not st.get("pending")))

    def _cue_transitions(self, prev: dict, cur: dict) -> None:
        if cur["recording"] and not prev.get("recording") and not prev.get("leader_recentering"):
            self.cues.play("start")
        if cur["success"] > prev.get("success", 0):
            self.cues.play("success")
        if cur["fail"] > prev.get("fail", 0):
            self.cues.play("fail")
        if cur["discarded"] > prev.get("discarded", 0):
            self.cues.play("delete")
        healthy_now = cur["cam_ok"] and cur.get("robot_ok", True)
        healthy_was = prev.get("cam_ok", True) and prev.get("robot_ok", True)
        if healthy_was and not healthy_now:
            self.cues.play("error")

    def _scrub(self, value: int) -> None:
        if self.recorder is None:
            return
        frames = self.recorder.get_review_frames()
        if frames:
            self._review_idx = max(0, min(value, len(frames) - 1))
            self._show_review(frames[self._review_idx])

    def _advance_review(self) -> None:
        if self.recorder is None or not self.recorder.get_status()["pending"]:
            self._review_idx = 0
            return
        frames = self.recorder.get_review_frames()
        if not frames:
            return
        self.review_slider.setEnabled(True)
        self.review_slider.setMaximum(len(frames) - 1)
        if not self.review_slider.isSliderDown():
            self._review_idx = (self._review_idx + 1) % len(frames)
            self.review_slider.setValue(self._review_idx)
            self._show_review(frames[self._review_idx])

    def _show_review(self, img: np.ndarray) -> None:
        if isinstance(img, np.ndarray) and img.ndim == 3:
            self.review_lbl.setPixmap(_np_to_pixmap(img).scaled(self.review_lbl.size(), QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_setup_settings()
        self._reference_player.stop()
        if self.recorder is not None:
            self.recorder.shutdown()
        lock = getattr(self, "_lock", None)
        if lock is not None:
            try:
                lock.close()  # releases the flock
            except Exception:
                pass
        super().closeEvent(event)
