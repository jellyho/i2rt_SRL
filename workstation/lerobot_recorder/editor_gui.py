"""PyQt viewer + editor for a recorded LeRobot dataset.

Browse the datasets under ``root``, pick one, and inspect every episode: scrub the
frames across all camera views side-by-side, play them back, and read the per-frame
state/action. Then edit:

* **relabel** an episode's outcome (success / fail / discard) — instant, sidecar only;
* **set the task** (language instruction) for the selected episodes;
* **delete** the selected episode(s) — the dataset is re-indexed and (by default)
  backed up first.

Nothing here touches the robot or cameras — it reads the dataset off disk. Heavy
edits run on a worker thread so the UI stays responsive, and every structural edit
goes through :mod:`workstation.lerobot_recorder.dataset_editor` (LeRobot's official
``dataset_tools``) so the data / videos / metadata stay consistent.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_editor import DatasetEditor
from workstation.lerobot_recorder.dataset_reader import DatasetReader
from workstation.lerobot_recorder.dataset_writer import list_datasets
from workstation.lerobot_recorder.views import compose_camera_strip

_MARKS = {"success": "✓", "fail": "✗", "discard": "·"}
# record_source → how it reads in the viewer (what the operator called it).
_SOURCE_LABEL = {"teleop": "teleop", "dagger": "intervention", "eval": "policy rollout"}
# observation.control_mode scalar → label (mirrors config.CONTROL_MODE).
_CONTROL_MODE = {0: "teleop", 1: "policy", 2: "intervention", 3: "replay"}


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class _LogRelay(QtCore.QObject, logging.Handler):
    """A logging handler that re-emits records as a Qt signal, so a long editor op
    running on a worker thread can narrate its progress (LeRobot logs each video file
    it re-encodes) into a progress dialog on the GUI thread."""

    message = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.message.emit(record.getMessage())
        except Exception:
            pass


class _EditWorker(QtCore.QThread):
    """Run one blocking editor operation off the GUI thread."""

    done = QtCore.pyqtSignal(bool, str)  # (ok, message)

    def __init__(self, fn: Callable[[], object], label: str) -> None:
        super().__init__()
        self._fn = fn
        self._label = label

    def run(self) -> None:
        try:
            self._fn()
        except Exception as e:  # surface to the GUI thread
            self.done.emit(False, f"{self._label} failed: {e}")
            return
        self.done.emit(True, f"{self._label} done")


class _LoadWorker(QtCore.QThread):
    """Open the dataset off the GUI thread (LeRobotDataset construction can be slow).

    Emits ``stage`` messages so the progress dialog can narrate, and opens the dataset
    once — the editor reuses the reader's handle instead of opening it a second time.
    """

    stage = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool, str)  # (ok, error)

    def __init__(self, reader: DatasetReader, editor: DatasetEditor) -> None:
        super().__init__()
        self.reader = reader
        self.editor = editor

    def run(self) -> None:
        try:
            self.stage.emit(f"Opening “{self.reader.repo_id}” — decoding metadata & indexing frames…")
            self.reader.load()
            self.stage.emit(f"Grouping {self.reader.num_episodes} episodes…")
            self.editor.attach(self.reader._ds)  # share the open handle; no second full open
            _ = self.editor.tasks_by_episode()  # warms meta.episodes here, off the GUI thread
        except Exception as e:
            self.done.emit(False, str(e))
            return
        self.done.emit(True, "")


class EditorGUI(QtWidgets.QWidget):
    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.reader: Optional[DatasetReader] = None
        self.editor: Optional[DatasetEditor] = None
        self._episode = 0
        self._n_frames = 0
        self._playing = False
        self._worker: Optional[_EditWorker] = None
        self._loader: Optional[_LoadWorker] = None
        self._load_dlg: Optional[QtWidgets.QProgressDialog] = None
        self._edit_dlg: Optional[QtWidgets.QProgressDialog] = None
        self._log_relay: Optional[_LogRelay] = None
        self._by_ep: dict = {}
        self._tasks: dict = {}
        self._sources: dict = {}
        self.setWindowTitle("YAM · LeRobot Dataset Editor")
        self._build()

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance)
        self._refresh_dataset_choices()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.banner = QtWidgets.QLabel("no dataset loaded")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setStyleSheet(theme.banner_style(theme.IDLE))

        # ---- top: dataset picker
        self.root_edit = QtWidgets.QLineEdit(self.cfg.root)
        self.root_edit.editingFinished.connect(self._refresh_dataset_choices)
        self.dataset_cb = QtWidgets.QComboBox()
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load)
        self.rescan_btn = QtWidgets.QPushButton("Re-scan")
        self.rescan_btn.clicked.connect(self._refresh_dataset_choices)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("root"))
        top.addWidget(self.root_edit, 2)
        top.addWidget(QtWidgets.QLabel("dataset"))
        top.addWidget(self.dataset_cb, 3)
        top.addWidget(self.rescan_btn)
        top.addWidget(self.load_btn)

        # ---- left: episode list
        self.ep_list = QtWidgets.QListWidget()
        self.ep_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ep_list.currentRowChanged.connect(self._on_episode_changed)
        self.ep_list.setMinimumWidth(360)
        ep_box = QtWidgets.QGroupBox("Episodes")
        epl = QtWidgets.QVBoxLayout(ep_box)
        epl.addWidget(self.ep_list)
        self.count_lbl = QtWidgets.QLabel("—")
        self.count_lbl.setStyleSheet(f"color:{theme.MUTED};")
        epl.addWidget(self.count_lbl)

        # ---- right: viewer
        self.view = QtWidgets.QLabel("(load a dataset)")
        self.view.setMinimumHeight(360)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.view.setStyleSheet(f"background:#000;border:1px solid #30363d;border-radius:8px;color:{theme.MUTED};")

        # per-episode info line: source (teleop / policy rollout / intervention), outcome, task.
        self.info_lbl = QtWidgets.QLabel("—")
        self.info_lbl.setTextFormat(QtCore.Qt.RichText)
        self.info_lbl.setStyleSheet("font-size:24px;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_scrub)

        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.1, 8.0)
        self.speed.setSingleStep(0.5)
        self.speed.setValue(1.0)
        self.speed.setPrefix("x ")
        self.frame_lbl = QtWidgets.QLabel("frame —")
        self.frame_lbl.setStyleSheet(f"color:{theme.MUTED};")
        transport = QtWidgets.QHBoxLayout()
        transport.addWidget(self.play_btn)
        transport.addWidget(QtWidgets.QLabel("speed"))
        transport.addWidget(self.speed)
        transport.addStretch(1)
        transport.addWidget(self.frame_lbl)

        # ---- edit controls
        edit_box = QtWidgets.QGroupBox("Edit")
        editl = QtWidgets.QVBoxLayout(edit_box)

        self.keep_btn = QtWidgets.QPushButton("✓ success")
        self.fail_btn = QtWidgets.QPushButton("✗ fail")
        self.discard_btn = QtWidgets.QPushButton("· discard")
        self.keep_btn.clicked.connect(lambda: self._on_relabel("success"))
        self.fail_btn.clicked.connect(lambda: self._on_relabel("fail"))
        self.discard_btn.clicked.connect(lambda: self._on_relabel("discard"))
        relabel_row = QtWidgets.QHBoxLayout()
        relabel_row.addWidget(QtWidgets.QLabel("outcome:"))
        for b in (self.keep_btn, self.fail_btn, self.discard_btn):
            relabel_row.addWidget(b)
        editl.addLayout(relabel_row)

        self.task_edit = QtWidgets.QLineEdit()
        self.task_edit.setPlaceholderText("language instruction for the selected episode(s)")
        self.task_btn = QtWidgets.QPushButton("Set task")
        self.task_btn.clicked.connect(self._on_set_task)
        task_row = QtWidgets.QHBoxLayout()
        task_row.addWidget(QtWidgets.QLabel("task:"))
        task_row.addWidget(self.task_edit, 1)
        task_row.addWidget(self.task_btn)
        editl.addLayout(task_row)

        self.delete_btn = QtWidgets.QPushButton("🗑 Delete selected episode(s)")
        self.delete_btn.setStyleSheet(f"color:{theme.BAD};font-weight:600;")
        self.delete_btn.clicked.connect(self._on_delete)
        self.backup_cb = QtWidgets.QCheckBox("Back up before structural edits (safer, uses more disk)")
        self.backup_cb.setChecked(True)
        editl.addWidget(self.delete_btn)
        editl.addWidget(self.backup_cb)

        for w in (self.keep_btn, self.fail_btn, self.discard_btn, self.task_edit,
                  self.task_btn, self.delete_btn):
            w.setEnabled(False)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.view, 1)
        right.addWidget(self.info_lbl)
        right.addWidget(self.slider)
        right.addLayout(transport)
        right.addWidget(edit_box)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(ep_box, 0)
        body.addLayout(right, 1)

        self.status = QtWidgets.QLabel("pick a dataset and press Load")
        self.status.setStyleSheet(f"color:{theme.MUTED};")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(self.banner)
        root.addLayout(top)
        root.addLayout(body, 1)
        root.addWidget(self.status)

    # ------------------------------------------------------------------ dataset picker
    def _refresh_dataset_choices(self) -> None:
        root = self.root_edit.text().strip()
        current = self.dataset_cb.currentText()
        self.dataset_cb.blockSignals(True)
        self.dataset_cb.clear()
        self.dataset_cb.addItems(list_datasets(root))
        if current:
            idx = self.dataset_cb.findText(current)
            if idx >= 0:
                self.dataset_cb.setCurrentIndex(idx)
        elif self.cfg.repo_id:
            idx = self.dataset_cb.findText(self.cfg.repo_id.strip("/").split("/")[-1])
            if idx >= 0:
                self.dataset_cb.setCurrentIndex(idx)
        self.dataset_cb.blockSignals(False)

    # ------------------------------------------------------------------ load
    def _on_load(self) -> None:
        name = self.dataset_cb.currentText().strip()
        root = self.root_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.information(self, "No dataset", "Pick a dataset folder first.")
            return
        if self._busy():
            return
        self.play_btn.setChecked(False)
        self._begin_load(name, root)

    def _begin_load(self, name: str, root: str) -> None:
        """Open <root>/<name> on a worker thread behind a progress dialog."""
        # Release any previously-open dataset on the GUI thread *before* opening the new
        # one — its video decoders / memory-mapped parquet must be torn down here, not
        # concurrently with the new open on the worker thread (that races → segfault).
        self.reader = None
        self.editor = None
        gc.collect()

        self.reader = DatasetReader(name, root, display_cam=self.cfg.review_cam, mock=self.cfg.mock)
        self.editor = DatasetEditor(name, root)

        # Busy (indeterminate) progress dialog; the worker narrates each stage into it.
        dlg = QtWidgets.QProgressDialog(f"Loading “{name}”…", None, 0, 0, self)
        dlg.setWindowTitle("Loading dataset")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        self._load_dlg = dlg
        self._loader = _LoadWorker(self.reader, self.editor)
        self._loader.stage.connect(dlg.setLabelText)
        self._loader.done.connect(self._on_loaded)
        self.setEnabled(False)
        dlg.show()
        self._loader.start()

    def _on_loaded(self, ok: bool, err: str) -> None:
        # Ensure the worker QThread has fully terminated before we drop its reference —
        # dropping a still-running QThread aborts the process ("Destroyed while running").
        loader = self._loader
        if loader is not None:
            loader.wait()
        self._loader = None
        self.setEnabled(True)
        if not ok:
            self._load_dlg.close()
            QtWidgets.QMessageBox.critical(self, "Load failed", err)
            self.reader = self.editor = None
            return
        name = self.reader.repo_id
        # Determinate bar for the part we drive ourselves: building the episode list.
        n = self.reader.num_episodes
        self._load_dlg.setLabelText(f"Building the episode list ({n} episodes)…")
        self._load_dlg.setRange(0, max(n, 1))
        self._populate_episodes(progress=self._load_dlg)
        self._load_dlg.close()
        self.banner.setText(f"{name} · {n} episodes @ {self.reader.fps} fps")
        self.banner.setStyleSheet(theme.banner_style(theme.ACCENT))
        for w in (self.keep_btn, self.fail_btn, self.discard_btn, self.task_edit,
                  self.task_btn, self.delete_btn):
            w.setEnabled(True)

    def _refresh_meta_cache(self) -> None:
        """Cache the per-episode sidecar/metadata dicts used to label the list + info line."""
        self._by_ep = self.editor.outcomes_by_episode()
        self._tasks = self.editor.tasks_by_episode()
        self._sources = self.editor.sources_by_episode()

    def _episode_label(self, e: int) -> str:
        mark = _MARKS.get(self._by_ep.get(e), "")
        src = self._sources.get(e)
        src_tag = f" [{_SOURCE_LABEL.get(src, src)}]" if src else ""
        label = f"ep {e:>3}  ({self.reader.episode_length(e)} fr) {mark}{src_tag}"
        task = self._tasks.get(e, "")
        if task:
            label += f"  — {task[:32]}"
        return label

    def _populate_episodes(self, progress: "QtWidgets.QProgressDialog | None" = None) -> None:
        assert self.reader is not None and self.editor is not None
        self._refresh_meta_cache()
        self.ep_list.blockSignals(True)
        self.ep_list.clear()
        for e in range(self.reader.num_episodes):
            if progress is not None and e % 25 == 0:
                progress.setValue(e)
                QtWidgets.QApplication.processEvents()
            self.ep_list.addItem(self._episode_label(e))
        self.ep_list.blockSignals(False)
        self.count_lbl.setText(f"{self.reader.num_episodes} episodes")
        if self.reader.num_episodes:
            self.ep_list.setCurrentRow(0)

    # ------------------------------------------------------------------ episode view
    def _on_episode_changed(self, row: int) -> None:
        if self.reader is None or row < 0:
            return
        self._episode = row
        self._n_frames = self.reader.episode_length(row)
        self.slider.blockSignals(True)
        self.slider.setEnabled(self._n_frames > 0)
        self.slider.setMaximum(max(self._n_frames - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        # prime the task field from this episode's current task
        if row in getattr(self, "_tasks", {}):
            self.task_edit.setText(self._tasks[row])
        self._update_info_line(row)
        self._show_frame(0)

    def _update_info_line(self, e: int) -> None:
        """Episode-level info: source (teleop / policy rollout / intervention) + outcome + task."""
        src = self._sources.get(e)
        src_txt = _SOURCE_LABEL.get(src, src) if src else "—"
        outcome = self._by_ep.get(e)
        colors = {"success": theme.OK, "fail": theme.BAD, "discard": theme.MUTED}
        oc = (
            f'<span style="color:{colors.get(outcome, theme.MUTED)};">{outcome}</span>'
            if outcome else '<span style="color:%s;">unlabelled</span>' % theme.MUTED
        )
        task = self._tasks.get(e, "")
        task_txt = f" &nbsp;·&nbsp; “{task}”" if task else ""
        self.info_lbl.setText(
            f'<b>ep {e}</b> &nbsp;·&nbsp; source: <b style="color:{theme.ACCENT};">{src_txt}</b>'
            f' &nbsp;·&nbsp; outcome: {oc} &nbsp;·&nbsp; {self._n_frames} frames{task_txt}'
        )

    def _show_frame(self, frame: int) -> None:
        if self.reader is None or self._n_frames == 0:
            return
        frame = max(0, min(frame, self._n_frames - 1))
        images = self.reader.get_images(self._episode, frame)
        composite = compose_camera_strip(images, agent_key=self.cfg.review_cam)
        if isinstance(composite, np.ndarray) and composite.ndim == 3:
            self.view.setPixmap(_np_to_pixmap(composite).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio))
        # per-frame control mode (teleop/policy/intervention) + homing flag, when present.
        tags = ""
        cm = self.reader.get_scalar(self._episode, frame, "observation.control_mode")
        if cm is not None:
            mode = round(cm)
            tags += f"  ·  {_CONTROL_MODE.get(mode, f'mode {mode}')}"
        hz = self.reader.get_scalar(self._episode, frame, "observation.homing")
        if hz is not None and hz >= 0.5:
            tags += "  ·  ⌂ homing"
        self.frame_lbl.setText(f"ep {self._episode} · frame {frame + 1}/{self._n_frames}{tags}")

    def _on_scrub(self, value: int) -> None:
        self._show_frame(value)

    # ------------------------------------------------------------------ playback
    def _on_play_toggled(self, on: bool) -> None:
        self.play_btn.setText("⏸ Pause" if on else "▶ Play")
        self._playing = on
        if on and self.reader is not None:
            interval = max(10, int(1000 / (self.reader.fps * self.speed.value())))
            self._play_timer.start(interval)
        else:
            self._play_timer.stop()

    def _advance(self) -> None:
        if self.reader is None or self._n_frames == 0:
            return
        # keep the timer interval in sync with the speed spinbox
        interval = max(10, int(1000 / (self.reader.fps * self.speed.value())))
        if abs(self._play_timer.interval() - interval) > 2:
            self._play_timer.start(interval)
        nxt = self.slider.value() + 1
        if nxt >= self._n_frames:
            nxt = 0  # loop the episode
        self.slider.setValue(nxt)  # triggers _on_scrub -> _show_frame

    # ------------------------------------------------------------------ edits
    def _selected_episodes(self) -> List[int]:
        return sorted({i.row() for i in self.ep_list.selectedIndexes()}) or [self._episode]

    def _on_relabel(self, outcome: str) -> None:
        if self.editor is None:
            return
        eps = self._selected_episodes()
        try:
            for e in eps:
                self.editor.relabel(e, outcome)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Relabel failed", str(e))
            return
        self._refresh_episode_labels()
        self.status.setText(f"relabelled ep {eps} → {outcome}")

    def _refresh_episode_labels(self) -> None:
        """Cheap in-place refresh of the list marks/tasks/source (no reader reload)."""
        if self.reader is None or self.editor is None:
            return
        self._refresh_meta_cache()
        for e in range(self.ep_list.count()):
            self.ep_list.item(e).setText(self._episode_label(e))
        row = self.ep_list.currentRow()
        if row >= 0:
            self._update_info_line(row)

    def _on_set_task(self) -> None:
        if self.editor is None or self._busy():
            return
        eps = self._selected_episodes()
        task = self.task_edit.text().strip()
        if not task:
            QtWidgets.QMessageBox.information(self, "No task", "Type a language instruction first.")
            return
        backup = self.backup_cb.isChecked()
        self._run_edit(lambda: self.editor.set_task(eps, task, backup=backup), f"Set task for ep {eps}")

    def _on_delete(self) -> None:
        if self.editor is None or self.reader is None or self._busy():
            return
        eps = self._selected_episodes()
        if len(eps) >= self.reader.num_episodes:
            QtWidgets.QMessageBox.warning(self, "Cannot delete", "You can't delete every episode.")
            return
        backup = self.backup_cb.isChecked()
        note = "A backup will be made first." if backup else "NO backup will be made."
        ok = QtWidgets.QMessageBox.warning(
            self,
            "Delete episodes",
            f"Delete {len(eps)} episode(s): {eps}?\n\nThe dataset is re-indexed and rewritten. {note}\n"
            "This cannot be undone from the UI.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if ok != QtWidgets.QMessageBox.Yes:
            return
        self.play_btn.setChecked(False)
        self._run_edit(lambda: self.editor.delete_episodes(eps, backup=backup),
                       f"Delete {len(eps)} episode(s)", reload_after=True)

    # ------------------------------------------------------------------ worker plumbing
    def _busy(self) -> bool:
        return (self._worker is not None and self._worker.isRunning()) or (
            self._loader is not None and self._loader.isRunning()
        )

    def _run_edit(self, fn: Callable[[], object], label: str, reload_after: bool = False) -> None:
        self.banner.setText(f"⟳ {label} …")
        self.banner.setStyleSheet(theme.banner_style(theme.WARN))
        self.status.setText(f"{label} …")
        self._reload_after = reload_after

        # Busy dialog that narrates the op. Re-encoding videos dominates delete time and
        # LeRobot logs each file it processes — forward those lines so the bar isn't silent.
        dlg = QtWidgets.QProgressDialog(f"{label} …\n(re-encoding videos can take a while)", None, 0, 0, self)
        dlg.setWindowTitle("Working")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(560)
        self._edit_dlg = dlg
        self._log_relay = _LogRelay()
        self._log_relay.message.connect(lambda m: dlg.setLabelText(f"{label}\n{m}"))
        self._edit_log_target = logging.getLogger("lerobot")
        self._prev_lerobot_level = self._edit_log_target.level
        self._edit_log_target.setLevel(logging.INFO)
        self._edit_log_target.addHandler(self._log_relay)
        logging.getLogger("workstation.lerobot_recorder.dataset_editor").addHandler(self._log_relay)

        self.setEnabled(False)
        dlg.show()
        self._worker = _EditWorker(fn, label)
        self._worker.done.connect(self._on_edit_done)
        self._worker.start()

    def _teardown_edit_log(self) -> None:
        relay = getattr(self, "_log_relay", None)
        if relay is None:
            return
        self._edit_log_target.removeHandler(relay)
        self._edit_log_target.setLevel(self._prev_lerobot_level)
        logging.getLogger("workstation.lerobot_recorder.dataset_editor").removeHandler(relay)
        self._log_relay = None

    def _on_edit_done(self, ok: bool, message: str) -> None:
        # Ensure the worker QThread finished before dropping it (else the process aborts).
        worker = self._worker
        if worker is not None:
            worker.wait()
        self._worker = None
        self._teardown_edit_log()
        if getattr(self, "_edit_dlg", None) is not None:
            self._edit_dlg.close()
            self._edit_dlg = None
        self.setEnabled(True)
        if not ok:
            self.banner.setText("edit failed")
            self.banner.setStyleSheet(theme.banner_style(theme.BAD))
            QtWidgets.QMessageBox.critical(self, "Edit failed", message)
            self.status.setText(message)
            return
        self.status.setText(message)
        if getattr(self, "_reload_after", False):
            # Structural change (delete): reopen the dataset from disk (threaded, with the
            # progress dialog) and repopulate — this can be slow for large datasets.
            self._begin_load(self.reader.repo_id, self.reader.root)
        else:
            self.editor.load()  # task change altered metadata; reload for fresh reads
            self._refresh_episode_labels()
            self.banner.setText(f"{self.reader.repo_id} · {self.reader.num_episodes} episodes @ {self.reader.fps} fps")
            self.banner.setStyleSheet(theme.banner_style(theme.ACCENT))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._play_timer.stop()
        for t in (self._worker, self._loader):
            if t is not None and t.isRunning():
                t.wait(2000)
        super().closeEvent(event)
