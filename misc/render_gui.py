"""Point-and-click front end for ``yam-data render-samples``.

    workstation/yam-data render-gui

Pick a dataset, pick an episode, press Render. Everything the command line asks for that the
recording already knows -- how many candidates, where the replans were, whether a critic scored
them, whether the run was adaptive -- is read from the dataset and shown, rather than typed in and
got wrong (a mistyped ``--candidates`` is a reshape error, a mistyped ``--horizon`` silently draws
every chunk in the wrong place).

The render itself is :func:`render_deploy_samples.render`, unchanged: this window builds the same
argparse Namespace the CLI would and calls it on a worker thread, so there is one renderer and the
GUI cannot drift from it.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import traceback
from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.dataset_writer import list_datasets

if TYPE_CHECKING:  # the reader pulls in LeRobot, which a GUI import should not
    from workstation.lerobot_recorder.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "~/lerobot_rollout"


class _RenderWorker(QtCore.QThread):
    """Runs one render off the GUI thread; the window stays responsive and can report failure."""

    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

    def run(self) -> None:
        from misc.render_deploy_samples import render

        try:
            out = render(self._args)
            self.done.emit(True, str(out))
        except BaseException as e:  # a failed render must land in the window, not the console
            logger.exception("render failed")
            self.done.emit(False, f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}")


class RenderGUI(QtWidgets.QWidget):
    def __init__(self, root: str = DEFAULT_ROOT) -> None:
        super().__init__()
        self.setWindowTitle("YAM · Render rollout")
        self.setStyleSheet(theme.QSS)
        self._worker: _RenderWorker | None = None
        self._episodes: list[tuple[int, int]] = []  # (episode, frames)

        self.root_edit = QtWidgets.QLineEdit(root)
        self.root_edit.setToolTip("Folder holding one subfolder per dataset (the recorder's root)")
        self.root_edit.editingFinished.connect(self._refresh_datasets)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._browse)

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.currentTextChanged.connect(lambda *_: self._refresh_episodes())
        self.episode_combo = QtWidgets.QComboBox()
        self.episode_combo.currentIndexChanged.connect(lambda *_: self._sync_out())

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["samples", "action"])
        self.source_combo.setToolTip(
            "samples: the multi-candidate fan (needs an action_samples column)\n"
            "action: the single executed trajectory -- works on any recording"
        )
        self.source_combo.currentTextChanged.connect(lambda *_: self._sync_out())

        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 8.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setToolTip("Playback speed of the written file (2 = half the duration)")
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(120, 1080)
        self.height_spin.setSingleStep(60)
        self.height_spin.setValue(360)
        self.height_spin.setToolTip("Per-camera panel height in px")

        self.value_check = QtWidgets.QCheckBox("critic value strip")
        self.value_check.setChecked(True)
        self.chunk_check = QtWidgets.QCheckBox("chunk length strip")
        self.chunk_check.setChecked(True)

        self.out_edit = QtWidgets.QLineEdit()
        self.render_btn = QtWidgets.QPushButton("Render")
        self.render_btn.clicked.connect(self._on_render)

        self.info = QtWidgets.QLabel("—")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color:{theme.MUTED};")
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse)
        form.addRow("root", root_row)
        form.addRow("dataset", self.dataset_combo)
        form.addRow("episode", self.episode_combo)
        form.addRow("overlay", self.source_combo)
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("speed"))
        opts.addWidget(self.speed_spin)
        opts.addWidget(QtWidgets.QLabel("panel height"))
        opts.addWidget(self.height_spin)
        opts.addWidget(self.value_check)
        opts.addWidget(self.chunk_check)
        opts.addStretch(1)
        form.addRow("options", opts)
        form.addRow("output", self.out_edit)

        lay = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Render a recorded rollout")
        box.setLayout(form)
        lay.addWidget(box)
        lay.addWidget(self.info)
        lay.addWidget(self.render_btn)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.resize(760, 380)

        self._refresh_datasets()

    # ------------------------------------------------------------------ dataset discovery
    def _browse(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Datasets root", os.path.expanduser(self.root_edit.text().strip() or "~")
        )
        if chosen:
            self.root_edit.setText(chosen)
            self._refresh_datasets()

    def _refresh_datasets(self) -> None:
        names = list_datasets(self.root_edit.text().strip() or DEFAULT_ROOT)
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(names)
        if current in names:
            self.dataset_combo.setCurrentText(current)
        self.dataset_combo.blockSignals(False)
        self.dataset_combo.setEnabled(bool(names))
        if not names:
            self.info.setText(f"No dataset folders under {os.path.expanduser(self.root_edit.text().strip())}.")
            self.episode_combo.clear()
            return
        self._refresh_episodes()

    def _reader(self) -> "DatasetReader":
        """A metadata-only reader for the selected dataset (no frames, no video index)."""
        from workstation.lerobot_recorder.dataset_reader import DatasetReader

        name = self.dataset_combo.currentText().strip()
        reader = DatasetReader(name, self.root_edit.text().strip() or DEFAULT_ROOT)
        reader.load()
        return reader

    def _refresh_episodes(self) -> None:
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self._episodes = []
        try:
            reader = self._reader()
            for ep in range(reader.num_episodes):
                frames = reader.episode_length(ep)
                self._episodes.append((ep, frames))
                self.episode_combo.addItem(f"episode {ep}  ·  {frames} frames  ({frames / max(reader.fps, 1):.0f}s)")
            self._describe(reader)
        except Exception as e:
            self.info.setText(f"Could not read this dataset: {e}")
        self.episode_combo.blockSignals(False)
        self.episode_combo.setEnabled(bool(self._episodes))
        self._sync_out()

    def _describe(self, reader: "DatasetReader") -> None:
        """Say what the recording carries, so the operator does not have to remember it."""
        bits = [f"{reader.num_episodes} episode(s) at {reader.fps} fps"]
        samples = reader.feature_shape("action_samples")
        if samples:
            bits.append(f"{samples[0]} candidates")
            self.source_combo.setCurrentText("samples")
        else:
            bits.append("no action_samples — executed trajectory only")
            self.source_combo.setCurrentText("action")
        if reader.has_feature("critic_scores"):
            bits.append("critic scores")
        if reader.feature_shape("action_samples_full"):
            bits.append("ADAPTIVE (full candidates recorded)")
        if reader.has_feature("policy.chunk_index"):
            bits.append("replan boundaries from the run")
        self.info.setText(" · ".join(bits))
        self.source_combo.setEnabled(bool(samples))

    def _sync_out(self) -> None:
        name = self.dataset_combo.currentText().strip() or "rollout"
        ep = self._current_episode()
        if ep is None:
            self.out_edit.clear()
            return
        self.out_edit.setText(str(pathlib.Path.home() / f"{name}_ep{ep}.mp4"))

    def _current_episode(self) -> "int | None":
        row = self.episode_combo.currentIndex()
        return self._episodes[row][0] if 0 <= row < len(self._episodes) else None

    # ------------------------------------------------------------------ rendering
    def _build_args(self) -> argparse.Namespace:
        """The same Namespace the CLI builds -- defaults included, so the renderer sees one shape.

        Everything the recording knows is left as None for the renderer to recover: candidates from
        the action_samples column, chunk boundaries from policy.chunk_index.
        """
        return argparse.Namespace(
            repo_id=self.dataset_combo.currentText().strip(),
            root=self.root_edit.text().strip() or DEFAULT_ROOT,
            config=None,
            episode=self._current_episode(),
            wrists=["left", "right"],
            agentview_arms=["left", "right"],
            source=self.source_combo.currentText(),
            horizon=None,
            candidates=None,
            no_value_plot=not self.value_check.isChecked(),
            no_chunk_plot=not self.chunk_check.isChecked(),
            replans=0,
            hold=1,
            height=self.height_spin.value(),
            # Speed is applied as the written frame rate: the render draws one frame per recorded
            # tick, so N x speed is just N x the dataset rate rather than a re-encode.
            fps=max(1, round(10 * self.speed_spin.value())),
            out=self.out_edit.text().strip(),
            fx=430.0,
            fy=430.0,
            cx=320.0,
            cy=240.0,
            agent_fx=390.0,
            agent_fy=390.0,
            agent_cx=320.0,
            agent_cy=240.0,
        )

    def _on_render(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._current_episode() is None:
            self.status.setText("Pick an episode first.")
            return
        args = self._build_args()
        if not args.out:
            self.status.setText("Set an output path first.")
            return
        self.render_btn.setEnabled(False)
        self.status.setStyleSheet(f"color:{theme.MUTED};")
        self.status.setText(f"Rendering episode {args.episode} → {args.out} …")
        self._worker = _RenderWorker(args)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, ok: bool, message: str) -> None:
        self.render_btn.setEnabled(True)
        if ok:
            size = pathlib.Path(message).stat().st_size / 1e6 if pathlib.Path(message).exists() else 0.0
            self.status.setStyleSheet(f"color:{theme.OK if hasattr(theme, 'OK') else theme.ACCENT};")
            self.status.setText(f"Wrote {message}  ({size:.0f} MB)")
        else:
            self.status.setStyleSheet(f"color:{theme.BAD};")
            self.status.setText(message)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="datasets root to open with")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = RenderGUI(args.root)
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
