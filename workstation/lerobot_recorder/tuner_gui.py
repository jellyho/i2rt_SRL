"""PyQt tuner window: live feeds + per-camera exposure sliders + brightness match.

Launched by ``workstation/yam-data tune``. One column per camera: the live frame, its
mean luma / clipped-highlight readout and delta against the reference camera, then a
slider per supported sensor control. Changes hit the hardware immediately, so you can
watch the feeds converge.

Numbers are never copied between camera models — see
:mod:`workstation.lerobot_recorder.exposure_tuner` for why D405 and D455 exposure
units differ by 100x. The reference picker sets which camera the deltas measure
against; "Match to reference" steps the other cameras' exposure toward its luma.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig
from workstation.lerobot_recorder.exposure_tuner import (
    TOGGLES,
    brightness_report,
    format_options_yaml,
    same_model_groups,
    sensor_controls,
    set_control,
    splice_camera_options,
    suggest_exposure,
)

logger = logging.getLogger(__name__)


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


class CameraColumn(QtWidgets.QGroupBox):
    """Live view + metrics + sliders for a single camera."""

    def __init__(self, spec: CameraSpec, parent=None) -> None:
        super().__init__(spec.key, parent)
        self.spec = spec
        self.sensor = None
        self.model = ""  # RealSense product name, e.g. "Intel RealSense D405"
        self.controls: Dict[str, tuple] = {}
        self.on_changed = None  # set by TunerWindow to mirror edits onto linked cameras
        self._sliders: Dict[str, QtWidgets.QWidget] = {}
        self._spins: Dict[str, QtWidgets.QSpinBox] = {}
        self._guard = False  # suppress valueChanged while we set sliders programmatically

        lay = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QLabel("waiting for frames…")
        self.view.setMinimumSize(320, 240)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet(f"background:#000; border:1px solid {theme.IDLE}; border-radius:8px;")
        lay.addWidget(self.view, 1)

        self.metrics = QtWidgets.QLabel("luma —")
        self.metrics.setAlignment(QtCore.Qt.AlignCenter)
        self.metrics.setStyleSheet("font-size:22px;")
        lay.addWidget(self.metrics)

        self.model_lbl = QtWidgets.QLabel("")
        self.model_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.model_lbl.setStyleSheet(f"color:{theme.MUTED}; font-size:18px;")
        lay.addWidget(self.model_lbl)

        self.form = QtWidgets.QFormLayout()
        self.form.setLabelAlignment(QtCore.Qt.AlignRight)
        lay.addLayout(self.form)
        lay.addStretch(0)

    # ------------------------------------------------------------------ hardware bind
    def bind(self, device) -> None:
        """Discover the tunable controls on this camera and build a row per control."""
        try:
            self.sensor, self.controls = sensor_controls(device)
        except Exception as e:
            self.model_lbl.setText(f"no tunable controls: {e}")
            return
        import pyrealsense2 as rs

        self.model = device.get_info(rs.camera_info.name)
        sensor_name = self.sensor.get_info(rs.camera_info.name)
        # the sensor matters: exposure units differ between RGB (100us) and stereo (1us)
        self.model_lbl.setText(f"{self.model} — via {sensor_name}")
        # Seed each row from config.yaml (the source of truth) when it specifies the
        # option, not from the live sensor read. Otherwise "Write to config.yaml" would
        # round-trip whatever the sensor happens to report (e.g. white_balance drifting
        # back to a driver default), silently clobbering a hand-tuned value the operator
        # never touched. Config-seeded controls also get re-asserted onto the sensor.
        cfg_opts = self.spec.options or {}
        for opt, (lo, hi, cur) in self.controls.items():
            if opt in cfg_opts:
                cur = min(hi, max(lo, float(cfg_opts[opt])))
                set_control(self.sensor, opt, cur)
            self._add_row(opt, lo, hi, cur)

    def _add_row(self, opt: str, lo: float, hi: float, cur: float) -> None:
        if opt in TOGGLES:
            w = QtWidgets.QCheckBox(opt.replace("enable_auto_", "auto "))
            w.setChecked(bool(cur))
            w.stateChanged.connect(lambda _s, o=opt: self._on_change(o))
            self.form.addRow(w)
            self._sliders[opt] = w
            return

        # A slider alone is unusable here: the 26px theme font leaves the row only
        # ~190px of travel, so one pixel is ~58 exposure units on a D455 and ~950 on
        # a D405's stereo sensor. The spinbox gives exact values (and arrow-key steps),
        # the slider stays for coarse sweeps, and both drive the same option.
        box = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(box)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)

        caption = QtWidgets.QLabel(f"{opt}  [{int(lo)}..{int(hi)}]")
        caption.setStyleSheet(f"color:{theme.MUTED}; font-size:18px;")
        vbox.addWidget(caption)

        row = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        w = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        w.setMinimum(int(lo))
        w.setMaximum(int(hi))
        w.setValue(int(cur))
        # page-step ~5% so clicking the groove is a sane nudge, not a jump to the rail
        w.setPageStep(max(1, int((hi - lo) / 20)))
        w.setSingleStep(max(1, int((hi - lo) / 200)))
        w.setMinimumWidth(180)

        spin = QtWidgets.QSpinBox()
        spin.setRange(int(lo), int(hi))
        spin.setValue(int(cur))
        spin.setSingleStep(max(1, int((hi - lo) / 200)))
        spin.setMinimumWidth(150)
        spin.setStyleSheet("font-size:20px; padding:4px 6px;")
        spin.setKeyboardTracking(False)  # apply on commit, not on every keystroke

        # keep the pair in sync without re-entering _on_change twice
        def _from_slider(v, o=opt, sp=spin):
            if self._guard:
                return
            self._guard = True
            sp.setValue(v)
            self._guard = False
            self._on_change(o)

        def _from_spin(v, o=opt, sl=w):
            if self._guard:
                return
            self._guard = True
            sl.setValue(v)
            self._guard = False
            self._on_change(o)

        w.valueChanged.connect(_from_slider)
        spin.valueChanged.connect(_from_spin)

        hbox.addWidget(w, 1)
        hbox.addWidget(spin)
        vbox.addWidget(row)
        self.form.addRow(box)

        self._sliders[opt] = w
        self._spins[opt] = spin

    # ---------------------------------------------------------------------- interaction
    def _on_change(self, opt: str) -> None:
        if self._guard or self.sensor is None:
            return
        value = self.value_of(opt)
        if self.on_changed is not None:  # let the window mirror this to linked cameras
            self.on_changed(self.spec.key, opt, value)
        # Manual exposure/gain is ignored while auto-exposure owns the sensor, so turn
        # auto off as soon as the operator touches one — otherwise the slider looks
        # broken (it moves, nothing changes).
        if opt in ("exposure", "gain"):
            auto = self._sliders.get("enable_auto_exposure")
            if isinstance(auto, QtWidgets.QCheckBox) and auto.isChecked():
                self._guard = True
                auto.setChecked(False)
                self._guard = False
                set_control(self.sensor, "enable_auto_exposure", 0.0)
        set_control(self.sensor, opt, value)

    def value_of(self, opt: str) -> float:
        w = self._sliders.get(opt)
        if isinstance(w, QtWidgets.QCheckBox):
            return 1.0 if w.isChecked() else 0.0
        if isinstance(w, QtWidgets.QSlider):
            return float(w.value())
        return 0.0

    def set_value(self, opt: str, value: float) -> None:
        """Move a slider AND the hardware (used by the auto-match button)."""
        w = self._sliders.get(opt)
        if w is None or self.sensor is None:
            return
        self._guard = True
        if isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(int(value))
            if opt in self._spins:
                self._spins[opt].setValue(int(value))
        self._guard = False
        set_control(self.sensor, opt, value)

    def options(self) -> Dict[str, float]:
        return {opt: self.value_of(opt) for opt in self.controls}

    def show_frame(self, frame: np.ndarray, stats: dict, reference: bool) -> None:
        pix = _np_to_pixmap(frame).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pix)
        luma, clip, delta = stats["luma"], stats["clipped"], stats["delta"]
        parts = [f"luma <b>{luma:.0f}</b>"]
        if clip > 0.02:  # only nag when clipping is real
            color = theme.BAD if clip > 0.10 else theme.WARN
            parts.append(f"<span style='color:{color}'>clip {clip * 100:.0f}%</span>")
        if reference:
            parts.append(f"<span style='color:{theme.ACCENT}'>reference</span>")
        elif delta is not None:
            color = theme.OK if abs(delta) <= 5 else (theme.WARN if abs(delta) <= 20 else theme.BAD)
            parts.append(f"<span style='color:{color}'>Δ {delta:+.0f}</span>")
        self.metrics.setText("  ·  ".join(parts))


class TunerWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: RecorderConfig, config_path: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.setWindowTitle("YAM — camera exposure tuner")
        self.setStyleSheet(theme.QSS)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.columns: Dict[str, CameraColumn] = {}
        strip = QtWidgets.QHBoxLayout()
        for spec in cfg.cameras:
            col = CameraColumn(spec)
            self.columns[spec.key] = col
            strip.addWidget(col, 1)
        root.addLayout(strip, 1)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("reference:"))
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems([s.key for s in cfg.cameras])
        # default to the DARKEST camera as reference: matching up into clipping
        # destroys highlight detail, matching down is recoverable.
        if any(s.key == "agentview" for s in cfg.cameras):
            self.ref_combo.setCurrentText("agentview")
        bar.addWidget(self.ref_combo)

        # Same-model cameras (the two D405 wrists) take identical numbers, so tuning
        # them as one pair is both safe and usually what you want.
        self.link_chk = QtWidgets.QCheckBox("link same-model cameras")
        self.link_chk.setChecked(True)
        self.link_chk.setToolTip("Mirror every slider change onto other cameras of the same model")
        bar.addWidget(self.link_chk)
        bar.addStretch(1)

        self.match_btn = QtWidgets.QPushButton("Match to reference")
        self.match_btn.setToolTip("Step every other camera's exposure toward the reference luma")
        self.match_btn.clicked.connect(self._match_once)
        bar.addWidget(self.match_btn)

        self.copy_btn = QtWidgets.QPushButton("Copy YAML")
        self.copy_btn.clicked.connect(self._copy_yaml)
        bar.addWidget(self.copy_btn)

        self.write_btn = QtWidgets.QPushButton("Write to config.yaml")
        self.write_btn.clicked.connect(self._write_config)
        self.write_btn.setEnabled(bool(config_path))
        if not config_path:
            self.write_btn.setToolTip("no config.yaml found — use Copy YAML instead")
        bar.addWidget(self.write_btn)
        root.addLayout(bar)

        self.status = QtWidgets.QLabel("starting cameras…")
        self.status.setStyleSheet(f"color:{theme.MUTED}; font-size:20px;")
        root.addWidget(self.status)

        self.cams = CameraManager(cfg)
        self.cams.start()
        self._bind_devices()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)  # 10 Hz preview is plenty for judging brightness

    def _bind_devices(self) -> None:
        """Attach each column to its opened device so the sliders reach real hardware."""
        if self.cfg.mock:
            self.status.setText("mock mode — synthetic frames, sliders inert")
            return
        try:
            import pyrealsense2 as rs

            devices = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
        except Exception as e:
            self.status.setText(f"RealSense unavailable: {e}")
            return
        for key, col in self.columns.items():
            serial = self.cams._serials.get(key) or col.spec.serial
            device = devices.get(serial)
            if device is None:
                col.model_lbl.setText("camera not found")
                continue
            col.bind(device)
            col.on_changed = self._mirror
        groups = same_model_groups({k: c.model for k, c in self.columns.items()})
        if groups:
            pairs = "; ".join(f"{m.replace('Intel RealSense ', '')}: {'+'.join(ks)}" for m, ks in groups.items())
            self.status.setText(f"ready — type an exact value in the spinbox or drag the slider.  linked → {pairs}")
        else:
            self.status.setText("ready — type an exact value in the spinbox or drag the slider")

    def _mirror(self, source_key: str, opt: str, value: float) -> None:
        """Copy one control change onto every other camera of the SAME model.

        Restricted to identical models on purpose: exposure/gain scales are per model,
        so mirroring a D405 value onto a D455 (or vice versa) would be nonsense.
        """
        if not self.link_chk.isChecked():
            return
        source = self.columns.get(source_key)
        if source is None or not source.model:
            return
        for key, col in self.columns.items():
            if key == source_key or col.model != source.model or opt not in col.controls:
                continue
            if col.value_of(opt) != value:
                col.set_value(opt, value)

    def _tick(self) -> None:
        frames = self.cams.read()
        ref = self.ref_combo.currentText()
        report = brightness_report(frames, reference=ref)
        for key, col in self.columns.items():
            if key in frames:
                col.show_frame(frames[key], report[key], reference=(key == ref))

    # ------------------------------------------------------------------------ actions
    def _match_once(self) -> None:
        """One convergence step per camera (or per linked group) toward the reference luma.

        When cameras are linked, a group gets ONE exposure derived from its MEAN luma,
        so both wrists keep identical settings instead of drifting to two values that
        merely happen to match the reference individually.
        """
        frames = self.cams.read()
        ref = self.ref_combo.currentText()
        report = brightness_report(frames, reference=ref)
        target = report.get(ref, {}).get("luma")
        if target is None:
            return

        linked = self.link_chk.isChecked()
        # group cameras that must move together: same model when linked, else alone
        buckets: Dict[str, List[str]] = {}
        for key, col in self.columns.items():
            if key == ref or "exposure" not in col.controls:
                continue
            bucket = col.model if (linked and col.model) else key
            buckets.setdefault(bucket, []).append(key)

        moved = []
        for keys in buckets.values():
            cols = [self.columns[k] for k in keys]
            luma = sum(report[k]["luma"] for k in keys) / len(keys)
            lo, hi, _cur = cols[0].controls["exposure"]
            current = cols[0].value_of("exposure")
            new = suggest_exposure(current, luma, target, (lo, hi))
            for col in cols:
                col.set_value("exposure", new)
            moved.append(f"{'+'.join(keys)} {int(current)}→{int(new)}")
        if not moved:
            self.status.setText("nothing to match")
            return
        self.status.setText(
            f"matched toward {ref} (luma {target:.0f}): " + ", ".join(moved) + " — click again to refine"
        )

    def _yaml_text(self) -> str:
        blocks = []
        for key, col in self.columns.items():
            opts = col.options()
            if not opts:
                continue
            opts["serial"] = self.cams._serials.get(key, col.spec.serial)
            blocks.append(format_options_yaml(key, opts))
        return "cameras:\n" + "\n".join(blocks) if blocks else ""

    def _copy_yaml(self) -> None:
        text = self._yaml_text()
        if not text:
            self.status.setText("no tunable controls to export")
            return
        QtWidgets.QApplication.clipboard().setText(text)
        self.status.setText("YAML copied to clipboard")
        print(text)  # also to the terminal, so it survives the clipboard

    def _write_config(self) -> None:
        if not self.config_path:
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Write config.yaml?",
            f"Update the camera options in\n{self.config_path}\n\n"
            "Only the camera entries change; the rest of the file (comments included) is left alone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            updated, skipped = text, []
            for key, col in self.columns.items():
                opts = col.options()
                if not opts:
                    continue
                serial = self.cams._serials.get(key, col.spec.serial)
                try:
                    updated = splice_camera_options(updated, key, opts, serial=serial)
                except ValueError as e:
                    skipped.append(f"{key} ({e})")
            # keep a .bak so a bad write is always recoverable
            with open(self.config_path + ".bak", "w", encoding="utf-8") as fh:
                fh.write(text)
            with open(self.config_path, "w", encoding="utf-8") as fh:
                fh.write(updated)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", str(e))
            self.status.setText(f"write failed: {e}")
            return
        msg = f"wrote {self.config_path} (backup at {self.config_path}.bak)"
        if skipped:
            msg += " — skipped: " + ", ".join(skipped)
        self.status.setText(msg)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        try:
            self.timer.stop()
            self.cams.stop()
        finally:
            super().closeEvent(event)


def run(cfg: RecorderConfig, config_path: Optional[str] = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = TunerWindow(cfg, config_path)
    win.resize(1500, 900)
    win.show()
    return app.exec_()
