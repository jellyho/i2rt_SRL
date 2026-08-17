"""Calibrate the AGENTVIEW camera by grasping a ChArUco board with the gripper (eye-to-hand).

The agentview camera sits too high to co-see a desk board with a wrist camera, so it is calibrated
the other way round: **grasp the board with the gripper** and lift it into agentview's view. The
camera is fixed and the target rides the hand, so agentview + forward kinematics alone recover
``base_T_agentview`` per arm -- no wrist camera, no arm offset, no shared frame (see
:mod:`workstation.lerobot_recorder.charuco`). The wrist cameras are NOT calibrated here; their
mount comes from the CAD ``T_GRIPPER_CAMERA`` constant.

Live agentview preview with the board outline drawn when found. Grasp the board **rigidly and do
not re-grip mid-run** (the solve assumes its pose in the gripper is constant). Move to a few
poses **varying the grip TILT, not just position** (hand-eye needs that or the rotation is
unconstrained), and each pose is captured hands-free: while teleop is ENGAGED and agentview sees
the board, holding the arm still for ~1 s auto-captures, then re-arms once you move away. **Space**
(or the on-screen button) is a manual fallback; a leader-handle trigger is available but off by
default. With both arms available an "arm holding the board" selector picks which gripper's FK a
capture is attributed to -- run one arm, then move the board to the other and switch it.

The solve re-runs after every capture, per arm, with a live convergence trend (RMS over the last
few re-solves). Each arm needs **3+ varied-tilt captures**. Each arm solves in its OWN frame; they
are never pooled (see charuco's module docstring), so a consumer overlaying an arm's action uses
that arm's own extrinsic.

"Save" writes into ``config.yaml`` -- THE single source of truth for the rig -- via a line-range
splice (a ``.bak`` kept, only ``cameras.agentview.extrinsic.<arm>`` + ``calibration.board``
change, comments preserved), not a YAML load/dump round trip.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from i2rt.serving.teleop_common import handle_button_pressed
from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.charuco import (
    AgentviewEyeToHandResult,
    BoardSpec,
    EyeToHandCapture,
    detect_board_pose,
    solve_agentview_extrinsic_eyetohand_per_arm,
    splice_agentview_extrinsic,
    splice_board,
)
from workstation.lerobot_recorder.config import RecorderConfig

if TYPE_CHECKING:
    from i2rt.serving.robot_client import RobotClient

logger = logging.getLogger(__name__)


def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888))


def _draw_detection(img: np.ndarray, corners_px: Optional[np.ndarray]) -> np.ndarray:
    """The frame, with the board's detected corners marked -- or left untouched if not seen.

    A separate copy, not an in-place draw: the caller still needs the clean frame for the actual
    PnP solve, only the *preview* gets decorated.
    """
    if corners_px is None:
        return img
    out = np.ascontiguousarray(img).copy()
    for x, y in corners_px:
        # x, y are numpy float64 here, not Python float: round() on them returns a numpy scalar
        # too (unlike the builtin's int-returning overload ruff assumes), so int() is not
        # redundant -- it is what makes these valid slice bounds below.
        xi, yi = int(round(x)), int(round(y))  # noqa: RUF046
        out[max(0, yi - 3) : yi + 3, max(0, xi - 3) : xi + 3] = (60, 220, 130)
    return out


def _convergence_note(history: List[tuple], *, stable_mm: float = 1.0, window: int = 3) -> str:
    """A trend string from a ``[(n_captures, translation_rms_mm, rotation_rms_deg), ...]`` history,
    one entry per re-solve (see ``CalibrateAgentviewWindow._record``).

    "Converged" is read straight off the numbers: if the last ``window`` re-solves' translation RMS
    swung by less than ``stable_mm``, more captures are not visibly changing the answer. That is a
    statement about what has been seen so far, not a proof the true value is close -- too few
    DIVERSE poses can sit falsely flat. It is still the honest, cheap signal to put on screen live;
    the per-capture disagreement numbers next to it are the deeper check.
    """
    if len(history) < 2:
        return "(need more captures to judge convergence)"
    trend = " -> ".join(f"{t:.1f}" for _, t, _ in history[-5:])
    recent = [t for _, t, _ in history[-window:]]
    stable = len(recent) >= window and (max(recent) - min(recent)) < stable_mm
    verdict = "converged" if stable else "still moving -- keep capturing"
    return f"trend {trend} mm ({verdict})"


class CalibrateAgentviewWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        cams: CameraManager,
        robot: "Optional[RobotClient]",  # None in --mock
        geometries: Dict[str, object],  # {"left": WristCameraGeometry, "right": WristCameraGeometry}
        *,
        board: BoardSpec,
        config_path: Optional[str],
        mock: bool = False,
        capture_buttons: Sequence[str] = (),
        auto_capture: bool = True,
        auto_dwell_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Calibrate agentview -- board grasped on the gripper (eye-to-hand)")
        self.cams = cams
        self.robot = robot
        self.geometries = geometries
        self.arms = list(geometries.keys())  # e.g. ["left", "right"]
        self.board = board
        self.config_path = config_path  # None = no config.yaml found; Save will say so
        self.mock = mock

        # "<side>.<index>" leader-handle buttons (upper=0, lower=1) that trigger a capture, any ONE
        # firing. EMPTY BY DEFAULT: in teleop the robot server already consumes these while engaged
        # (outcome buttons -> homing, fine button -> recentering), so a handle press would move the
        # arm rather than capture. Space always works.
        self.capture_buttons = list(capture_buttons)
        self._capture_btn_prev = False  # rising-edge state across the WHOLE set, not per-button

        # Hands-free capture: while teleop is ENGAGED and the board is in view, holding the arm
        # still for `auto_dwell_s` fires one capture; you must move away before it re-arms. Primary
        # trigger since both hands are on the leaders. See _auto_capture_check.
        self.auto_capture = bool(auto_capture)
        self.auto_dwell_s = float(auto_dwell_s)
        self._auto_still_rad = 0.01  # ~0.6 deg: drift under this over the dwell counts as "still"
        self._auto_move_rad = 0.05  # ~3 deg: motion over this after a capture re-arms the trigger
        self._auto_key: Optional[tuple] = None  # which ready arms the window/rearm state is for
        self._auto_ref_q: Optional[np.ndarray] = None  # joints at the start of the current still window
        self._auto_still_since: Optional[float] = None
        self._auto_rearm_q: Optional[np.ndarray] = None  # joints at last auto-capture; None = armed
        self._engaged: Optional[bool] = None  # teleop engage state from the last obs (None = unknown)

        self._last_agent_det = None
        self._last_q: Dict[str, Optional[np.ndarray]] = {}  # arm -> joints | None
        self.captures: List[EyeToHandCapture] = []
        # One (n_captures, translation_rms_mm, rotation_rms_deg) appended per re-solve that added a
        # new capture -- see _convergence_note and _record.
        self._history: Dict[str, List[tuple]] = {arm: [] for arm in self.arms}
        # What the last successful solve found -- what _on_save writes.
        self._last_results: Dict[str, AgentviewEyeToHandResult] = {}

        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)  # ~6-7 Hz: detection is the expensive part, not the camera read

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.setStyleSheet(theme.QSS)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.agent_view = QtWidgets.QLabel("agentview")
        self.agent_view.setMinimumSize(480, 360)
        self.agent_view.setAlignment(QtCore.Qt.AlignCenter)
        self.agent_view.setStyleSheet(f"background:#000; border:1px solid {theme.IDLE}; border-radius:8px;")
        root.addWidget(self.agent_view, 1)

        # With both arms available: which arm is currently holding the board. Captures are
        # attributed to it (the tool cannot tell from the camera which gripper the board is in).
        self.arm_selector: Optional[QtWidgets.QComboBox] = None
        if len(self.arms) > 1:
            selrow = QtWidgets.QHBoxLayout()
            selrow.addWidget(QtWidgets.QLabel("Board held by:"))
            self.arm_selector = QtWidgets.QComboBox()
            self.arm_selector.addItems(self.arms)
            self.arm_selector.currentTextChanged.connect(self._on_active_arm_changed)
            selrow.addWidget(self.arm_selector)
            selrow.addStretch(1)
            root.addLayout(selrow)

        self.status = QtWidgets.QLabel("waiting for the grasped board...")
        self.status.setStyleSheet(f"color:{theme.MUTED};")
        root.addWidget(self.status)

        row = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton("Capture  [Space]")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._on_capture)
        # Space is the intended way to capture (both hands are usually busy). Bound on the window,
        # and only fires when the button itself is enabled -- same readiness gate as clicking it.
        self._capture_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self._capture_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._capture_shortcut.activated.connect(lambda: self.capture_btn.isEnabled() and self._on_capture())
        self.remove_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._on_remove)
        self.solve_btn = QtWidgets.QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        for b in (self.capture_btn, self.remove_btn, self.solve_btn, self.save_btn):
            row.addWidget(b)
        self.auto_chk = QtWidgets.QCheckBox("Auto-capture (hold still)")
        self.auto_chk.setChecked(self.auto_capture)
        self.auto_chk.toggled.connect(self._on_auto_toggled)
        row.addWidget(self.auto_chk)
        root.addLayout(row)

        self.capture_list = QtWidgets.QListWidget()
        self.capture_list.setMaximumHeight(160)
        root.addWidget(self.capture_list)

        # Read-only scrolling log (monospace so the RMS columns line up), not a QLabel that would
        # clip the multi-line per-arm report.
        self.result_label = QtWidgets.QPlainTextEdit("not solved yet")
        self.result_label.setReadOnly(True)
        self.result_label.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.result_label.setMinimumHeight(120)
        mono = QtGui.QFont("monospace", 12)
        mono.setStyleHint(QtGui.QFont.Monospace)
        self.result_label.setFont(mono)
        root.addWidget(self.result_label, 1)

    def _active_arm(self) -> str:
        """The arm currently holding the board -- the selector's value, or the sole arm."""
        if self.arm_selector is not None:
            return self.arm_selector.currentText()
        return self.arms[0]

    def _on_active_arm_changed(self, _arm: str) -> None:
        # Switching which arm holds the board restarts the auto-capture dwell (the watched joints
        # changed), like toggling auto-capture does.
        self._auto_still_since = self._auto_ref_q = self._auto_rearm_q = self._auto_key = None

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        try:
            frames = self.cams.read()
        except Exception as e:
            self.status.setText(f"camera read failed: {e}")
            return

        robot_obs = self._get_robot_obs()
        agent_img = frames.get("agentview")
        agent_intr = self.cams.intrinsics("agentview")
        self._last_agent_det = (
            detect_board_pose(agent_img, self.board, agent_intr) if agent_img is not None and agent_intr else None
        )
        if agent_img is not None:
            preview = _draw_detection(agent_img, self._last_agent_det.corners_px if self._last_agent_det else None)
            self.agent_view.setPixmap(_np_to_pixmap(preview).scaled(self.agent_view.size(), QtCore.Qt.KeepAspectRatio))

        for arm in self.arms:
            self._last_q[arm] = self._joints_from_obs(robot_obs, arm)
        arm = self._active_arm()
        ready = self._last_agent_det is not None and self._last_q.get(arm) is not None
        ready_arms = [arm] if ready else []

        self._engaged = robot_obs.get("active") if "active" in robot_obs else None
        self.capture_btn.setEnabled(bool(ready_arms))
        triggers = "Space" + (f"/leader button ({'/'.join(self.capture_buttons)})" if self.capture_buttons else "")
        auto = " · auto-capture: hold still" if self.auto_capture else ""
        if self._last_agent_det is None:
            self.status.setText(f"[{arm}] not ready -- lift the grasped board into agentview's view")
        elif self._last_q.get(arm) is None:
            self.status.setText(f"[{arm}] agentview sees the board but no joints for this arm yet")
        else:
            engaged_note = "" if self._engaged is None else (" [engaged]" if self._engaged else " [NOT engaged]")
            self.status.setText(f"[{arm}] ready ({triggers}){auto}{engaged_note} -- vary the grip TILT between poses")

        self._check_capture_button(robot_obs)
        self._auto_capture_check(ready_arms, self._engaged, time.monotonic())

    def _on_auto_toggled(self, on: bool) -> None:
        self.auto_capture = bool(on)
        self._auto_still_since = self._auto_ref_q = self._auto_rearm_q = self._auto_key = None

    def _auto_capture_check(self, ready_arms: List[str], engaged: Optional[bool], now: float) -> None:
        """Fire one capture when the active arm has been held still, board in view, while engaged.

        "Held still" keeps each capture's image and joint pose from the same instant, and naturally
        excludes homing/ramp motion. Re-arms only after the arm moves ``_auto_move_rad`` away, so
        holding a pose does not machine-gun captures. ``now`` is injected (``time.monotonic()`` from
        the tick) so the dwell logic is testable without real time. ``engaged`` gates it to active
        teleoperation when the robot reports engage state; None means "do not gate on it".
        """
        capturable = self.auto_capture and bool(ready_arms) and self.capture_btn.isEnabled() and engaged is not False
        if not capturable:
            self._auto_still_since = self._auto_ref_q = None
            return
        key = tuple(ready_arms)
        q = np.concatenate([np.asarray(self._last_q[a], dtype=float) for a in ready_arms])
        if key != self._auto_key:  # the active arm changed -> start fresh
            self._auto_key, self._auto_ref_q, self._auto_still_since, self._auto_rearm_q = key, q, now, None
            return
        if self._auto_rearm_q is not None:  # captured here already; wait until the arm moves away
            if np.max(np.abs(q - self._auto_rearm_q)) > self._auto_move_rad:
                self._auto_rearm_q = None  # moved -> armed again; fall through to open a new window
            else:
                return
        if self._auto_ref_q is None or np.max(np.abs(q - self._auto_ref_q)) > self._auto_still_rad:
            self._auto_ref_q, self._auto_still_since = q, now  # drifted -> restart the dwell window
            return
        if now - self._auto_still_since >= self.auto_dwell_s:
            self._on_capture()
            self._auto_rearm_q, self._auto_ref_q, self._auto_still_since = q, None, None

    def _get_robot_obs(self) -> dict:
        if self.mock or self.robot is None:
            return {}
        try:
            return self.robot.get_observation() or {}
        except Exception as e:
            logger.warning("could not read robot observation: %s", e)
            return {}

    def _joints_from_obs(self, obs: dict, arm: str) -> Optional[np.ndarray]:
        if self.mock or self.robot is None:
            return np.zeros(7, np.float64)
        side = obs.get(arm)
        if not side or side.get("pos") is None:
            return None
        return np.asarray(side["pos"], dtype=np.float64)

    def _check_capture_button(self, obs: dict) -> None:
        """Fire a capture on the rising edge of any configured leader-handle button.

        Same ``pressed and not was`` edge test the recorder uses, against ITS OWN previous-state
        flag. Treated as one aggregate signal across every configured button, so pressing two at
        once still fires exactly once.
        """
        if self.mock or self.robot is None or not self.capture_buttons:
            return
        side_buttons = {arm: (obs.get(arm, {}) or {}).get("buttons", []) for arm in self.arms}
        pressed = any(handle_button_pressed(side_buttons, key) for key in self.capture_buttons)
        if pressed and not self._capture_btn_prev and self.capture_btn.isEnabled():
            self._on_capture()
        self._capture_btn_prev = pressed

    # ------------------------------------------------------------------ actions
    def _on_capture(self) -> None:
        """Bank one capture for the active arm (agentview PnP of the grasped board + FK flange pose)."""
        agent = self._last_agent_det
        arm = self._active_arm()
        q = self._last_q.get(arm)
        if agent is None or q is None:
            self.status.setText("capture pressed but agentview does not see the grasped board -- see status above")
            return
        cap = EyeToHandCapture(
            arm=arm,
            base_t_flange=self.geometries[arm].flange_pose(q),  # FK only, no camera extrinsic
            agentview_t_board=agent.cam_t_board,
            agentview_reproj_error_px=agent.reproj_error_px,
        )
        self.captures.append(cap)
        n_arm = sum(1 for c in self.captures if c.arm == arm)
        self.capture_list.addItem(
            f"#{len(self.captures)}  [{arm}]  agentview err {agent.reproj_error_px:.2f}px  "
            f"(arm total {n_arm})  ({time.strftime('%H:%M:%S')})"
        )
        self._solve_and_report(warn_if_empty=False)

    def _on_remove(self) -> None:
        row = self.capture_list.currentRow()
        if row < 0:
            return
        del self.captures[row]
        self.capture_list.takeItem(row)
        self._solve_and_report(warn_if_empty=False)

    def _on_solve(self) -> None:
        self._solve_and_report(warn_if_empty=True)

    def _record(self, history: List[tuple], n: int, t_rms: float, r_rms: float) -> None:
        """Append to a convergence history, skipping a duplicate entry when re-solving found no new
        capture for that arm (e.g. removing a capture from a DIFFERENT arm still re-solves this one
        at the same n as last time)."""
        if not history or history[-1][0] != n:
            history.append((n, t_rms, r_rms))

    def _solve_and_report(self, *, warn_if_empty: bool) -> None:
        """Re-run the per-arm eye-to-hand solve and refresh the on-screen report.

        Called after every capture/removal (silently) and by the Solve button (``warn_if_empty``).
        Each arm needs 3+ varied-tilt captures; each solves in its own frame, never pooled.
        """
        counts = {arm: sum(1 for c in self.captures if c.arm == arm) for arm in self.arms}
        if all(n < 3 for n in counts.values()):
            if warn_if_empty:
                QtWidgets.QMessageBox.warning(
                    self, "Not enough captures", f"eye-to-hand needs 3+ varied-tilt captures per arm; have {counts}."
                )
            self.result_label.setPlainText("not solved yet -- capture 3+ varied-tilt poses per arm")
            self.save_btn.setEnabled(False)
            return

        lines = []
        results = solve_agentview_extrinsic_eyetohand_per_arm(self.captures)
        for arm in self.arms:
            if arm in results:
                r = results[arm]
                self._record(self._history[arm], r.n_captures, r.translation_rms_mm, r.rotation_rms_deg)
                quality = "good" if r.rotation_rms_deg < 1.0 and r.translation_rms_mm < 10.0 else "CHECK POSES"
                lines.append(
                    f"[agentview {arm}] eye-to-hand n={r.n_captures}  grip-consistency RMS "
                    f"{r.translation_rms_mm:.2f} mm / {r.rotation_rms_deg:.3f} deg  -- {quality}  "
                    f"{_convergence_note(self._history[arm])}"
                )
            elif counts.get(arm):
                lines.append(f"[agentview {arm}] {counts[arm]} capture(s) -- need 3+ (varied grip tilt) to solve")

        self.result_label.setPlainText("\n".join(lines) if lines else "not solved yet")
        self._last_results = results
        self.save_btn.setEnabled(bool(results))

    def _on_save(self) -> None:
        """Write the solved per-arm agentview extrinsics into config.yaml (line-range splice; a
        ``.bak`` kept, only ``cameras.agentview.extrinsic.<arm>`` + ``calibration.board`` change)."""
        if not self._last_results:
            return
        self._commit_config(self._build_config)

    def _commit_config(self, build: Callable[[str, str], tuple]) -> None:
        """Config.yaml write path: confirm dialog naming the file, a ``.bak`` of the original kept
        first, then only the blocks ``build`` touches change. ``build(original, calibrated_at) ->
        (updated, written)`` does the splicing; ``written`` names the blocks for the status line.
        """
        if not self.config_path:
            QtWidgets.QMessageBox.critical(self, "No config.yaml", "No config.yaml was found to write into.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Write config.yaml?",
            f"Write the solved agentview extrinsic(s) into\n{self.config_path}\n\n"
            "Only cameras.agentview.extrinsic and calibration.board change; the rest of the file "
            "(comments included) is left alone. A .bak copy of the current file is kept first.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        calibrated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.config_path, encoding="utf-8") as fh:
                original = fh.read()
            updated, written = build(original, calibrated_at)
            with open(self.config_path + ".bak", "w", encoding="utf-8") as fh:
                fh.write(original)
            with open(self.config_path, "w", encoding="utf-8") as fh:
                fh.write(updated)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", str(e))
            self.status.setText(f"write failed: {e}")
            return
        self.status.setText(f"wrote {', '.join(written)} -> {self.config_path} (backup at .bak)")

    def _build_config(self, original: str, calibrated_at: str) -> tuple:
        written: List[str] = []
        updated = original
        for arm, result in self._last_results.items():
            updated = splice_agentview_extrinsic(updated, arm, result, calibrated_at=calibrated_at)
            written.append(f"agentview.extrinsic.{arm}")
        updated = splice_board(updated, self.board, calibrated_at=calibrated_at)
        written.append("calibration.board")
        return updated, written

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.timer.stop()
            self.cams.stop()
        finally:
            super().closeEvent(event)


def run(
    cfg: RecorderConfig,
    robot: "Optional[RobotClient]",
    geometries: Dict[str, object],
    *,
    board: BoardSpec,
    config_path: Optional[str],
    mock: bool = False,
    capture_buttons: Sequence[str] = (),
    auto_capture: bool = True,
    auto_dwell_s: float = 1.0,
) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cams = CameraManager(cfg)
    cams.start()
    win = CalibrateAgentviewWindow(
        cams,
        robot,
        geometries,
        board=board,
        config_path=config_path,
        mock=mock,
        capture_buttons=capture_buttons,
        auto_capture=auto_capture,
        auto_dwell_s=auto_dwell_s,
    )
    win.resize(1000, 800)
    win.show()
    return app.exec_()
