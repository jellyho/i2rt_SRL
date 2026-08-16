"""Calibrate the whole camera rig against one ChArUco board sitting on the desk.

From a single board (never moved, never attached to the robot) this recovers, per re-solve, in
order -- see :mod:`workstation.lerobot_recorder.charuco` for the geometry:

1. **each wrist camera's own extrinsic** (``gripper_T_camera``), by eye-in-hand hand-eye
   calibration -- so the mount is measured, not trusted from the unverified CAD constant;
2. **each agentview extrinsic**, chained through that arm's (now measured) wrist extrinsic;
3. **the left<->right arm offset**, and the fused/cross-checked shared-frame answer.

YAM is bimanual, so BOTH wrist cameras can bridge -- whichever one currently has the board in
view. Live preview of agentview + both wrists with the board outline drawn when found; hit
**Space**, press a **leader-handle button** (default: the "lower" button on either handle,
``left.1``/``right.1`` -- both hands are usually busy holding the robot in position by the time a
pose is worth capturing, so a keyboard is not always reachable), or click Capture -- all three are
gated by the exact same readiness check, so none of them can capture out of a bad state. Grabs one
sample per wrist camera that currently sees the board (0, 1, or 2 -- both arms do not have to be
posed at once, and when they are, one press banks two independent estimates instead of one). Move
the arm(s) to a few different poses -- **varying wrist TILT, not just position**, which hand-eye
needs -- and capture at each. Whenever a press happens to catch BOTH wrist cameras seeing the
board AT ONCE, it also banks an ``ArmPairCapture`` for free towards the arm-to-arm offset; this
needs both arms actually posed together for at least a couple of presses.

The solve re-runs automatically after every capture -- no need to click "Solve" and wait; watch
the numbers on screen while you work instead of guessing how many more poses to collect. Hand-eye
(step 1) needs 3+ varied-orientation captures per arm; until an arm reaches that, steps 2-3 fall
back to the CAD wrist extrinsic (and say so) so they still produce an early, less-trustworthy
answer. Everything is PER ARM, never pooled: each arm's ``WristCameraGeometry`` is its own MJCF
with no known transform to the other's -- there is no shared "robot base" frame anywhere in this
codebase (see ``charuco``'s module docstring), which is exactly what the arm offset (step 3)
recovers. Each line carries a **convergence trend** (the RMS from the last several re-solves) so
"still dropping" reads differently from "flat for the last 3".

"Save" writes into ``config.yaml`` -- THE single source of truth for the rig (camera serials,
robot host, button map, ... all already live there). A confirmation dialog names the file first,
a ``.bak`` copy is kept before writing, and only the touched blocks change --
``cameras.wrist_<arm>.extrinsic`` (the hand-eye mount, overriding CAD), ``cameras.agentview
.extrinsic.<arm>`` per arm, ``cameras.agentview.extrinsic.unified`` (the fused answer) when both
arms solved, and ``robot.arm_offset`` -- via a line-range splice (see ``charuco.splice_*``), not
a YAML load/dump round trip that would strip every comment and reflow the whole document. All
blocks are written from the SAME live solve in one call, so nothing this tool writes can disagree
with anything else it writes.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from yam_policy.viz import T_GRIPPER_CAMERA

from i2rt.serving.teleop_common import handle_button_pressed
from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.charuco import (
    ArmOffsetResult,
    ArmPairCapture,
    BoardSpec,
    CalibrationResult,
    Capture,
    UnifiedRigCalibration,
    WristExtrinsicResult,
    detect_board_pose,
    solve_agentview_extrinsic_per_arm,
    solve_arm_offset,
    solve_wrist_extrinsic_per_arm,
    splice_agentview_extrinsic,
    splice_agentview_unified,
    splice_arm_offset,
    splice_board,
    splice_wrist_extrinsic,
    unify_rig_calibration,
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
    PnP solve on ``Capture``, only the *preview* gets decorated.
    """
    if corners_px is None:
        return img
    out = np.ascontiguousarray(img).copy()
    for x, y in corners_px:
        # x, y are numpy float64 here, not Python float: round() on them returns a numpy
        # scalar too (unlike the builtin's int-returning overload ruff assumes), so int() is
        # not redundant -- it is what makes these valid slice bounds below.
        xi, yi = int(round(x)), int(round(y))  # noqa: RUF046
        out[max(0, yi - 3) : yi + 3, max(0, xi - 3) : xi + 3] = (60, 220, 130)
    return out


def _convergence_note(history: List[tuple], *, stable_mm: float = 1.0, window: int = 3) -> str:
    """A trend string from a ``[(n_captures, translation_rms_mm, rotation_rms_deg), ...]``
    history, one entry per re-solve (see ``CalibrateAgentviewWindow._record``).

    "Converged" is read straight off the numbers, not a separate model: if the last ``window``
    re-solves' translation RMS swung by less than ``stable_mm``, more captures are not visibly
    changing the answer. That is a statement about what has been seen so far, not a proof the
    true value is close -- a rig with too few DIVERSE poses can sit falsely flat (e.g. every
    capture from nearly the same angle). It is still the honest, cheap signal to put on screen
    live; the per-capture disagreement numbers next to it are the deeper check.
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
        capture_buttons: Sequence[str] = ("left.1", "right.1"),
    ) -> None:
        super().__init__()
        self.setWindowTitle("Calibrate rig -- wrist extrinsics, agentview, arm offset (one desk board)")
        self.cams = cams
        self.robot = robot
        self.geometries = geometries
        self.arms = list(geometries.keys())  # e.g. ["left", "right"]
        self.board = board
        self.config_path = config_path  # None = no config.yaml found; Save will say so
        self.mock = mock
        # "<side>.<index>" leader-handle buttons (upper=0, lower=1 -- same convention as
        # config.py's button_map / DEFAULT_TELEOP_BUTTON_OUTCOMES), any ONE of which capturing:
        # whichever hand is free. Empty disables the handle trigger (Space still works).
        self.capture_buttons = list(capture_buttons)
        self._capture_btn_prev = False  # rising-edge state across the WHOLE set, not per-button

        self._last_agent_det = None
        self._last_wrist_det: Dict[str, object] = {}  # arm -> Detection | None
        self._last_q: Dict[str, Optional[np.ndarray]] = {}  # arm -> joints | None
        self.captures: List[Capture] = []
        # Only ever appended alongside a regular Capture (see _on_capture), and only when BOTH
        # wrist cameras saw the board in the SAME tick -- see ArmPairCapture's docstring for why
        # "same tick" (board unmoved) is what makes the offset valid at all.
        self.pair_captures: List[ArmPairCapture] = []
        # One (n_captures, translation_rms_mm, rotation_rms_deg) appended per re-solve that
        # actually added a new capture -- see _convergence_note and _record.
        self._history: Dict[str, List[tuple]] = {arm: [] for arm in self.arms}
        self._wrist_history: Dict[str, List[tuple]] = {arm: [] for arm in self.arms}
        self._offset_history: List[tuple] = []

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

        views = QtWidgets.QHBoxLayout()
        self.agent_view = QtWidgets.QLabel("agentview")
        self.wrist_views: Dict[str, QtWidgets.QLabel] = {arm: QtWidgets.QLabel(f"wrist_{arm}") for arm in self.arms}
        for lbl in (self.agent_view, *self.wrist_views.values()):
            lbl.setMinimumSize(360, 270)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"background:#000; border:1px solid {theme.IDLE}; border-radius:8px;")
        views.addWidget(self.agent_view)
        for lbl in self.wrist_views.values():
            views.addWidget(lbl)
        root.addLayout(views, 1)

        self.status = QtWidgets.QLabel("waiting for the board...")
        self.status.setStyleSheet(f"color:{theme.MUTED};")
        root.addWidget(self.status)

        row = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton("Capture  [Space / leader button]")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._on_capture)
        # Space, not the button, is the intended way to capture: both hands are usually busy
        # holding the robot in position by the time a pose is worth capturing. Bound on the
        # window (WidgetWithChildrenShortcut default would miss it once focus is on a button/
        # list), and only fires the click when the button itself is enabled -- same readiness
        # gate as clicking it, so this cannot capture out of a bad state either.
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
        root.addLayout(row)

        self.capture_list = QtWidgets.QListWidget()
        self.capture_list.setMaximumHeight(160)
        root.addWidget(self.capture_list)

        self.pair_label = QtWidgets.QLabel("arm-pair samples (for the left/right offset): 0")
        self.pair_label.setStyleSheet(f"color:{theme.MUTED};")
        root.addWidget(self.pair_label)

        self.result_label = QtWidgets.QLabel("not solved yet")
        self.result_label.setStyleSheet("font-size:16px;")
        self.result_label.setWordWrap(True)
        root.addWidget(self.result_label)

        # What the last successful _solve_and_report found -- what _on_save actually writes.
        self._last_wrist: Dict[str, WristExtrinsicResult] = {}
        self._last_results: Dict[str, CalibrationResult] = {}
        self._last_arm_offset: Optional[ArmOffsetResult] = None
        self._last_unified: Optional[UnifiedRigCalibration] = None

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        try:
            frames = self.cams.read()
        except Exception as e:
            self.status.setText(f"camera read failed: {e}")
            return

        robot_obs = self._get_robot_obs()  # one RPC round trip for both arms' joints + buttons

        agent_img = frames.get("agentview")
        agent_intr = self.cams.intrinsics("agentview")
        self._last_agent_det = (
            detect_board_pose(agent_img, self.board, agent_intr) if agent_img is not None and agent_intr else None
        )
        if agent_img is not None:
            preview = _draw_detection(agent_img, self._last_agent_det.corners_px if self._last_agent_det else None)
            self.agent_view.setPixmap(_np_to_pixmap(preview).scaled(self.agent_view.size(), QtCore.Qt.KeepAspectRatio))

        ready_arms = []
        for arm in self.arms:
            key = f"wrist_{arm}"
            img = frames.get(key)
            intr = self.cams.intrinsics(key)
            det = detect_board_pose(img, self.board, intr) if img is not None and intr else None
            self._last_wrist_det[arm] = det
            self._last_q[arm] = self._joints_from_obs(robot_obs, arm)
            if img is not None:
                preview = _draw_detection(img, det.corners_px if det else None)
                self.wrist_views[arm].setPixmap(
                    _np_to_pixmap(preview).scaled(self.wrist_views[arm].size(), QtCore.Qt.KeepAspectRatio)
                )
            if self._last_agent_det is not None and det is not None and self._last_q[arm] is not None:
                ready_arms.append(arm)

        self.capture_btn.setEnabled(bool(ready_arms))
        hotkeys = "Space" + (f" or leader button ({'/'.join(self.capture_buttons)})" if self.capture_buttons else "")
        if self._last_agent_det is None:
            self.status.setText("not ready -- board not seen in agentview")
        elif ready_arms:
            self.status.setText(
                f"ready -- press {hotkeys} to capture: {', '.join(ready_arms)} "
                "(agentview + that wrist both see the board)"
            )
        else:
            self.status.setText("board seen in agentview but neither wrist sees it (or robot link is down)")

        self._check_capture_button(robot_obs)

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

        Same ``pressed and not was`` edge test ``Recorder._scan_buttons`` uses (recorder.py),
        against ITS OWN previous-state flag rather than that class's -- this tool has no
        recorder instance to share one with, and shouldn't: reading its buttons must not
        interact with whatever the recorder GUI is doing with the SAME physical handles on a
        different run. Treated as one aggregate signal across every configured button (not one
        edge test per button), so pressing two at once still fires exactly once.
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
        if self._last_agent_det is None:
            return
        added = 0
        for arm in self.arms:
            det = self._last_wrist_det.get(arm)
            q = self._last_q.get(arm)
            if det is None or q is None:
                continue
            cap = Capture(
                arm=arm,
                # FK ONLY (no camera extrinsic) -- so the solve can hand-eye the mount itself and
                # apply the measured value, not the CAD one baked in here. See charuco docstring.
                base_t_flange=self.geometries[arm].flange_pose(q),
                wrist_t_board=det.cam_t_board,
                agentview_t_board=self._last_agent_det.cam_t_board,
                wrist_reproj_error_px=det.reproj_error_px,
                agentview_reproj_error_px=self._last_agent_det.reproj_error_px,
            )
            self.captures.append(cap)
            self.capture_list.addItem(
                f"#{len(self.captures)}  [{arm}]  wrist err {cap.wrist_reproj_error_px:.2f}px  "
                f"agentview err {cap.agentview_reproj_error_px:.2f}px  ({time.strftime('%H:%M:%S')})"
            )
            added += 1

        if "left" in self.arms and "right" in self.arms:
            left_det, right_det = self._last_wrist_det.get("left"), self._last_wrist_det.get("right")
            left_q, right_q = self._last_q.get("left"), self._last_q.get("right")
            if left_det is not None and right_det is not None and left_q is not None and right_q is not None:
                self.pair_captures.append(
                    ArmPairCapture(
                        left_t_flange=self.geometries["left"].flange_pose(left_q),
                        left_wrist_t_board=left_det.cam_t_board,
                        right_t_flange=self.geometries["right"].flange_pose(right_q),
                        right_wrist_t_board=right_det.cam_t_board,
                        left_reproj_error_px=left_det.reproj_error_px,
                        right_reproj_error_px=right_det.reproj_error_px,
                    )
                )
                self.pair_label.setText(f"arm-pair samples (for the left/right offset): {len(self.pair_captures)}")

        if not added:
            self.status.setText("capture pressed but nothing was ready -- see status above")
        else:
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
        """Append to a convergence history, skipping a duplicate entry when re-solving found no
        new capture for that arm/offset (e.g. removing a capture from a DIFFERENT arm still
        triggers a re-solve of this one, at the same n as last time)."""
        if not history or history[-1][0] != n:
            history.append((n, t_rms, r_rms))

    def _solve_and_report(self, *, warn_if_empty: bool) -> None:
        """Re-run every solve that currently has enough data and refresh the on-screen report.

        Called after every capture/removal (silently -- ``warn_if_empty=False``, since "not
        enough yet" is the normal state early on, not an error) and by the Solve button itself
        (``warn_if_empty=True``, the one case a popup is warranted: the user asked explicitly and
        got nothing). Idempotent and cheap (closed-form least-squares over however many captures
        exist), so calling it this often is not a performance concern.
        """
        # Each arm's wrist camera is an independent, uncalibrated FK frame (see charuco.py's
        # module docstring) -- solved and reported separately, never pooled.
        counts = {arm: sum(1 for c in self.captures if c.arm == arm) for arm in self.arms}
        if all(n < 2 for n in counts.values()):
            if warn_if_empty:
                QtWidgets.QMessageBox.warning(
                    self, "Not enough captures", f"Need at least 2 captures for some arm; have {counts}."
                )
            self.result_label.setText("not solved yet -- capture at least 2 per arm")
            self.save_btn.setEnabled(False)
            return

        lines = []

        # Step 1: each arm's own wrist-camera extrinsic (gripper_T_camera), by hand-eye. Needs 3+
        # captures with varied ORIENTATION; until then the agentview/offset solves below fall
        # back to the CAD constant so they still produce a (less trustworthy) answer early on.
        wrist = solve_wrist_extrinsic_per_arm(self.captures)
        extrinsics = {}
        for arm in self.arms:
            if arm in wrist:
                w = wrist[arm]
                extrinsics[arm] = w.gripper_t_camera
                self._record(self._wrist_history[arm], w.n_captures, w.translation_rms_mm, w.rotation_rms_deg)
                quality = "good" if w.rotation_rms_deg < 1.0 and w.translation_rms_mm < 10.0 else "CHECK POSES"
                lines.append(
                    f"[wrist {arm}] hand-eye n={w.n_captures}  board-consistency RMS "
                    f"{w.translation_rms_mm:.2f} mm / {w.rotation_rms_deg:.3f} deg  -- {quality}  "
                    f"{_convergence_note(self._wrist_history[arm])}"
                )
            elif counts.get(arm):
                extrinsics[arm] = T_GRIPPER_CAMERA  # CAD fallback until 3+ varied captures exist
                lines.append(f"[wrist {arm}] {counts[arm]} capture(s) -- need 3+ (varied tilt) to hand-eye; using CAD")

        # Step 2: agentview extrinsic per arm, chained through each arm's wrist extrinsic (solved
        # above, or CAD fallback).
        results = solve_agentview_extrinsic_per_arm(self.captures, wrist_extrinsics=extrinsics)
        for arm, result in results.items():
            self._record(self._history[arm], result.n_captures, result.translation_rms_mm, result.rotation_rms_deg)
            quality = (
                "good" if result.rotation_rms_deg < 1.0 and result.translation_rms_mm < 10.0 else "CHECK CAPTURES"
            )
            via = "hand-eye" if arm in wrist else "CAD extrinsic"
            lines.append(
                f"[agentview {arm}] n={result.n_captures}  translation RMS {result.translation_rms_mm:.2f} mm  "
                f"rotation RMS {result.rotation_rms_deg:.3f} deg  ({via})  -- {quality}  "
                f"{_convergence_note(self._history[arm])}"
            )
        for arm, n in counts.items():
            if arm not in results and n:
                lines.append(f"[agentview {arm}] {n} capture(s) -- need at least 2 to solve")

        # Step 3: the arm-to-arm offset, and (once both single-arm extrinsics AND the offset
        # exist) the fused/cross-checked shared-frame answer -- see charuco.py's module docstring
        # for why this needs simultaneous both-wrists-see-the-board captures. The offset bridges
        # through each arm's wrist extrinsic too, so it also improves once hand-eye replaces CAD.
        arm_offset = None
        if len(self.pair_captures) >= 2 and "left" in extrinsics and "right" in extrinsics:
            arm_offset = solve_arm_offset(
                self.pair_captures, left_extrinsic=extrinsics["left"], right_extrinsic=extrinsics["right"]
            )
            self._record(
                self._offset_history, arm_offset.n_captures, arm_offset.translation_rms_mm, arm_offset.rotation_rms_deg
            )
            lines.append(
                f"[left<->right] n={arm_offset.n_captures}  distance {arm_offset.distance_m * 100:.1f} cm  "
                f"translation RMS {arm_offset.translation_rms_mm:.2f} mm  "
                f"rotation RMS {arm_offset.rotation_rms_deg:.3f} deg  {_convergence_note(self._offset_history)}"
            )
        elif self.pair_captures:
            lines.append(f"[left<->right] {len(self.pair_captures)} paired capture(s) -- need at least 2")

        unified = None
        if arm_offset is not None and "left" in results and "right" in results:
            unified = unify_rig_calibration(results, arm_offset)
            lines.append(
                f"[unified, left frame] cross-check {unified.cross_check_translation_mm:.2f} mm / "
                f"{unified.cross_check_rotation_deg:.3f} deg"
            )

        self.result_label.setText("\n".join(lines) if lines else "not solved yet")

        self._last_wrist = wrist
        self._last_results = results
        self._last_arm_offset = arm_offset
        self._last_unified = unified
        self.save_btn.setEnabled(bool(results) or bool(wrist))

    def _on_save(self) -> None:
        """Write the last solve into config.yaml -- see the module docstring for why this is a
        line-range splice (charuco.splice_*) rather than a YAML load/dump round trip, and mirrors
        tuner_gui.py's ``_write_config`` (confirm dialog naming the file, a ``.bak`` kept before
        writing, only the touched blocks change)."""
        if not self._last_results and self._last_arm_offset is None and not self._last_wrist:
            return
        if not self.config_path:
            QtWidgets.QMessageBox.critical(self, "No config.yaml", "No config.yaml was found to write into.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Write config.yaml?",
            f"Write the solved extrinsic(s) into\n{self.config_path}\n\n"
            "Only cameras.wrist_*/agentview.extrinsic and robot.arm_offset change; the rest of the "
            "file (comments included) is left alone. A .bak copy of the current file is kept first.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        calibrated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        written = []
        try:
            with open(self.config_path, encoding="utf-8") as fh:
                original = fh.read()
            updated = original
            for arm, w in self._last_wrist.items():
                updated = splice_wrist_extrinsic(updated, arm, w, calibrated_at=calibrated_at)
                written.append(f"wrist_{arm}.extrinsic")
            for arm, result in self._last_results.items():
                updated = splice_agentview_extrinsic(updated, arm, result, calibrated_at=calibrated_at)
                written.append(f"agentview.extrinsic.{arm}")
            if self._last_arm_offset is not None:
                updated = splice_arm_offset(updated, self._last_arm_offset, calibrated_at=calibrated_at)
                written.append("arm_offset")
            if self._last_unified is not None:
                # Derived from the two writes above, but always from THIS SAME live solve, in
                # this same save -- see splice_agentview_unified's docstring for why that keeps
                # it safe to store despite being recomputable.
                updated = splice_agentview_unified(updated, self._last_unified, calibrated_at=calibrated_at)
                written.append("extrinsic.unified")
            # Record which physical board produced all of the above -- provenance, and so the next
            # calibration auto-loads it (BoardSpec.from_config) without re-passing the flags.
            updated = splice_board(updated, self.board, calibrated_at=calibrated_at)
            written.append("calibration.board")
            # keep a .bak so a bad write is always recoverable -- same safety net tuner_gui's
            # own config.yaml writer uses.
            with open(self.config_path + ".bak", "w", encoding="utf-8") as fh:
                fh.write(original)
            with open(self.config_path, "w", encoding="utf-8") as fh:
                fh.write(updated)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", str(e))
            self.status.setText(f"write failed: {e}")
            return
        self.status.setText(f"wrote {', '.join(written)} -> {self.config_path} (backup at .bak)")

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
    capture_buttons: Sequence[str] = ("left.1", "right.1"),
) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cams = CameraManager(cfg)
    cams.start()
    win = CalibrateAgentviewWindow(
        cams, robot, geometries, board=board, config_path=config_path, mock=mock, capture_buttons=capture_buttons
    )
    win.resize(1600, 800)
    win.show()
    return app.exec_()
