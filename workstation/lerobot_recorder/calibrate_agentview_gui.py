"""Calibrate agentview's extrinsic against a ChArUco board sitting on the desk.

See :mod:`workstation.lerobot_recorder.charuco` for the geometry this drives: the board never
moves, and a wrist camera's pose is already known at every instant (published extrinsic + that
arm's own FK, the same ``WristCameraGeometry`` the candidate-fan renderer uses), so it stands in
for a ruler between the board and that arm's own frame.

YAM is bimanual, so BOTH wrist cameras can bridge -- whichever one currently has the board in
view. Live preview of agentview + both wrists with the board outline drawn when found; "Capture"
grabs one sample per wrist camera that currently sees the board (0, 1, or 2 -- both arms do not
have to be posed at once, and when they are, one click banks two independent estimates instead of
one). Move the arm(s) to a few different poses that keep the board in view and capture at each.

"Solve" produces ONE extrinsic PER ARM, not one pooled answer: each arm's ``WristCameraGeometry``
is its own MJCF loaded in isolation, with no known transform to the other arm's -- there is no
shared "robot base" frame anywhere in this codebase (see ``charuco``'s module docstring). Mixing
left- and right-wrist captures into one solve would silently average two different questions'
answers together, so ``solve_agentview_extrinsic_per_arm`` groups by arm first. Each arm's result
reports how much ITS OWN captures disagree with each other, which is that arm's calibration
confidence rather than a separate validation step. "Save" writes both results (whichever arms had
enough captures) + the intrinsics they were solved against to one JSON file another tool can load
-- keyed by arm, since a consumer has to know which arm's frame it is asking for.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.charuco import (
    BoardSpec,
    Capture,
    detect_board_pose,
    solve_agentview_extrinsic_per_arm,
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


class CalibrateAgentviewWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        cams: CameraManager,
        robot: "Optional[RobotClient]",  # None in --mock
        geometries: Dict[str, object],  # {"left": WristCameraGeometry, "right": WristCameraGeometry}
        *,
        board: BoardSpec,
        out_path: str,
        mock: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Calibrate agentview -- board via wrist_left / wrist_right")
        self.cams = cams
        self.robot = robot
        self.geometries = geometries
        self.arms = list(geometries.keys())  # e.g. ["left", "right"]
        self.board = board
        self.out_path = out_path
        self.mock = mock

        self._last_agent_det = None
        self._last_wrist_det: Dict[str, object] = {}  # arm -> Detection | None
        self._last_q: Dict[str, Optional[np.ndarray]] = {}  # arm -> joints | None
        self.captures: List[Capture] = []

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
        self.capture_btn = QtWidgets.QPushButton("Capture")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._on_capture)
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

        self.result_label = QtWidgets.QLabel("not solved yet")
        self.result_label.setStyleSheet("font-size:20px;")
        root.addWidget(self.result_label)

        self._result_json: Optional[dict] = None

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        try:
            frames = self.cams.read()
        except Exception as e:
            self.status.setText(f"camera read failed: {e}")
            return

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
            self._last_q[arm] = self._read_joints(arm)
            if img is not None:
                preview = _draw_detection(img, det.corners_px if det else None)
                self.wrist_views[arm].setPixmap(
                    _np_to_pixmap(preview).scaled(self.wrist_views[arm].size(), QtCore.Qt.KeepAspectRatio)
                )
            if self._last_agent_det is not None and det is not None and self._last_q[arm] is not None:
                ready_arms.append(arm)

        self.capture_btn.setEnabled(bool(ready_arms))
        if self._last_agent_det is None:
            self.status.setText("not ready -- board not seen in agentview")
        elif ready_arms:
            self.status.setText(
                f"ready to capture: {', '.join(ready_arms)} (agentview + that wrist both see the board)"
            )
        else:
            self.status.setText("board seen in agentview but neither wrist sees it (or robot link is down)")

    def _read_joints(self, arm: str) -> Optional[np.ndarray]:
        if self.mock or self.robot is None:
            return np.zeros(7, np.float64)
        try:
            obs = self.robot.get_observation()
            side = obs.get(arm)
            if not side or side.get("pos") is None:
                return None
            return np.asarray(side["pos"], dtype=np.float64)
        except Exception as e:
            logger.warning("could not read robot joints: %s", e)
            return None

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
            base_t_wrist = self.geometries[arm].camera_pose(q)
            cap = Capture(
                arm=arm,
                base_t_wrist=base_t_wrist,
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
        if not added:
            self.status.setText("capture pressed but nothing was ready -- see status above")

    def _on_remove(self) -> None:
        row = self.capture_list.currentRow()
        if row < 0:
            return
        del self.captures[row]
        self.capture_list.takeItem(row)

    def _on_solve(self) -> None:
        # Each arm's wrist camera is an independent, uncalibrated FK frame (see charuco.py's
        # module docstring) -- solved and reported separately, never pooled.
        counts = {arm: sum(1 for c in self.captures if c.arm == arm) for arm in self.arms}
        if all(n < 2 for n in counts.values()):
            QtWidgets.QMessageBox.warning(
                self, "Not enough captures", f"Need at least 2 captures for some arm; have {counts}."
            )
            return
        results = solve_agentview_extrinsic_per_arm(self.captures)

        lines = []
        per_arm_json = {}
        for arm, result in results.items():
            quality = (
                "good" if result.rotation_rms_deg < 1.0 and result.translation_rms_mm < 10.0 else "CHECK CAPTURES"
            )
            lines.append(
                f"[{arm}] n={result.n_captures}  translation RMS {result.translation_rms_mm:.2f} mm  "
                f"rotation RMS {result.rotation_rms_deg:.3f} deg  -- {quality}"
            )
            per_arm_json[arm] = {
                # This extrinsic is expressed in `arm`'s own FK frame -- NOT interchangeable
                # with the other arm's entry, and not a robot-wide "base" frame. See
                # charuco.py's module docstring.
                "base_t_agentview": result.base_t_agentview.tolist(),
                "n_captures": result.n_captures,
                "translation_rms_mm": result.translation_rms_mm,
                "rotation_rms_deg": result.rotation_rms_deg,
                "per_capture_translation_mm": result.per_capture_translation_mm,
                "per_capture_rotation_deg": result.per_capture_rotation_deg,
            }
        for arm, n in counts.items():
            if arm not in results and n:
                lines.append(f"[{arm}] {n} capture(s) -- need at least 2 to solve")
        self.result_label.setText("   ".join(lines) if lines else "not solved yet")

        if not per_arm_json:
            self.save_btn.setEnabled(False)
            return
        self._result_json = {
            "by_arm": per_arm_json,
            "board": {
                "squares_x": self.board.squares_x,
                "squares_y": self.board.squares_y,
                "square_length_m": self.board.square_length_m,
                "marker_length_m": self.board.marker_length_m,
                "dictionary": self.board.dictionary,
            },
            "agentview_intrinsics": self.cams.intrinsics("agentview"),
            "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_btn.setEnabled(True)

    def _on_save(self) -> None:
        if self._result_json is None:
            return
        import pathlib

        path = pathlib.Path(self.out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._result_json, indent=2))
        self.status.setText(f"saved -> {path}")

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
    out_path: str,
    mock: bool = False,
) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cams = CameraManager(cfg)
    cams.start()
    win = CalibrateAgentviewWindow(cams, robot, geometries, board=board, out_path=out_path, mock=mock)
    win.resize(1600, 800)
    win.show()
    return app.exec_()
