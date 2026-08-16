"""ChArUco board detection + pose math for calibrating a fixed (non-wrist) camera.

Built for exactly one job: recover ``base_T_agentview`` -- the agentview camera's pose in the
robot's base frame -- from a ChArUco board sitting still on the desk, with NO physical
attachment to the robot and NO separate intrinsic-calibration step.

The trick is a chain, not a direct measurement. Nothing ties the board to the base frame by
itself, but the WRIST camera does: its pose in the base frame is already known at every instant
via forward kinematics (``WristCameraGeometry``, published extrinsic + the arm's own MJCF -- see
that module). So at any arm pose that has the board in the wrist camera's view:

    base_T_board = base_T_wrist(q) @ wrist_T_board      (wrist_T_board from a PnP solve)

and the same board, seen by agentview from wherever IT sits:

    agentview_T_board                                    (another PnP solve, same board)

give the one thing being solved for:

    base_T_agentview = base_T_board @ inv(agentview_T_board)

Move the arm to a few different poses (the board stays put) and average -- see
``solve_agentview_extrinsic`` -- both to smooth out per-shot PnP noise and to catch a bad
capture (a wildly different pose than the others) before it goes in the answer.

Intrinsics come from the RealSense devices themselves (``CameraManager.intrinsics``, factory
calibration off the device) rather than a checkerboard sweep -- accurate enough for this, and it
removes a whole separate calibration stage.

Needs ``opencv-contrib-python`` (``cv2.aruco``, specifically the ``CharucoDetector``/
``CharucoBoard.matchImagePoints`` API introduced in OpenCV 4.7 -- this does not fall back to the
pre-4.7 ``aruco.detectMarkers`` + ``interpolateCornersCharuco`` calls, which were removed).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import cv2


@dataclasses.dataclass(frozen=True)
class BoardSpec:
    """A physical ChArUco board's geometry -- must match the printed board exactly.

    ``square_length_m``/``marker_length_m`` are edge-to-edge in metres, not the printed page
    size: a board scaled when printed (fit-to-page, wrong DPI) silently biases every distance
    this module ever produces, with no error raised anywhere.
    """

    squares_x: int = 8
    squares_y: int = 6
    square_length_m: float = 0.030
    marker_length_m: float = 0.022
    dictionary: str = "DICT_4X4_50"


def make_board(spec: BoardSpec) -> "cv2.aruco.CharucoBoard":
    import cv2

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), spec.square_length_m, spec.marker_length_m, aruco_dict
    )


def intrinsics_matrix(intr: dict) -> np.ndarray:
    """``{fx, fy, cx, cy, ...}`` (``CameraManager.intrinsics()``'s shape) -> the 3x3 K matrix."""
    return np.array([[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclasses.dataclass
class Detection:
    """One image's board pose, and enough to judge whether to trust it."""

    cam_t_board: np.ndarray  # 4x4
    n_corners: int
    reproj_error_px: float
    corners_px: np.ndarray  # [n, 2], for drawing


def detect_board_pose(
    image: np.ndarray,
    spec: BoardSpec,
    intr: dict,
    *,
    dist_coeffs: Optional[Sequence[float]] = None,
    min_corners: int = 6,
) -> Optional[Detection]:
    """One image -> the board's pose in that camera's frame, or None if not (confidently) seen.

    ``min_corners`` guards against a PnP solve off a handful of corners at the board's edge --
    technically solvable, practically noisy. 6 of a typical 8x6 board's 35 interior corners is
    already a partial, oblique view; below that the solve is more extrapolation than measurement.
    """
    import cv2

    board = make_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(image)
    if charuco_corners is None or len(charuco_corners) < min_corners:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < min_corners:
        return None

    K = intrinsics_matrix(intr)
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    reproj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    err = float(np.linalg.norm(reproj.reshape(-1, 2) - img_points.reshape(-1, 2), axis=1).mean())

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return Detection(
        cam_t_board=T, n_corners=len(charuco_corners), reproj_error_px=err, corners_px=charuco_corners.reshape(-1, 2)
    )


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """The nearest proper rotation to an (averaged, no longer exactly orthogonal) 3x3 matrix.

    Element-wise-averaging several valid rotation matrices does not itself produce a valid one;
    this is the standard SVD projection back onto SO(3) (Kabsch/Markley), with the reflection
    case (det < 0, an SVD degeneracy rather than a real solution) corrected by flipping the last
    singular vector.
    """
    U, _S, Vt = np.linalg.svd(R)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of the rotation R represents, via the trace identity -- clipped because floating
    point can push ``(trace-1)/2`` a hair outside [-1, 1] and arccos of that is NaN."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


@dataclasses.dataclass
class Capture:
    """One calibration sample: the arm pose the wrist camera saw the board from, plus both
    cameras' independent board detections at (as close to) that same instant."""

    base_t_wrist: np.ndarray  # 4x4, from WristCameraGeometry.camera_pose(q) at capture time
    wrist_t_board: np.ndarray  # 4x4
    agentview_t_board: np.ndarray  # 4x4
    wrist_reproj_error_px: float
    agentview_reproj_error_px: float


@dataclasses.dataclass
class CalibrationResult:
    base_t_agentview: np.ndarray  # 4x4, the answer
    n_captures: int
    translation_rms_mm: float  # spread across captures -- a per-capture consistency check
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]  # distance from each estimate to the mean
    per_capture_rotation_deg: List[float]


def solve_agentview_extrinsic(captures: Sequence[Capture]) -> CalibrationResult:
    """Chain each capture into an independent ``base_T_agentview`` estimate, then average.

    Averaging AFTER the chain (not averaging ``base_T_board`` first) means a single bad capture
    -- an arm pose where the wrist camera grazed the board at a steep, noisy angle -- shows up as
    one outlier estimate rather than silently degrading a shared intermediate everything else
    depends on.
    """
    if len(captures) < 2:
        raise ValueError("need at least 2 captures (1 cannot show whether the estimates agree)")

    estimates = []
    for c in captures:
        base_t_board = c.base_t_wrist @ c.wrist_t_board
        estimates.append(base_t_board @ np.linalg.inv(c.agentview_t_board))

    t_mean = np.mean([e[:3, 3] for e in estimates], axis=0)
    R_mean = _orthonormalize(np.mean([e[:3, :3] for e in estimates], axis=0))

    per_t = [float(np.linalg.norm(e[:3, 3] - t_mean) * 1000.0) for e in estimates]
    per_r = [_rotation_angle_deg(R_mean.T @ e[:3, :3]) for e in estimates]

    out = np.eye(4)
    out[:3, :3] = R_mean
    out[:3, 3] = t_mean
    return CalibrationResult(
        base_t_agentview=out,
        n_captures=len(captures),
        # RMS distance-from-mean across captures, not std-of-those-distances: the summary this
        # is for is "how far do the per-capture estimates typically sit from the answer", which
        # np.std would instead report as "how much do the distances vary from each other" --
        # a real but much less useful number (e.g. every capture 5mm off in a different
        # direction reads as 0 std despite 5mm of real disagreement).
        translation_rms_mm=float(np.sqrt(np.mean(np.square(per_t)))),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(per_r)))),
        per_capture_translation_mm=per_t,
        per_capture_rotation_deg=per_r,
    )
