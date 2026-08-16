"""ChArUco board detection + pose math for calibrating a fixed (non-wrist) camera.

Built for exactly one job: recover an agentview extrinsic from a ChArUco board sitting still on
the desk, with NO physical attachment to the robot and NO separate intrinsic-calibration step.

The trick is a chain, not a direct measurement. Nothing ties the board to a robot-relative frame
by itself, but a WRIST camera does: its pose is already known at every instant via forward
kinematics (``WristCameraGeometry``, published extrinsic + the arm's own MJCF -- see that
module). So at any arm pose that has the board in the wrist camera's view:

    frame_T_board = frame_T_wrist(q) @ wrist_T_board      (wrist_T_board from a PnP solve)

and the same board, seen by agentview from wherever IT sits:

    agentview_T_board                                     (another PnP solve, same board)

give the one thing being solved for:

    frame_T_agentview = frame_T_board @ inv(agentview_T_board)

Move the arm to a few different poses (the board stays put) and average -- see
``solve_agentview_extrinsic`` -- both to smooth out per-shot PnP noise and to catch a bad
capture (a wildly different pose than the others) before it goes in the answer.

**"frame_T_wrist" is per-arm, not a shared robot frame -- there is no cross-arm transform
anywhere in this codebase.** ``WristCameraGeometry`` builds FK from ONE arm's own MJCF
(``combine_arm_and_gripper_xml`` has no left/right parameter and no rig-level origin), so its
"base frame" is that model's own (0,0,0) -- unrelated to wherever the arm is actually bolted to
the table, and unrelated to the OTHER arm's own (0,0,0). Two ``WristCameraGeometry`` instances
for left and right are two different, uncalibrated coordinate systems that happen to share a
class. Chaining a left-wrist capture and a right-wrist capture into one solve would therefore
silently average together the answers to two different questions. ``Capture.arm`` records which
one each capture came from, ``solve_agentview_extrinsic`` refuses to mix them, and
``solve_agentview_extrinsic_per_arm`` is the entry point that actually accounts for this --
solving (and reporting) one extrinsic per arm, each valid only within that arm's own frame.

**That missing cross-arm transform is itself solvable the same way, and the SAME board gets it
for free.** Whenever both wrist cameras happen to see the still-unmoved board AT ONCE, each
independently bridges to it:

    left_T_board  = left_T_wrist(q_left)   @ left_wrist_T_board
    right_T_board = right_T_wrist(q_right) @ right_wrist_T_board

and since it is the same board, unmoved, between the two:

    left_T_right = left_T_board @ inv(right_T_board)

See ``ArmPairCapture``/``solve_arm_offset``. Its translation magnitude is the physical distance
between the two arms' own FK origins -- not measured with a ruler, recovered from the same board
captures. Once known, it also lets ``unify_rig_calibration`` cross-check the two independent
agentview extrinsics against each other (bridge the right-arm one through ``left_T_right`` and
compare it to the directly-solved left-arm one) -- an end-to-end confidence number on all three
calibrations at once, not a separate validation step.

Intrinsics come from the RealSense devices themselves (``CameraManager.intrinsics``, factory
calibration off the device) rather than a checkerboard sweep -- accurate enough for this, and it
removes a whole separate calibration stage.

Needs ``opencv-contrib-python`` (``cv2.aruco``, specifically the ``CharucoDetector``/
``CharucoBoard.matchImagePoints`` API introduced in OpenCV 4.7 -- this does not fall back to the
pre-4.7 ``aruco.detectMarkers`` + ``interpolateCornersCharuco`` calls, which were removed).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

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


def _average_poses(estimates: Sequence[np.ndarray]) -> tuple:
    """Average several independent 4x4 pose estimates (translation: mean; rotation: SVD-projected
    mean, see ``_orthonormalize``), and each one's distance from that average.

    The shared machinery behind every "chain several captures into one relative pose, then
    average" solve in this module (``solve_agentview_extrinsic``, ``solve_arm_offset``) --
    averaging AFTER each capture is independently chained, not averaging an intermediate first,
    is what keeps one bad capture an outlier in the result instead of a silent corruption of a
    shared intermediate everything else depends on.

    Returns ``(mean_pose, per_capture_translation_mm, per_capture_rotation_deg)``.
    """
    t_mean = np.mean([e[:3, 3] for e in estimates], axis=0)
    R_mean = _orthonormalize(np.mean([e[:3, :3] for e in estimates], axis=0))
    per_t = [float(np.linalg.norm(e[:3, 3] - t_mean) * 1000.0) for e in estimates]
    per_r = [_rotation_angle_deg(R_mean.T @ e[:3, :3]) for e in estimates]
    out = np.eye(4)
    out[:3, :3] = R_mean
    out[:3, 3] = t_mean
    return out, per_t, per_r


def _rms(values: Sequence[float]) -> float:
    """RMS, not std: the summary wanted here is "how far do these typically sit from the mean",
    which np.std would instead report as "how much do THEY vary from EACH OTHER" -- a real but
    much less useful number (e.g. every value 5mm off in a different direction reads as 0 std
    despite 5mm of real disagreement). See solve_agentview_extrinsic's original note on this."""
    return float(np.sqrt(np.mean(np.square(values))))


@dataclasses.dataclass
class Capture:
    """One calibration sample: the arm pose a wrist camera saw the board from, plus both
    cameras' independent board detections at (as close to) that same instant.

    ``arm`` records which wrist camera / FK model ``base_t_wrist`` came from -- required so
    ``solve_agentview_extrinsic`` can refuse to mix frames it has no business averaging (see the
    module docstring)."""

    arm: str  # "left" | "right" -- whichever WristCameraGeometry built base_t_wrist
    base_t_wrist: np.ndarray  # 4x4, from WristCameraGeometry.camera_pose(q) at capture time
    wrist_t_board: np.ndarray  # 4x4
    agentview_t_board: np.ndarray  # 4x4
    wrist_reproj_error_px: float
    agentview_reproj_error_px: float


@dataclasses.dataclass
class CalibrationResult:
    arm: str  # which arm's frame this extrinsic is expressed in -- see the module docstring
    base_t_agentview: np.ndarray  # 4x4, the answer, valid only within `arm`'s own FK frame
    n_captures: int
    translation_rms_mm: float  # spread across captures -- a per-capture consistency check
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]  # distance from each estimate to the mean
    per_capture_rotation_deg: List[float]


def solve_agentview_extrinsic(captures: Sequence[Capture]) -> CalibrationResult:
    """Chain each capture into an independent extrinsic estimate, then average.

    Averaging AFTER the chain (not averaging ``frame_T_board`` first) means a single bad capture
    -- an arm pose where the wrist camera grazed the board at a steep, noisy angle -- shows up as
    one outlier estimate rather than silently degrading a shared intermediate everything else
    depends on.

    Every capture must be the SAME arm: see the module docstring for why mixing left- and
    right-wrist captures here would silently average two unrelated coordinate frames together.
    Use ``solve_agentview_extrinsic_per_arm`` for a mixed batch -- it calls this once per arm.
    """
    if len(captures) < 2:
        raise ValueError("need at least 2 captures (1 cannot show whether the estimates agree)")
    arms = {c.arm for c in captures}
    if len(arms) > 1:
        raise ValueError(
            f"captures span more than one arm ({sorted(arms)}) -- each arm's wrist camera is an "
            "independent, uncalibrated frame (see the module docstring), so they cannot be "
            "solved together. Use solve_agentview_extrinsic_per_arm for a mixed batch."
        )
    (arm,) = arms

    estimates = []
    for c in captures:
        base_t_board = c.base_t_wrist @ c.wrist_t_board
        estimates.append(base_t_board @ np.linalg.inv(c.agentview_t_board))

    out, per_t, per_r = _average_poses(estimates)
    return CalibrationResult(
        arm=arm,
        base_t_agentview=out,
        n_captures=len(captures),
        translation_rms_mm=_rms(per_t),
        rotation_rms_deg=_rms(per_r),
        per_capture_translation_mm=per_t,
        per_capture_rotation_deg=per_r,
    )


def solve_agentview_extrinsic_per_arm(captures: Sequence[Capture]) -> Dict[str, CalibrationResult]:
    """Group captures by ``arm`` and solve each group independently.

    This -- not ``solve_agentview_extrinsic`` directly -- is the entry point for a session that
    captured from both wrist cameras: each arm gets its OWN extrinsic, valid only within that
    arm's own (uncalibrated-against-the-other) FK frame (see the module docstring for why they
    cannot be pooled). An arm with fewer than 2 captures is silently skipped rather than raising
    -- a caller collecting from both wrists but so far only succeeding on one should still get
    that one back, the same way a mixed collection session naturally produces uneven counts.
    """
    by_arm: Dict[str, List[Capture]] = {}
    for c in captures:
        by_arm.setdefault(c.arm, []).append(c)
    return {arm: solve_agentview_extrinsic(group) for arm, group in by_arm.items() if len(group) >= 2}


@dataclasses.dataclass
class ArmPairCapture:
    """Both wrist cameras seeing the SAME (still, unmoved) board at once -- what
    ``solve_arm_offset`` needs.

    Distinct from ``Capture``: that one only ever needs ONE wrist camera + agentview.
    ``left_T_right`` needs BOTH wrist cameras in the same instant, since "the board did not move
    between these two detections" is the only thing tying the two independent FK chains
    together -- a left-wrist capture from one moment and a right-wrist capture from another,
    with the board nudged in between, would silently corrupt the offset with however far it
    moved.
    """

    left_t_wrist: np.ndarray  # 4x4, left WristCameraGeometry.camera_pose(q_left)
    left_wrist_t_board: np.ndarray  # 4x4
    right_t_wrist: np.ndarray  # 4x4, right WristCameraGeometry.camera_pose(q_right)
    right_wrist_t_board: np.ndarray  # 4x4
    left_reproj_error_px: float
    right_reproj_error_px: float


@dataclasses.dataclass
class ArmOffsetResult:
    left_t_right: np.ndarray  # 4x4, the answer
    distance_m: float  # ||translation|| -- how far apart the two arms' own FK origins sit
    n_captures: int
    translation_rms_mm: float
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]
    per_capture_rotation_deg: List[float]


def solve_arm_offset(captures: Sequence[ArmPairCapture]) -> ArmOffsetResult:
    """``left_T_right`` from moments both wrist cameras saw the same still board at once.

    Same "chain each capture independently, then average" pattern as
    ``solve_agentview_extrinsic`` -- here BOTH sides of the chain are FK-bridged rather than one
    being a fixed camera, but the maths (and the reason to average after chaining, not before)
    is identical; see that function's docstring.
    """
    if len(captures) < 2:
        raise ValueError("need at least 2 captures (1 cannot show whether the estimates agree)")

    estimates = []
    for c in captures:
        left_t_board = c.left_t_wrist @ c.left_wrist_t_board
        right_t_board = c.right_t_wrist @ c.right_wrist_t_board
        estimates.append(left_t_board @ np.linalg.inv(right_t_board))

    out, per_t, per_r = _average_poses(estimates)
    return ArmOffsetResult(
        left_t_right=out,
        distance_m=float(np.linalg.norm(out[:3, 3])),
        n_captures=len(captures),
        translation_rms_mm=_rms(per_t),
        rotation_rms_deg=_rms(per_r),
        per_capture_translation_mm=per_t,
        per_capture_rotation_deg=per_r,
    )


@dataclasses.dataclass
class UnifiedRigCalibration:
    """Everything expressed in ONE shared frame -- canonically the left arm's -- once
    ``left_T_right`` is known.

    ``left_t_agentview`` is the FUSED estimate: the mean of the direct left-wrist solve and the
    right-wrist solve bridged through ``left_t_right``, when both exist (otherwise whichever one
    does -- a rig calibrated from only one arm's captures still produces an answer, just without
    the cross-check). ``cross_check_*`` is how far apart those two independent routes to
    agentview landed, when both exist -- an end-to-end confidence number on all three
    calibrations (left-agentview, right-agentview, left-right) at once, not a separate
    validation step. None when only one arm's agentview extrinsic was available to fuse.
    """

    left_t_agentview: np.ndarray
    left_t_right: np.ndarray
    distance_m: float
    cross_check_translation_mm: Optional[float]
    cross_check_rotation_deg: Optional[float]


def unify_rig_calibration(
    agentview_by_arm: Dict[str, CalibrationResult], arm_offset: ArmOffsetResult
) -> UnifiedRigCalibration:
    """Fuse a per-arm agentview solve (``solve_agentview_extrinsic_per_arm``) with the arm
    offset (``solve_arm_offset``) into one shared-frame answer, cross-checking the two
    independent routes to agentview against each other when both are available."""
    left = agentview_by_arm.get("left")
    right = agentview_by_arm.get("right")
    if left is None and right is None:
        raise ValueError("agentview_by_arm has neither 'left' nor 'right' -- nothing to fuse")

    bridged_right = arm_offset.left_t_right @ right.base_t_agentview if right is not None else None
    candidates = [m for m in (left.base_t_agentview if left is not None else None, bridged_right) if m is not None]
    fused, _per_t, _per_r = _average_poses(candidates)

    cross_t = cross_r = None
    if left is not None and bridged_right is not None:
        cross_t = float(np.linalg.norm(bridged_right[:3, 3] - left.base_t_agentview[:3, 3]) * 1000.0)
        cross_r = _rotation_angle_deg(left.base_t_agentview[:3, :3].T @ bridged_right[:3, :3])

    return UnifiedRigCalibration(
        left_t_agentview=fused,
        left_t_right=arm_offset.left_t_right,
        distance_m=arm_offset.distance_m,
        cross_check_translation_mm=cross_t,
        cross_check_rotation_deg=cross_r,
    )
