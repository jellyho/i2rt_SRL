"""ChArUco board detection + pose math for calibrating the whole camera rig from one desk board.

One board sitting still on the desk (NO physical attachment to the robot, NO separate
intrinsic-calibration step) recovers three things, in order, each feeding the next:

1. **Each wrist camera's own extrinsic** (``gripper_T_camera``) -- eye-in-hand hand-eye
   calibration (``solve_wrist_extrinsic``). ``WristCameraGeometry`` ships a CAD value for this
   (``T_GRIPPER_CAMERA``) but it was never checked against the built hardware; a real mount error
   there leaks into everything downstream, so we solve it ourselves and USE the solved value
   below instead of trusting the CAD one.
2. **Each agentview extrinsic** (``frame_T_agentview``, per arm) -- ``solve_agentview_extrinsic``.
3. **The left<->right arm offset** (``left_T_right``) -- ``solve_arm_offset``.

The bridge for (2) and (3) is a chain, not a direct measurement. Nothing ties the board to a
robot-relative frame by itself, but a wrist camera does once (1) is known: the FLANGE pose is
known at every instant from forward kinematics (``WristCameraGeometry.flange_pose`` -- joint
encoders + the arm MJCF, independent of any camera mount), and ``flange @ gripper_T_camera``
places the camera. So at any arm pose that has the board in the wrist camera's view:

    frame_T_board = flange_pose(q) @ gripper_T_camera @ wrist_T_board   (wrist_T_board from PnP)

and the same board, seen by agentview from wherever IT sits:

    agentview_T_board                                     (another PnP solve, same board)

give the one thing being solved for:

    frame_T_agentview = frame_T_board @ inv(agentview_T_board)

Move the arm to a few different poses (the board stays put) and average -- see
``solve_agentview_extrinsic`` -- both to smooth out per-shot PnP noise and to catch a bad
capture (a wildly different pose than the others) before it goes in the answer. Hand-eye (1)
additionally needs those poses to differ in ORIENTATION, not just position, or the rotation part
of ``gripper_T_camera`` is unconstrained -- see ``solve_wrist_extrinsic``.

Captures store the extrinsic-free ``base_t_flange`` (pure FK), NOT ``flange @ extrinsic``, so the
same captures can be re-solved against the freshly hand-eye'd extrinsic rather than baking the
CAD one in at capture time.

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

    def to_config_dict(self) -> dict:
        """As a plain dict for writing under config.yaml's ``calibration.board`` (see
        ``splice_board``). Keys match ``from_config``'s so a round trip is lossless."""
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "dictionary": self.dictionary,
        }

    @classmethod
    def from_config(cls, board: Optional[dict]) -> "BoardSpec":
        """Build from a config.yaml ``calibration.board`` mapping, falling back to the class
        defaults for any key it omits (so a partial or absent block is fine). Unknown keys are
        ignored rather than raising -- a config carrying extra provenance (``calibrated_at``) that
        this writes back should still load."""
        board = board or {}
        d = cls().to_config_dict()
        return cls(**{k: board.get(k, v) for k, v in d.items()})


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
    """One calibration sample: the arm's FK pose plus both cameras' independent board detections
    at (as close to) that same instant.

    ``arm`` records which wrist camera / FK model ``base_t_flange`` came from -- required so
    ``solve_agentview_extrinsic`` can refuse to mix frames it has no business averaging (see the
    module docstring).

    ``base_t_flange`` is the EXTRINSIC-FREE flange pose (``WristCameraGeometry.flange_pose``),
    not ``flange @ extrinsic``: it is both what hand-eye needs as the gripper pose, and what lets
    the agentview solve be re-run against a freshly hand-eye'd extrinsic instead of the CAD one
    baked in at capture time."""

    arm: str  # "left" | "right" -- whichever WristCameraGeometry built base_t_flange
    base_t_flange: np.ndarray  # 4x4, WristCameraGeometry.flange_pose(q) -- FK only, no extrinsic
    wrist_t_board: np.ndarray  # 4x4, wrist camera PnP (camera_T_board)
    agentview_t_board: np.ndarray  # 4x4, agentview PnP
    wrist_reproj_error_px: float
    agentview_reproj_error_px: float


@dataclasses.dataclass
class WristExtrinsicResult:
    arm: str
    gripper_t_camera: np.ndarray  # 4x4, the answer -- drop-in for WristCameraGeometry's CAD extrinsic
    n_captures: int
    # Consistency residual: with the solved extrinsic, base_T_board should be the SAME (the board
    # did not move) across every capture. How far the per-capture base_T_board estimates sit from
    # their mean is the honest end-to-end error of the hand-eye solve + the FK behind it. NOT a
    # pure hand-eye reprojection -- it folds in FK error too, which is exactly what you want to
    # know before trusting this extrinsic downstream.
    translation_rms_mm: float
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]
    per_capture_rotation_deg: List[float]


def solve_wrist_extrinsic(captures: Sequence[Capture]) -> WristExtrinsicResult:
    """``gripper_T_camera`` for ONE arm, by eye-in-hand hand-eye calibration off the fixed board.

    Solves AX = XB (``cv2.calibrateHandEye``, Tsai-Lenz) from the arm's own FK flange poses
    (``base_t_flange``, the "gripper2base" input) and the wrist camera's board detections
    (``wrist_t_board``, the "target2cam" input). The board being bolted to nothing and never
    moving is what makes it a valid stationary target.

    Needs at least 3 captures, and -- unlike the agentview solve -- those poses must differ in
    ORIENTATION, not just position: a pure translation between two poses leaves the rotation part
    of X unconstrained (AX = XB degenerates), so a set of captures all at the same wrist tilt can
    return a confident-looking but wrong extrinsic. The reported RMS residual (board-in-base
    consistency, see ``WristExtrinsicResult``) is the check on whether the poses were varied
    enough; a tight residual with too-similar poses is the one failure it cannot catch, same
    caveat as the agentview convergence trend.
    """
    import cv2

    if len(captures) < 3:
        raise ValueError("hand-eye needs at least 3 captures (2 relative motions) to solve X")
    arms = {c.arm for c in captures}
    if len(arms) > 1:
        raise ValueError(f"captures span more than one arm ({sorted(arms)}) -- each arm's mount is its own solve")
    (arm,) = arms

    r_g2b = [c.base_t_flange[:3, :3] for c in captures]
    t_g2b = [c.base_t_flange[:3, 3] for c in captures]
    r_t2c = [c.wrist_t_board[:3, :3] for c in captures]
    t_t2c = [c.wrist_t_board[:3, 3] for c in captures]
    r_x, t_x = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    x = np.eye(4)
    x[:3, :3] = r_x
    x[:3, 3] = np.asarray(t_x).reshape(3)
    # On too-similar poses cv2 logs "Not enough informative motions" and can hand back a
    # non-finite X rather than raising. Turn that into the same "not solved" path a <3-capture
    # arm takes (per_arm skips it -> CAD fallback), so a NaN mount never reaches the config.
    if not np.all(np.isfinite(x)):
        raise ValueError("hand-eye did not converge (too-similar poses -- vary the wrist tilt more)")

    # base_T_board per capture, which should be identical if X (and the FK) are right.
    boards = [c.base_t_flange @ x @ c.wrist_t_board for c in captures]
    _mean, per_t, per_r = _average_poses(boards)
    return WristExtrinsicResult(
        arm=arm,
        gripper_t_camera=x,
        n_captures=len(captures),
        translation_rms_mm=_rms(per_t),
        rotation_rms_deg=_rms(per_r),
        per_capture_translation_mm=per_t,
        per_capture_rotation_deg=per_r,
    )


def solve_wrist_extrinsic_per_arm(captures: Sequence[Capture]) -> Dict[str, WristExtrinsicResult]:
    """Group captures by ``arm`` and hand-eye-solve each; an arm with fewer than 3 captures, or
    one whose solve does not converge (too-similar poses -- see ``solve_wrist_extrinsic``), is
    skipped (omitted from the returned dict), the same lenient behaviour as
    ``solve_agentview_extrinsic_per_arm``. The caller then uses the CAD fallback for that arm."""
    by_arm: Dict[str, List[Capture]] = {}
    for c in captures:
        by_arm.setdefault(c.arm, []).append(c)
    out: Dict[str, WristExtrinsicResult] = {}
    for arm, group in by_arm.items():
        if len(group) < 3:
            continue
        try:
            out[arm] = solve_wrist_extrinsic(group)
        except ValueError:
            pass  # not converged yet -> leave it out, caller falls back to CAD
    return out


@dataclasses.dataclass
class CalibrationResult:
    arm: str  # which arm's frame this extrinsic is expressed in -- see the module docstring
    base_t_agentview: np.ndarray  # 4x4, the answer, valid only within `arm`'s own FK frame
    n_captures: int
    translation_rms_mm: float  # spread across captures -- a per-capture consistency check
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]  # distance from each estimate to the mean
    per_capture_rotation_deg: List[float]


def solve_agentview_extrinsic(captures: Sequence[Capture], *, wrist_extrinsic: np.ndarray) -> CalibrationResult:
    """Chain each capture into an independent extrinsic estimate, then average.

    ``wrist_extrinsic`` (``gripper_T_camera``) places the camera on the flange: the per-capture
    bridge is ``base_t_flange @ wrist_extrinsic @ wrist_t_board``. Pass the hand-eye result
    (``solve_wrist_extrinsic``) to use the measured mount, or ``WristCameraGeometry``'s CAD
    ``T_GRIPPER_CAMERA`` to fall back to the unverified spec -- the caller decides which, and this
    is exactly the value hand-eye exists to replace.

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
        base_t_board = c.base_t_flange @ wrist_extrinsic @ c.wrist_t_board
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


def solve_agentview_extrinsic_per_arm(
    captures: Sequence[Capture], *, wrist_extrinsics: Dict[str, np.ndarray]
) -> Dict[str, CalibrationResult]:
    """Group captures by ``arm`` and solve each group independently.

    This -- not ``solve_agentview_extrinsic`` directly -- is the entry point for a session that
    captured from both wrist cameras: each arm gets its OWN extrinsic, valid only within that
    arm's own (uncalibrated-against-the-other) FK frame (see the module docstring for why they
    cannot be pooled). An arm with fewer than 2 captures is silently skipped rather than raising
    -- a caller collecting from both wrists but so far only succeeding on one should still get
    that one back, the same way a mixed collection session naturally produces uneven counts.

    ``wrist_extrinsics`` maps arm -> ``gripper_T_camera`` (hand-eye solve or CAD fallback); an arm
    absent from it is skipped too, since there is no camera placement to chain through.
    """
    by_arm: Dict[str, List[Capture]] = {}
    for c in captures:
        by_arm.setdefault(c.arm, []).append(c)
    return {
        arm: solve_agentview_extrinsic(group, wrist_extrinsic=wrist_extrinsics[arm])
        for arm, group in by_arm.items()
        if len(group) >= 2 and arm in wrist_extrinsics
    }


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

    left_t_flange: np.ndarray  # 4x4, left WristCameraGeometry.flange_pose(q_left) -- FK only
    left_wrist_t_board: np.ndarray  # 4x4, left wrist camera PnP
    right_t_flange: np.ndarray  # 4x4, right WristCameraGeometry.flange_pose(q_right) -- FK only
    right_wrist_t_board: np.ndarray  # 4x4, right wrist camera PnP
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


def solve_arm_offset(
    captures: Sequence[ArmPairCapture], *, left_extrinsic: np.ndarray, right_extrinsic: np.ndarray
) -> ArmOffsetResult:
    """``left_T_right`` from moments both wrist cameras saw the same still board at once.

    Each side places its camera with its OWN ``gripper_T_camera`` (``left_extrinsic`` /
    ``right_extrinsic`` -- hand-eye results or CAD fallback), so a mount error on either arm feeds
    straight into the offset: this is why solving the wrist extrinsics first, and passing the
    measured ones here, matters rather than trusting CAD on both.

    Same "chain each capture independently, then average" pattern as
    ``solve_agentview_extrinsic`` -- here BOTH sides of the chain are FK-bridged rather than one
    being a fixed camera, but the maths (and the reason to average after chaining, not before)
    is identical; see that function's docstring.
    """
    if len(captures) < 2:
        raise ValueError("need at least 2 captures (1 cannot show whether the estimates agree)")

    estimates = []
    for c in captures:
        left_t_board = c.left_t_flange @ left_extrinsic @ c.left_wrist_t_board
        right_t_board = c.right_t_flange @ right_extrinsic @ c.right_wrist_t_board
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


# --------------------------------------------------------------------------------------- #
# Writing results into config.yaml -- THE single source of truth for the rig (see its own
# header comment). Same technique as workstation/lerobot_recorder/exposure_tuner.py's
# splice_camera_options: a line-range splice, not a yaml.safe_load/dump round trip, because
# config.yaml is heavily commented and a round trip would strip every comment and reflow the
# whole document. Only the target block is touched; everything else survives byte for byte.
# --------------------------------------------------------------------------------------- #
def _format_matrix_yaml(matrix: np.ndarray, indent: int) -> List[str]:
    pad = " " * indent
    return [f"{pad}- [{', '.join(f'{v:.6f}' for v in row)}]" for row in matrix]


def format_extrinsic_yaml(result: CalibrationResult, *, calibrated_at: str, indent: int = 8) -> List[str]:
    """The body of one arm's ``extrinsic`` entry -- ready to splice under
    ``cameras.agentview.extrinsic.<arm>`` (see ``splice_agentview_extrinsic``)."""
    pad = " " * indent
    return [
        f"{pad}matrix:",
        *_format_matrix_yaml(result.base_t_agentview, indent + 2),
        f"{pad}n_captures: {result.n_captures}",
        f"{pad}translation_rms_mm: {result.translation_rms_mm:.3f}",
        f"{pad}rotation_rms_deg: {result.rotation_rms_deg:.4f}",
        f'{pad}calibrated_at: "{calibrated_at}"',
    ]


def format_arm_offset_yaml(result: ArmOffsetResult, *, calibrated_at: str, indent: int = 4) -> List[str]:
    """The body of the ``robot.arm_offset`` entry (see ``splice_arm_offset``)."""
    pad = " " * indent
    return [
        f"{pad}matrix:",
        *_format_matrix_yaml(result.left_t_right, indent + 2),
        f"{pad}distance_m: {result.distance_m:.4f}",
        f"{pad}n_captures: {result.n_captures}",
        f"{pad}translation_rms_mm: {result.translation_rms_mm:.3f}",
        f"{pad}rotation_rms_deg: {result.rotation_rms_deg:.4f}",
        f'{pad}calibrated_at: "{calibrated_at}"',
    ]


def format_wrist_extrinsic_yaml(result: WristExtrinsicResult, *, calibrated_at: str, indent: int = 6) -> List[str]:
    """The body of one wrist camera's ``extrinsic`` entry -- ready under
    ``cameras.wrist_<arm>.extrinsic`` (see ``splice_wrist_extrinsic``). ``matrix`` is
    ``gripper_T_camera``, the hand-eye replacement for ``WristCameraGeometry``'s CAD constant."""
    pad = " " * indent
    return [
        f"{pad}matrix:",
        *_format_matrix_yaml(result.gripper_t_camera, indent + 2),
        f"{pad}n_captures: {result.n_captures}",
        f"{pad}translation_rms_mm: {result.translation_rms_mm:.3f}",
        f"{pad}rotation_rms_deg: {result.rotation_rms_deg:.4f}",
        f'{pad}calibrated_at: "{calibrated_at}"',
    ]


def format_unified_yaml(result: UnifiedRigCalibration, *, calibrated_at: str, indent: int = 8) -> List[str]:
    """The body of ``cameras.agentview.extrinsic.unified`` (see ``splice_agentview_unified``).

    Only the fused matrix + the cross-check diagnostics -- ``left_t_right``/``distance_m`` are
    NOT repeated here even though ``UnifiedRigCalibration`` carries them too, since
    ``robot.arm_offset`` already holds that exact number; copying it a second place would be
    the same fact twice for no reason, in a file where the whole point is one source of truth.
    """
    pad = " " * indent
    lines = [f"{pad}matrix:", *_format_matrix_yaml(result.left_t_agentview, indent + 2)]
    if result.cross_check_translation_mm is not None:
        lines.append(f"{pad}cross_check_translation_mm: {result.cross_check_translation_mm:.3f}")
    if result.cross_check_rotation_deg is not None:
        lines.append(f"{pad}cross_check_rotation_deg: {result.cross_check_rotation_deg:.4f}")
    lines.append(f'{pad}calibrated_at: "{calibrated_at}"')
    return lines


def format_board_yaml(spec: BoardSpec, indent: int = 4) -> List[str]:
    """The body of the ``calibration.board`` entry (see ``splice_board``) -- the physical board's
    geometry, so the next calibration auto-loads it and every result records which board it used."""
    pad = " " * indent
    b = spec.to_config_dict()
    lines = [f"{pad}squares_x: {b['squares_x']}", f"{pad}squares_y: {b['squares_y']}"]
    lines.append(f"{pad}square_length_m: {b['square_length_m']:g}")
    lines.append(f"{pad}marker_length_m: {b['marker_length_m']:g}")
    lines.append(f'{pad}dictionary: "{b["dictionary"]}"')
    return lines


def _locate_block(lines: List[str], search: range, key: str) -> Optional[tuple]:
    """``(key_line, key_indent, block_end)`` for a ``"<key>:"`` line found within ``search``, or
    None. ``block_end`` is the line the key's (more-indented) body runs up to -- the same
    "next line at or below this indent" rule ``exposure_tuner.splice_camera_options`` uses,
    with blank lines inside the body absorbed rather than treated as the end of it.
    """
    key_line = next((i for i in search if lines[i].strip().startswith(f"{key}:")), None)
    if key_line is None:
        return None
    key_indent = len(lines[key_line]) - len(lines[key_line].lstrip())
    block_end = key_line + 1
    for i in range(key_line + 1, search.stop):
        ln = lines[i]
        if not ln.strip():
            block_end = i + 1
            continue
        if len(ln) - len(ln.lstrip()) <= key_indent:
            break
        block_end = i + 1
    return key_line, key_indent, block_end


def _locate_path(lines: List[str], parents: Sequence[str]) -> tuple:
    """Walk ``parents`` (each must already exist) and return the LAST one's
    ``(indent, block_end, search_range_for_its_children)``. Raises ``ValueError`` naming
    whichever segment could not be found, rather than guessing where to invent it."""
    search = range(0, len(lines))
    indent, block_end = -2, len(lines)  # indent=-2 so a top-level child comes out at indent 0
    for name in parents:
        found = _locate_block(lines, search, name)
        if found is None:
            raise ValueError(f"no '{name}:' section found in config.yaml")
        key_line, indent, block_end = found
        search = range(key_line + 1, block_end)
    return indent, block_end, search


def _expand_scalar_camera(text: str, cam_key: str) -> str:
    """Rewrite ``cameras.<cam_key>: "<serial>"`` (the shorthand form) into the mapping form
    ``cameras.<cam_key>:\\n    serial: "<serial>"`` so an ``extrinsic`` child can be added under it.

    config.yaml documents both forms for a camera (a bare serial string, or a map with
    ``serial:`` + options); ``apply_camera_serials`` treats the scalar as the serial. A scalar
    node cannot also carry child keys, so writing an extrinsic under one would produce invalid
    YAML -- this converts it first, losing nothing (the serial moves onto its own line, any
    trailing comment is kept). No-op if the entry is already a mapping.
    """
    lines = text.splitlines()
    _pi, _pe, search = _locate_path(lines, ("cameras",))
    found = _locate_block(lines, search, cam_key)
    if found is None:
        raise ValueError(f"no '{cam_key}:' entry under 'cameras:' in config.yaml")
    key_line, indent, _block_end = found
    after = lines[key_line].split(":", 1)[1]
    value, comment = (after.split("#", 1) + [""])[:2]
    if not value.strip():
        return text  # already a mapping (nothing but maybe a comment after the colon)
    header = f"{' ' * indent}{cam_key}:" + (f"  #{comment}" if comment else "")
    serial_line = f"{' ' * (indent + 2)}serial: {value.strip()}"
    out = lines[:key_line] + [header, serial_line] + lines[key_line + 1 :]
    result = "\n".join(out)
    return result + "\n" if text.endswith("\n") and not result.endswith("\n") else result


def _ensure_section(text: str, parents: Sequence[str], key: str, *, comment: str = "") -> str:
    """Make sure ``<parents...>.<key>:`` exists as a bare (possibly empty) mapping, so a LEAF
    under it can be spliced in afterwards without first wiping out a sibling already saved
    there (e.g. adding the right arm's extrinsic must not erase the left arm's). No-op if it
    already exists -- never resets an existing section back to empty, and never re-adds
    ``comment`` on top of whatever the section already carries.

    ``comment`` is config.yaml's own convention (every existing field is documented inline) --
    written once, only at the moment this key is first created, since that is the only point a
    brand new field genuinely needs an explanation next to it.
    """
    lines = text.splitlines()
    parent_indent, parent_block_end, search = _locate_path(lines, parents)
    if _locate_block(lines, search, key) is not None:
        return text
    header = f"{' ' * (parent_indent + 2)}{key}:"
    if comment:
        header += f"  # {comment}"
    out = lines[:parent_block_end] + [header] + lines[parent_block_end:]
    result = "\n".join(out)
    return result + "\n" if text.endswith("\n") and not result.endswith("\n") else result


def _splice_leaf(text: str, parents: Sequence[str], key: str, block_lines: Sequence[str], *, comment: str = "") -> str:
    """Insert or REPLACE ``<parents...>.<key>:``'s own block. Every entry in ``parents`` must
    already exist (see ``_ensure_section`` to create an intermediate one first); ``key`` itself
    is replaced if present, or appended at the end of its parent's block if not. ``comment`` is
    rewritten on every call (unlike ``_ensure_section``'s once-only version) since a leaf that is
    replaced wholesale each time has no "already explained" state to preserve.
    """
    lines = text.splitlines()
    parent_indent, parent_block_end, search = _locate_path(lines, parents)
    header = f"{' ' * (parent_indent + 2)}{key}:"
    if comment:
        header += f"  # {comment}"
    replacement = [header, *block_lines]
    found = _locate_block(lines, search, key)
    if found is not None:
        key_line, _indent, block_end = found
        out = lines[:key_line] + replacement + lines[block_end:]
    else:
        out = lines[:parent_block_end] + replacement + lines[parent_block_end:]
    result = "\n".join(out)
    return result + "\n" if text.endswith("\n") and not result.endswith("\n") else result


def splice_agentview_extrinsic(text: str, arm: str, result: CalibrationResult, *, calibrated_at: str) -> str:
    """Insert/replace ``cameras.agentview.extrinsic.<arm>`` in a config.yaml (text in, text
    out). The OTHER arm's entry, if one was saved previously, and everything else in the file
    -- comments included -- survive untouched."""
    text = _ensure_section(
        text,
        ("cameras", "agentview"),
        "extrinsic",
        comment="base_T_agentview per arm's own FK frame (no shared robot-base frame -- see "
        "workstation/yam-data calibrate)",
    )
    block = format_extrinsic_yaml(result, calibrated_at=calibrated_at, indent=8)
    return _splice_leaf(text, ("cameras", "agentview", "extrinsic"), arm, block)


def splice_agentview_unified(text: str, result: UnifiedRigCalibration, *, calibrated_at: str) -> str:
    """Insert/replace ``cameras.agentview.extrinsic.unified`` -- the fused left+right answer,
    the recommended one for a consumer that does not want to redo the fusion itself.

    Safe to keep alongside ``left``/``right`` despite being derived from them: a save always
    recomputes and writes all three from the SAME live solve in one call (see
    ``CalibrateAgentviewWindow._on_save``), so there is no window where they could disagree from
    this tool's own writes -- only a config.yaml hand-edited afterward could make them drift, the
    same risk any other derived field in a hand-editable file carries.
    """
    text = _ensure_section(
        text,
        ("cameras", "agentview"),
        "extrinsic",
        comment="base_T_agentview per arm's own FK frame (no shared robot-base frame -- see "
        "workstation/yam-data calibrate)",
    )  # no-op if already there (see splice_agentview_extrinsic, which usually creates it first)
    block = format_unified_yaml(result, calibrated_at=calibrated_at, indent=8)
    return _splice_leaf(
        text,
        ("cameras", "agentview", "extrinsic"),
        "unified",
        block,
        comment="fused left+right through robot.arm_offset -- use this one unless you "
        "specifically need a single arm's own frame",
    )


def splice_arm_offset(text: str, result: ArmOffsetResult, *, calibrated_at: str) -> str:
    """Insert/replace the top-level ``robot.arm_offset`` block in a config.yaml."""
    block = format_arm_offset_yaml(result, calibrated_at=calibrated_at, indent=4)
    return _splice_leaf(
        text, ("robot",), "arm_offset", block, comment="left_T_right -- see workstation/yam-data calibrate"
    )


def splice_wrist_extrinsic(text: str, arm: str, result: WristExtrinsicResult, *, calibrated_at: str) -> str:
    """Insert/replace ``cameras.wrist_<arm>.extrinsic`` (``gripper_T_camera``) in a config.yaml.

    Writing this OVER the CAD default is the whole point of hand-eye: a consumer building a
    ``WristCameraGeometry`` should read this back instead of the hardcoded ``T_GRIPPER_CAMERA``
    when it exists. ``cameras.wrist_<arm>`` must already exist (it does -- every rig config lists
    the wrist cameras by serial); this only adds/replaces its ``extrinsic`` child.
    """
    cam_key = f"wrist_{arm}"
    text = _expand_scalar_camera(text, cam_key)  # so a scalar "wrist_x: serial" can take a child
    block = format_wrist_extrinsic_yaml(result, calibrated_at=calibrated_at, indent=6)
    return _splice_leaf(
        text,
        ("cameras", cam_key),
        "extrinsic",
        block,
        comment="gripper_T_camera, hand-eye solved -- overrides the CAD default in "
        "yam_policy.viz.geometry (see workstation/yam-data calibrate)",
    )


def splice_board(text: str, spec: BoardSpec, *, calibrated_at: str = "") -> str:
    """Insert/replace the top-level ``calibration.board`` block -- the ChArUco geometry the
    calibration used, so ``BoardSpec.from_config`` auto-loads it next time and every saved
    result records which physical board produced it. Creates the ``calibration:`` section if the
    file has none yet.

    ``calibrated_at`` is appended only when given (on a save that records provenance); a caller
    that just wants to persist the geometry can omit it.
    """
    text = _ensure_section(text, (), "calibration", comment="workstation/yam-data calibrate settings")
    block = format_board_yaml(spec, indent=4)
    if calibrated_at:
        block = [*block, f'    calibrated_at: "{calibrated_at}"']
    return _splice_leaf(
        text, ("calibration",), "board", block, comment="ChArUco board geometry -- MUST match the printed board"
    )
