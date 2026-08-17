"""ChArUco board detection + pose math for calibrating the AGENTVIEW camera.

The agentview camera is bolted high above the workspace, too far to co-see a desk board with a
wrist camera — so it is calibrated **eye-to-hand**: **grasp the ChArUco board with the gripper**
and lift it into agentview's view. The camera is fixed and the target rides the hand, so agentview
+ forward kinematics alone recover the one thing being solved for:

    base_T_agentview                       (agentview's pose in that arm's own FK base frame)

The relation per capture (agentview fixed, board on the moving flange) is::

    agentview_T_board = inv(base_T_agentview) @ base_T_flange @ flange_T_board

with two unknown constants: ``base_T_agentview`` (the answer) and ``flange_T_board`` (how the board
sits in the grip). Two captures divide out ``flange_T_board`` and leave ``AX = XB`` in
``base_T_agentview`` — solved with ``cv2.calibrateHandEye`` (Tsai-Lenz), the eye-to-hand dual of
eye-in-hand hand-eye. See ``solve_agentview_extrinsic_eyetohand``.

**Per arm, never pooled.** ``WristCameraGeometry`` builds FK from ONE arm's own MJCF
(``combine_arm_and_gripper_xml`` has no left/right parameter and no rig-level origin), so each
arm's "base frame" is that model's own (0,0,0) — unrelated to the other arm's. Two arms therefore
solve two ``base_T_agentview`` in two different, uncalibrated frames; they are stored separately
(``cameras.agentview.extrinsic.left``/``.right``) and each is valid only within its own arm's
frame. A consumer overlaying an arm's action uses that arm's own extrinsic — no shared "robot
base" frame exists anywhere in this codebase, and none is needed for that.

**The wrist cameras are NOT calibrated here.** Their ``gripper_T_camera`` comes from the CAD
constant (``yam_policy.viz.T_GRIPPER_CAMERA``, i2rt's published D405 mount) — measured against the
built hardware it was found accurate enough to use directly, so there is no wrist hand-eye step.

Intrinsics come from the RealSense devices themselves (``CameraManager.intrinsics``, factory
calibration off the device) rather than a checkerboard sweep — accurate enough for this, and it
removes a whole separate calibration stage.

Needs ``opencv-contrib-python`` (``cv2.aruco``, specifically the ``CharucoDetector``/
``CharucoBoard.matchImagePoints`` API introduced in OpenCV 4.7 — this does not fall back to the
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
    despite 5mm of real disagreement)."""
    return float(np.sqrt(np.mean(np.square(values))))


# --------------------------------------------------------------------------------------- #
# Board-on-gripper (eye-to-hand) agentview calibration -- see the module docstring.
# --------------------------------------------------------------------------------------- #
@dataclasses.dataclass
class EyeToHandCapture:
    """One board-on-gripper sample: the arm's FK pose + agentview's detection of the grasped board.

    ``arm`` records which FK model ``base_t_flange`` came from (agentview is solved in that arm's
    own frame -- see the module docstring on there being no shared robot-base frame).

    ``base_t_flange`` is the EXTRINSIC-FREE flange pose (``WristCameraGeometry.flange_pose``).

    The board must be grasped RIGIDLY and NOT re-gripped mid-run -- ``flange_T_board`` (how it sits
    on the gripper) is an unknown the solve recovers, but only under the assumption it stayed
    constant across the captures. A slip or a re-grasp between captures silently corrupts the solve
    (the solved ``flange_T_board``'s per-capture spread, reported as the RMS residual, is the check
    on this)."""

    arm: str  # "left" | "right" -- whichever WristCameraGeometry built base_t_flange
    base_t_flange: np.ndarray  # 4x4, WristCameraGeometry.flange_pose(q) -- FK only, no extrinsic
    agentview_t_board: np.ndarray  # 4x4, agentview camera PnP (camera_T_board) of the grasped board
    agentview_reproj_error_px: float


@dataclasses.dataclass
class AgentviewEyeToHandResult:
    arm: str  # which arm's FK frame base_t_agentview is expressed in -- see the module docstring
    base_t_agentview: np.ndarray  # 4x4, the answer, valid only in `arm`'s frame
    flange_t_board: np.ndarray  # 4x4, how the grasped board sat on the flange -- a solve by-product,
    #                             kept as a diagnostic (its per-capture spread IS the residual)
    n_captures: int
    # Consistency residual: with the solved base_T_agentview, flange_T_board should be the SAME (the
    # board did not move in the grip) across every capture. How far the per-capture estimates sit
    # from their mean is the honest end-to-end error -- it folds in FK error and a slipping grip.
    translation_rms_mm: float
    rotation_rms_deg: float
    per_capture_translation_mm: List[float]
    per_capture_rotation_deg: List[float]


def solve_agentview_extrinsic_eyetohand(captures: Sequence[EyeToHandCapture]) -> AgentviewEyeToHandResult:
    """``base_T_agentview`` for ONE arm, by eye-to-hand calibration off a board GRASPED by the gripper.

    Runs ``cv2.calibrateHandEye`` with the standard eye-to-hand trick: feed the INVERTED flange
    poses (``inv(base_t_flange)``, i.e. flange->base) as the "gripper2base" argument and
    ``agentview_t_board`` as "target2cam"; it then returns X = ``base_T_agentview``, and the
    constant it divided out is ``flange_T_board`` (recovered below for the residual).

    Needs at least 3 captures with varied ORIENTATION (not just position) -- a pure translation
    between poses leaves the rotation of X unconstrained (AX = XB degenerates).
    """
    import cv2

    if len(captures) < 3:
        raise ValueError("eye-to-hand needs at least 3 captures (2 relative motions) to solve X")
    arms = {c.arm for c in captures}
    if len(arms) > 1:
        raise ValueError(f"captures span more than one arm ({sorted(arms)}) -- each arm's frame is its own solve")
    (arm,) = arms

    inv_flange = [np.linalg.inv(c.base_t_flange) for c in captures]
    r_g2b = [m[:3, :3] for m in inv_flange]
    t_g2b = [m[:3, 3] for m in inv_flange]
    r_t2c = [c.agentview_t_board[:3, :3] for c in captures]
    t_t2c = [c.agentview_t_board[:3, 3] for c in captures]
    r_x, t_x = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    x = np.eye(4)
    x[:3, :3] = r_x
    x[:3, 3] = np.asarray(t_x).reshape(3)
    # On too-similar poses cv2 can hand back a non-finite X (logging "Not enough informative
    # motions") rather than raising, so a NaN never reaches the config.
    if not np.all(np.isfinite(x)):
        raise ValueError("eye-to-hand did not converge (too-similar poses -- vary the grip tilt more)")

    # flange_T_board per capture, which should be identical if X (and the FK, and the grip) are
    # right. From agentview_T_board = inv(base_T_agentview) @ base_T_flange @ flange_T_board, so
    # flange_T_board = inv(base_T_flange) @ base_T_agentview @ agentview_T_board.
    boards = [np.linalg.inv(c.base_t_flange) @ x @ c.agentview_t_board for c in captures]
    mean_board, per_t, per_r = _average_poses(boards)
    return AgentviewEyeToHandResult(
        arm=arm,
        base_t_agentview=x,
        flange_t_board=mean_board,
        n_captures=len(captures),
        translation_rms_mm=_rms(per_t),
        rotation_rms_deg=_rms(per_r),
        per_capture_translation_mm=per_t,
        per_capture_rotation_deg=per_r,
    )


def solve_agentview_extrinsic_eyetohand_per_arm(
    captures: Sequence[EyeToHandCapture],
) -> Dict[str, AgentviewEyeToHandResult]:
    """Group board-on-gripper captures by ``arm`` and solve each independently.

    An arm with fewer than 3 captures, or one whose solve does not converge (too-similar poses),
    is silently skipped -- so a session that has only got one arm to enough varied poses still gets
    that one back."""
    by_arm: Dict[str, List[EyeToHandCapture]] = {}
    for c in captures:
        by_arm.setdefault(c.arm, []).append(c)
    out: Dict[str, AgentviewEyeToHandResult] = {}
    for arm, group in by_arm.items():
        if len(group) < 3:
            continue
        try:
            out[arm] = solve_agentview_extrinsic_eyetohand(group)
        except ValueError:
            pass  # not converged yet -> leave it out
    return out


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


def format_extrinsic_yaml(
    result: AgentviewEyeToHandResult, *, calibrated_at: str, indent: int = 8, method: str = ""
) -> List[str]:
    """The body of one arm's ``extrinsic`` entry -- ready to splice under
    ``cameras.agentview.extrinsic.<arm>`` (see ``splice_agentview_extrinsic``). ``matrix`` is
    ``base_T_agentview``. ``method`` (e.g. ``"eye_to_hand"``) records how it was produced."""
    pad = " " * indent
    lines = [f"{pad}matrix:", *_format_matrix_yaml(result.base_t_agentview, indent + 2)]
    if method:
        lines.append(f'{pad}method: "{method}"')
    lines += [
        f"{pad}n_captures: {result.n_captures}",
        f"{pad}translation_rms_mm: {result.translation_rms_mm:.3f}",
        f"{pad}rotation_rms_deg: {result.rotation_rms_deg:.4f}",
        f'{pad}calibrated_at: "{calibrated_at}"',
    ]
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


def splice_agentview_extrinsic(
    text: str, arm: str, result: AgentviewEyeToHandResult, *, calibrated_at: str, method: str = "eye_to_hand"
) -> str:
    """Insert/replace ``cameras.agentview.extrinsic.<arm>`` in a config.yaml (text in, text
    out). The OTHER arm's entry, if one was saved previously, and everything else in the file
    -- comments included -- survive untouched."""
    text = _expand_scalar_camera(text, "agentview")  # so a scalar "agentview: serial" can take a child
    text = _ensure_section(
        text,
        ("cameras", "agentview"),
        "extrinsic",
        comment="base_T_agentview per arm's own FK frame (no shared robot-base frame -- see "
        "workstation/yam-data calibrate)",
    )
    block = format_extrinsic_yaml(result, calibrated_at=calibrated_at, indent=8, method=method)
    return _splice_leaf(text, ("cameras", "agentview", "extrinsic"), arm, block)


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
