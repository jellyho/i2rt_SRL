"""ChArUco calibration math: the transform chain, and a real detector smoke test.

The chain (base_T_wrist @ wrist_T_board @ inv(agentview_T_board)) is pure numpy and the part
most at risk of a silent sign/inverse mistake, so it is tested against hand-built ground truth
with no camera or detector involved. The detector itself gets one real smoke test against an
image `cv2.aruco` renders and reads back -- not a physical rig, but real OpenCV code on both
ends rather than another layer of hand-built matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from workstation.lerobot_recorder.charuco import (
    ArmOffsetResult,
    ArmPairCapture,
    BoardSpec,
    Capture,
    _orthonormalize,
    _rotation_angle_deg,
    detect_board_pose,
    make_board,
    solve_agentview_extrinsic,
    solve_agentview_extrinsic_per_arm,
    solve_arm_offset,
    unify_rig_calibration,
)


def _rotation(axis: str, deg: float) -> np.ndarray:
    """A small hand-rolled Rodrigues rotation, so the test does not lean on cv2 for its own
    ground truth -- see cv2.Rodrigues used the same way in charuco.py itself."""
    import cv2

    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    rvec = np.array(axes[axis], dtype=np.float64) * np.radians(deg)
    R, _ = cv2.Rodrigues(rvec)
    return R


def _pose(t: tuple, axis: str = "x", deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rotation(axis, deg)
    T[:3, 3] = t
    return T


# --------------------------------------------------------------------------------------- #
# _orthonormalize / _rotation_angle_deg
# --------------------------------------------------------------------------------------- #
def test_orthonormalize_is_a_no_op_on_an_already_valid_rotation():
    R = _rotation("y", 37.0)
    assert np.allclose(_orthonormalize(R), R, atol=1e-9)


def test_orthonormalize_recovers_a_proper_rotation_from_a_perturbed_matrix():
    R = _rotation("z", 12.0) + np.random.default_rng(0).normal(0, 0.01, (3, 3))
    fixed = _orthonormalize(R)
    assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-9)  # orthogonal
    assert np.isclose(np.linalg.det(fixed), 1.0, atol=1e-9)  # proper (no reflection)


def test_rotation_angle_deg_matches_the_angle_it_was_built_from():
    for deg in (0.0, 15.0, 90.0, 179.0):
        assert np.isclose(_rotation_angle_deg(_rotation("x", deg)), deg, atol=1e-6)


# --------------------------------------------------------------------------------------- #
# solve_agentview_extrinsic: the transform chain, against known ground truth
# --------------------------------------------------------------------------------------- #
def test_recovers_the_true_extrinsic_from_noiseless_captures():
    """Build captures by RUNNING THE CHAIN FORWARD from a known base_T_agentview and known arm
    poses, then check solve_agentview_extrinsic inverts it -- the one thing this module exists
    to get right."""
    true_base_t_agent = _pose((1.2, -0.3, 0.8), axis="y", deg=25.0)
    board_pose_in_base = _pose((0.5, 0.1, 0.0), axis="z", deg=5.0)  # the board, sitting on the desk

    rng = np.random.default_rng(1)
    captures = []
    for _ in range(4):
        # A different arm pose each time -- the board does not move, the wrist camera does.
        base_t_wrist = _pose(rng.uniform(-0.1, 0.1, 3), axis="x", deg=float(rng.uniform(-20, 20)))
        wrist_t_board = np.linalg.inv(base_t_wrist) @ board_pose_in_base
        agent_t_board = np.linalg.inv(true_base_t_agent) @ board_pose_in_base
        captures.append(
            Capture(
                arm="left",
                base_t_wrist=base_t_wrist,
                wrist_t_board=wrist_t_board,
                agentview_t_board=agent_t_board,
                wrist_reproj_error_px=0.1,
                agentview_reproj_error_px=0.1,
            )
        )

    result = solve_agentview_extrinsic(captures)
    assert result.arm == "left"
    assert np.allclose(result.base_t_agentview, true_base_t_agent, atol=1e-8)
    assert result.translation_rms_mm < 1e-3
    assert result.rotation_rms_deg < 1e-3
    assert result.n_captures == 4


def test_disagreeing_captures_show_up_as_nonzero_spread_not_silently_averaged_away():
    """One capture perturbed away from the rest must move the RMS spread, not vanish into a
    mean that looks as confident as if every capture had agreed."""
    true_base_t_agent = _pose((1.0, 0.0, 0.5))
    board_pose_in_base = _pose((0.4, 0.0, 0.0))

    def make_capture(wrist_t: tuple, detection_error: np.ndarray | None = None) -> Capture:
        # wrist_t moves the ARM to a different (still internally-consistent) pose -- by itself
        # this cancels out of the chain exactly (base_t_wrist @ inv(base_t_wrist) @ board_pose),
        # regardless of where the arm is, which is the whole reason averaging several arm poses
        # is trustworthy. Only `detection_error` -- a wrong PnP READ, not a different truth --
        # can make one capture actually disagree with the rest.
        base_t_wrist = _pose(wrist_t)
        wrist_t_board = np.linalg.inv(base_t_wrist) @ board_pose_in_base
        if detection_error is not None:
            wrist_t_board = wrist_t_board @ detection_error
        return Capture(
            arm="left",
            base_t_wrist=base_t_wrist,
            wrist_t_board=wrist_t_board,
            agentview_t_board=np.linalg.inv(true_base_t_agent) @ board_pose_in_base,
            wrist_reproj_error_px=0.1,
            agentview_reproj_error_px=0.1,
        )

    agreeing = [make_capture((0.0, 0.0, 0.0)), make_capture((0.01, 0.0, 0.0)), make_capture((0.0, 0.01, 0.0))]
    clean = solve_agentview_extrinsic(agreeing)
    assert clean.translation_rms_mm < 1.0

    # A 5cm error in ONE capture's DETECTED board pose (a bad PnP read, not just a different arm
    # pose) -- solve_agentview_extrinsic cannot know it is wrong (no independent ground truth
    # either), so it must show up as spread instead of vanishing into the mean.
    outlier = [*agreeing, make_capture((0.0, 0.0, 0.0), detection_error=_pose((0.05, 0.0, 0.0)))]
    dirty = solve_agentview_extrinsic(outlier)
    assert dirty.translation_rms_mm > clean.translation_rms_mm
    assert max(dirty.per_capture_translation_mm) > 10.0  # the outlier itself is findable in the list


def test_refuses_to_solve_from_a_single_capture():
    """One capture cannot show whether the estimate is trustworthy -- there is nothing to
    compare it against, which is the entire point of averaging several."""
    c = Capture("left", np.eye(4), np.eye(4), np.eye(4), 0.0, 0.0)
    with pytest.raises(ValueError, match="at least 2"):
        solve_agentview_extrinsic([c])


# --------------------------------------------------------------------------------------- #
# Arms must never be pooled: each wrist camera is its own, unrelated FK frame (see the
# module docstring) -- there is no shared "robot base" anywhere in this codebase.
# --------------------------------------------------------------------------------------- #
def test_refuses_to_solve_captures_from_two_different_arms_together():
    """Silently averaging a left-wrist estimate with a right-wrist estimate would blend two
    numerically valid answers to two different questions -- this must be loud, not silent."""
    c_left = Capture("left", np.eye(4), np.eye(4), np.eye(4), 0.0, 0.0)
    c_right = Capture("right", np.eye(4), np.eye(4), np.eye(4), 0.0, 0.0)
    with pytest.raises(ValueError, match="more than one arm"):
        solve_agentview_extrinsic([c_left, c_right, c_left])


def test_per_arm_solves_each_arm_independently_and_reports_which_is_which():
    true_left_t_agent = _pose((1.0, 0.0, 0.5))
    true_right_t_agent = _pose((-1.0, 0.2, 0.5), axis="z", deg=180.0)  # a DIFFERENT frame/answer
    board_pose = _pose((0.4, 0.0, 0.0))

    def make(arm: str, true_t_agent: np.ndarray, wrist_t: tuple) -> Capture:
        base_t_wrist = _pose(wrist_t)
        return Capture(
            arm=arm,
            base_t_wrist=base_t_wrist,
            wrist_t_board=np.linalg.inv(base_t_wrist) @ board_pose,
            agentview_t_board=np.linalg.inv(true_t_agent) @ board_pose,
            wrist_reproj_error_px=0.1,
            agentview_reproj_error_px=0.1,
        )

    captures = [
        make("left", true_left_t_agent, (0.0, 0.0, 0.0)),
        make("left", true_left_t_agent, (0.02, 0.0, 0.0)),
        make("right", true_right_t_agent, (0.0, 0.0, 0.0)),
        make("right", true_right_t_agent, (0.0, -0.02, 0.0)),
    ]

    results = solve_agentview_extrinsic_per_arm(captures)

    assert set(results) == {"left", "right"}
    assert np.allclose(results["left"].base_t_agentview, true_left_t_agent, atol=1e-8)
    assert np.allclose(results["right"].base_t_agentview, true_right_t_agent, atol=1e-8)
    # If arms had been pooled, neither answer would match either true pose.
    assert not np.allclose(results["left"].base_t_agentview, true_right_t_agent, atol=1e-3)


def test_per_arm_skips_an_arm_with_too_few_captures_instead_of_raising():
    """A session that only managed one right-wrist capture should still get the left arm's
    result back, not lose everything to the arm that is not ready yet."""
    left_pose = _pose((1.0, 0.0, 0.5))
    board_pose = _pose((0.4, 0.0, 0.0))

    def make(arm: str, wrist_t: tuple) -> Capture:
        base_t_wrist = _pose(wrist_t)
        return Capture(
            arm=arm,
            base_t_wrist=base_t_wrist,
            wrist_t_board=np.linalg.inv(base_t_wrist) @ board_pose,
            agentview_t_board=np.linalg.inv(left_pose) @ board_pose,
            wrist_reproj_error_px=0.1,
            agentview_reproj_error_px=0.1,
        )

    captures = [make("left", (0.0, 0.0, 0.0)), make("left", (0.01, 0.0, 0.0)), make("right", (0.0, 0.0, 0.0))]
    results = solve_agentview_extrinsic_per_arm(captures)
    assert set(results) == {"left"}


# --------------------------------------------------------------------------------------- #
# solve_arm_offset: left_T_right from simultaneous both-wrists-see-the-board captures
# --------------------------------------------------------------------------------------- #
def test_recovers_the_true_arm_offset_and_its_distance():
    true_left_t_right = _pose((0.6, -0.05, 0.0), axis="z", deg=178.0)  # ~60cm apart, facing in
    board_pose = _pose((0.3, 0.0, 0.0))  # the board, between the two arms

    rng = np.random.default_rng(2)
    captures = []
    for _ in range(4):
        left_t_wrist = _pose(rng.uniform(-0.05, 0.05, 3), axis="y", deg=float(rng.uniform(-15, 15)))
        right_t_wrist = _pose(rng.uniform(-0.05, 0.05, 3), axis="y", deg=float(rng.uniform(-15, 15)))
        left_t_board_true = np.linalg.inv(left_t_wrist) @ board_pose
        # right's board pose, EXPRESSED IN right's own frame -- board_pose is in "left's frame"
        # here only because that is the ground truth we picked; right sees it through the
        # (also ground-truth) arm offset.
        right_t_board_true = np.linalg.inv(right_t_wrist) @ np.linalg.inv(true_left_t_right) @ board_pose
        captures.append(
            ArmPairCapture(
                left_t_wrist=left_t_wrist,
                left_wrist_t_board=left_t_board_true,
                right_t_wrist=right_t_wrist,
                right_wrist_t_board=right_t_board_true,
                left_reproj_error_px=0.1,
                right_reproj_error_px=0.1,
            )
        )

    result = solve_arm_offset(captures)
    assert np.allclose(result.left_t_right, true_left_t_right, atol=1e-8)
    assert np.isclose(result.distance_m, float(np.linalg.norm(true_left_t_right[:3, 3])), atol=1e-8)
    assert result.n_captures == 4
    assert result.translation_rms_mm < 1e-3


def test_arm_offset_refuses_a_single_capture():
    c = ArmPairCapture(np.eye(4), np.eye(4), np.eye(4), np.eye(4), 0.0, 0.0)
    with pytest.raises(ValueError, match="at least 2"):
        solve_arm_offset([c])


# --------------------------------------------------------------------------------------- #
# unify_rig_calibration: fuse + cross-check the two agentview routes via the arm offset
# --------------------------------------------------------------------------------------- #
def test_unify_fuses_agreeing_estimates_with_a_small_cross_check():
    """When left's direct solve and right's solve bridged through left_T_right agree (as they
    must, if every calibration was solved correctly against the same physical rig), the fused
    answer should sit close to both and the cross-check should be small."""
    true_left_t_agent = _pose((1.0, 0.0, 0.5))
    true_left_t_right = _pose((0.6, 0.0, 0.0), axis="z", deg=180.0)
    true_right_t_agent = np.linalg.inv(true_left_t_right) @ true_left_t_agent  # self-consistent

    left_result = solve_agentview_extrinsic(
        [
            Capture("left", np.eye(4), np.eye(4), np.linalg.inv(true_left_t_agent), 0.1, 0.1),
            Capture(
                "left",
                _pose((0.01, 0, 0)),
                np.linalg.inv(_pose((0.01, 0, 0))),
                np.linalg.inv(true_left_t_agent),
                0.1,
                0.1,
            ),
        ]
    )
    right_result = solve_agentview_extrinsic(
        [
            Capture("right", np.eye(4), np.eye(4), np.linalg.inv(true_right_t_agent), 0.1, 0.1),
            Capture(
                "right",
                _pose((0.01, 0, 0)),
                np.linalg.inv(_pose((0.01, 0, 0))),
                np.linalg.inv(true_right_t_agent),
                0.1,
                0.1,
            ),
        ]
    )
    arm_offset = ArmOffsetResult(
        true_left_t_right, float(np.linalg.norm(true_left_t_right[:3, 3])), 4, 0.0, 0.0, [], []
    )

    unified = unify_rig_calibration({"left": left_result, "right": right_result}, arm_offset)

    assert np.allclose(unified.left_t_agentview, true_left_t_agent, atol=1e-6)
    assert unified.cross_check_translation_mm is not None
    assert unified.cross_check_translation_mm < 1e-3
    assert unified.cross_check_rotation_deg < 1e-3
    assert np.isclose(unified.distance_m, 0.6, atol=1e-8)


def test_unify_falls_back_to_whichever_single_arm_is_available():
    """A rig calibrated so far from only one arm's captures still produces a usable answer --
    just without the cross-check, since there is nothing independent to check it against."""
    true_left_t_agent = _pose((1.0, 0.0, 0.5))
    left_result = solve_agentview_extrinsic(
        [
            Capture("left", np.eye(4), np.eye(4), np.linalg.inv(true_left_t_agent), 0.1, 0.1),
            Capture(
                "left",
                _pose((0.01, 0, 0)),
                np.linalg.inv(_pose((0.01, 0, 0))),
                np.linalg.inv(true_left_t_agent),
                0.1,
                0.1,
            ),
        ]
    )
    arm_offset = ArmOffsetResult(np.eye(4), 0.0, 2, 0.0, 0.0, [], [])

    unified = unify_rig_calibration({"left": left_result}, arm_offset)

    assert np.allclose(unified.left_t_agentview, true_left_t_agent, atol=1e-8)
    assert unified.cross_check_translation_mm is None
    assert unified.cross_check_rotation_deg is None


def test_unify_refuses_when_neither_arm_is_available():
    with pytest.raises(ValueError, match="neither"):
        unify_rig_calibration({}, ArmOffsetResult(np.eye(4), 0.0, 0, 0.0, 0.0, [], []))


# --------------------------------------------------------------------------------------- #
# detect_board_pose: one real cv2.aruco round trip (render -> detect), not hand-built matrices
# --------------------------------------------------------------------------------------- #
def test_detects_a_board_cv2_rendered_itself():
    spec = BoardSpec(squares_x=5, squares_y=4, square_length_m=0.03, marker_length_m=0.022)
    board = make_board(spec)
    img_size = 640
    image = board.generateImage((img_size, img_size), marginSize=20)

    # Intrinsics self-consistent with a frame that is (about) the board's own top-down render --
    # not a physical camera, but real enough for solvePnP to have something to solve.
    intr = {"fx": img_size, "fy": img_size, "cx": img_size / 2, "cy": img_size / 2}

    detection = detect_board_pose(image, spec, intr)

    assert detection is not None
    assert detection.n_corners >= 6
    assert detection.reproj_error_px < 1.0  # a clean synthetic render, not a photo -- should be tight
    assert detection.cam_t_board[2, 3] > 0  # the board is in front of the camera, not behind it


def test_returns_none_for_an_image_with_no_board_in_it():
    spec = BoardSpec()
    blank = np.full((480, 640, 3), 128, np.uint8)
    intr = {"fx": 600, "fy": 600, "cx": 320, "cy": 240}
    assert detect_board_pose(blank, spec, intr) is None
