"""ChArUco calibration math: the eye-to-hand transform, and a real detector smoke test.

The eye-to-hand solve (board grasped on the gripper -> base_T_agentview) is pure numpy over
cv2.calibrateHandEye and the part most at risk of a silent sign/inverse mistake, so it is tested
against hand-built ground truth. The detector itself gets one real smoke test against an image
`cv2.aruco` renders and reads back -- not a physical rig, but real OpenCV code on both ends.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from workstation.lerobot_recorder.charuco import (
    BoardSpec,
    EyeToHandCapture,
    _orthonormalize,
    _rotation_angle_deg,
    detect_board_pose,
    make_board,
    solve_agentview_extrinsic_eyetohand,
    solve_agentview_extrinsic_eyetohand_per_arm,
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
# Board-on-gripper (eye-to-hand) agentview calibration
# --------------------------------------------------------------------------------------- #
def _eth_captures(true_base_t_agentview, true_flange_t_board, *, arm="left", n=8, seed=0):
    """Board-on-gripper captures generated from a known base_T_agentview + flange_T_board.

    The relation the solve inverts is agentview_T_board = inv(base_T_agentview) @ base_T_flange
    @ flange_T_board, so we synthesise agentview's detection from a varied-orientation flange
    pose exactly that way -- noiseless, so the solve should recover both to floating-point."""
    rng = np.random.default_rng(seed)
    caps = []
    for _ in range(n):
        R = _rotation("x", rng.uniform(-40, 40)) @ _rotation("y", rng.uniform(-40, 40))
        base_t_flange = np.eye(4)
        base_t_flange[:3, :3] = R @ _rotation("z", rng.uniform(-40, 40))
        base_t_flange[:3, 3] = rng.uniform(-0.2, 0.2, 3) + np.array([0.0, 0.0, 0.35])
        agentview_t_board = np.linalg.inv(true_base_t_agentview) @ base_t_flange @ true_flange_t_board
        caps.append(EyeToHandCapture(arm, base_t_flange, agentview_t_board, 0.2))
    return caps


def test_eyetohand_recovers_the_true_agentview_pose_and_grip():
    x_true = _pose((0.4, -0.1, 0.9), "x", 150.0)  # base_T_agentview (mounted high, looking down)
    y_true = _pose((0.03, 0.0, 0.12), "y", 30.0)  # flange_T_board (how the grasped board sits)
    result = solve_agentview_extrinsic_eyetohand(_eth_captures(x_true, y_true))
    assert np.allclose(result.base_t_agentview, x_true, atol=1e-9)
    assert np.allclose(result.flange_t_board, y_true, atol=1e-9)  # nuisance recovered too
    assert result.translation_rms_mm < 1e-6  # noiseless -> the grip stays consistent to machine eps
    assert result.n_captures == 8


def test_eyetohand_refuses_fewer_than_three_captures():
    x_true, y_true = _pose((0.4, 0, 0.9)), _pose((0.03, 0, 0.12))
    with pytest.raises(ValueError):
        solve_agentview_extrinsic_eyetohand(_eth_captures(x_true, y_true, n=2))


def test_eyetohand_refuses_mixed_arms():
    x_true, y_true = _pose((0.4, 0, 0.9)), _pose((0.03, 0, 0.12))
    caps = _eth_captures(x_true, y_true, n=4)
    caps[-1] = EyeToHandCapture("right", caps[-1].base_t_flange, caps[-1].agentview_t_board, 0.2)
    with pytest.raises(ValueError):
        solve_agentview_extrinsic_eyetohand(caps)


def test_eyetohand_per_arm_solves_each_arm_in_its_own_frame():
    y_true = _pose((0.03, 0.0, 0.12), "y", 30.0)  # same physical board, but each arm grips its own way
    x_left = _pose((0.4, -0.1, 0.9), "x", 150.0)
    x_right = _pose((0.1, -0.6, 0.9), "x", 150.0)
    caps = _eth_captures(x_left, y_true, arm="left", n=6, seed=1) + _eth_captures(
        x_right, y_true, arm="right", n=5, seed=2
    )
    out = solve_agentview_extrinsic_eyetohand_per_arm(caps)
    assert set(out) == {"left", "right"}
    assert np.allclose(out["left"].base_t_agentview, x_left, atol=1e-9)
    assert np.allclose(out["right"].base_t_agentview, x_right, atol=1e-9)


def test_eyetohand_per_arm_skips_an_arm_below_three():
    x_true, y_true = _pose((0.4, 0, 0.9)), _pose((0.03, 0, 0.12))
    caps = _eth_captures(x_true, y_true, arm="left", n=5, seed=1) + _eth_captures(
        x_true, y_true, arm="right", n=2, seed=2
    )
    out = solve_agentview_extrinsic_eyetohand_per_arm(caps)
    assert set(out) == {"left"}  # the 2-capture right arm is dropped, not an error


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
