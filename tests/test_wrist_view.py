"""Wrist-camera geometry: FK to the flange, the published D405 extrinsic, and projection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from yam_policy.viz import (
    T_GRIPPER_CAMERA,
    CameraIntrinsics,
    WristCameraGeometry,
)


@pytest.fixture(scope="module")
def geometry():
    return WristCameraGeometry(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))


def _intrinsics():
    """Roughly a D405 at 640x480. Only the projection MATH is under test here; the real
    numbers come off the RealSense at run time."""
    return CameraIntrinsics(fx=430, fy=430, cx=320, cy=240, width=640, height=480)


def test_extrinsic_matches_the_figures_i2rt_publishes():
    """The constant is i2rt's transform at full precision, not a re-derivation of it.

    Its model states flange -> camera as pos [-0.07, 0, -0.077], quat_wxyz
    [0.153, 0.69, 0.69, 0.153], with the optical axis canted 25 deg off the flange -Z. Checking
    against all three is what makes it safe to carry the matrix here instead of vendoring the
    model and its meshes.
    """
    pos = T_GRIPPER_CAMERA[:3, 3]
    assert np.allclose(pos, [-0.07, 0.0, -0.077], atol=2e-3), pos

    rot = T_GRIPPER_CAMERA[:3, :3]
    w = np.sqrt(max(0.0, 1.0 + np.trace(rot))) / 2
    quat = np.array([w, (rot[2, 1] - rot[1, 2]) / (4 * w),
                     (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w)])
    quat /= np.linalg.norm(quat)
    assert np.allclose(np.abs(quat), [0.153, 0.69, 0.69, 0.153], atol=2e-3), quat

    optical_axis = rot @ np.array([0.0, 0.0, 1.0])
    cant = np.degrees(np.arccos(np.clip(optical_axis @ np.array([0.0, 0.0, -1.0]), -1, 1)))
    assert abs(cant - 25.0) < 0.1, cant

    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9), "not a rotation"


def test_the_wrist_camera_looks_at_the_gripper(geometry):
    """The strongest check available without hardware.

    The camera is bolted to the same body as the fingers, so the grasp point must land at the
    SAME pixel whatever the arm is doing -- and that pixel must be on screen. Get the extrinsic
    or the FK frame wrong (tcp_site is rotated 180 deg from the flange, an easy mix-up) and this
    fails in both ways at once: the point wanders with pose, and wanders off frame.
    """
    intrinsics = _intrinsics()
    rng = np.random.default_rng(0)
    poses = [np.zeros(6)] + [rng.uniform(-0.6, 0.6, 6) for _ in range(8)]

    pixels = []
    for q in poses:
        grasp = geometry.grasp_pose(q)[:3, 3]
        u, v, in_front = geometry.project(grasp[None], q, intrinsics)[0]
        assert in_front == 1.0, "the gripper is never behind its own wrist camera"
        assert 0 <= u < intrinsics.width and 0 <= v < intrinsics.height, (u, v)
        pixels.append((u, v))

    spread = np.asarray(pixels).std(axis=0)
    assert np.allclose(spread, 0, atol=1e-6), f"rigidly mounted, yet the pixel moved: {spread}"


def test_a_point_on_the_optical_axis_lands_on_the_principal_point(geometry):
    intrinsics = _intrinsics()
    q = np.zeros(6)
    on_axis = (geometry.camera_pose(q) @ np.array([0.0, 0.0, 0.3, 1.0]))[:3]
    u, v, in_front = geometry.project(on_axis[None], q, intrinsics)[0]
    assert in_front == 1.0
    assert np.allclose([u, v], [intrinsics.cx, intrinsics.cy], atol=1e-6)


def test_points_behind_the_lens_are_flagged_not_drawn(geometry):
    """A point behind the camera still divides to a finite pixel, and drawing it puts a
    mirrored path on screen -- which reads as a confident wrong prediction rather than a bug."""
    intrinsics = _intrinsics()
    q = np.zeros(6)
    behind = (geometry.camera_pose(q) @ np.array([0.0, 0.0, -0.3, 1.0]))[:3]
    assert geometry.project(behind[None], q, intrinsics)[0, 2] == 0.0


def test_a_chunk_of_joint_targets_becomes_a_metric_path(geometry):
    """FK per step, so the path is in real metres.

    RoboCasa's overlay cannot do this: its action is an unnormalised OSC delta, so it rescales
    each replan to a legible length and the absolute magnitude is not to scale. Joint targets
    have no such ambiguity, and the test pins that down by moving one joint a known amount and
    checking the path length is physical rather than merely non-zero.
    """
    chunk = np.zeros((10, 7))
    chunk[:, 1] = np.linspace(0.0, 0.5, 10)
    path = geometry.chunk_to_path(chunk)

    assert path.shape == (10, 3)
    assert np.allclose(path[0], geometry.grasp_pose(chunk[0])[:3, 3])
    travelled = np.linalg.norm(path[-1] - path[0])
    assert 0.05 < travelled < 0.5, travelled          # a 0.5 rad swing on a ~0.5 m arm
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.all(steps > 0), "a monotonic joint sweep cannot stand still"


def test_a_still_chunk_produces_a_still_path(geometry):
    path = geometry.chunk_to_path(np.zeros((5, 7)))
    assert np.allclose(path - path[0], 0)


def test_chunk_shape_is_checked(geometry):
    with pytest.raises(ValueError, match="chunk"):
        geometry.chunk_to_path(np.zeros(7))


def test_intrinsics_follow_a_resize():
    """Frames are usually resized before anyone looks at them, and intrinsics do not survive
    that: projecting with the sensor's numbers onto a half-size image puts every point twice as
    far from the centre as it belongs."""
    full = _intrinsics()
    half = full.scaled_to(320, 240)
    assert (half.fx, half.fy, half.cx, half.cy) == (215.0, 215.0, 160.0, 120.0)
    assert (half.width, half.height) == (320, 240)


def test_the_gripper_value_does_not_move_the_grasp_point(geometry):
    """The chunk carries a trailing gripper command; opening the fingers must not be mistaken
    for the arm reaching somewhere."""
    q_closed = np.zeros(7)
    q_open = np.zeros(7)
    q_open[6] = 0.04
    assert np.allclose(geometry.grasp_pose(q_closed)[:3, 3], geometry.grasp_pose(q_open)[:3, 3])


# --------------------------------------------------------------------------------------- #
# The projector ACRFT's HUD plugs in
# --------------------------------------------------------------------------------------- #
def test_the_projector_matches_the_hud_interface(geometry):
    """ACRFT's deploy_hud expects `chunks[N, H, A] -> list of [H, 2]`, and ships a
    SketchProjector that integrates two action dims into a corner minimap because, as it says,
    the fan has to be visible "before camera calibration/FK exist". Both exist now, so this
    returns a real projection in the same shape."""
    from yam_policy.viz import WristProjector

    projector = WristProjector(geometry, _intrinsics(), arm_slice=slice(0, 7))
    chunks = np.zeros((6, 30, 14))
    for i in range(6):
        chunks[i, :, 1] = np.linspace(0.0, 0.05 * (i + 1), 30)

    projector.set_pose(np.zeros(6))
    paths = projector(chunks)

    assert isinstance(paths, list) and len(paths) == 6
    assert all(p.shape == (30, 2) for p in paths)
    assert len({tuple(np.round(p[-1], 1)) for p in paths}) == 6, "candidates must diverge"


def test_the_projection_follows_the_arm(geometry):
    """The camera rides on the wrist, so the same chunk lands somewhere else once the arm has
    moved. A projector that ignores set_pose would return identical pixels — and quietly draw
    the fan for a pose the robot left."""
    from yam_policy.viz import WristProjector

    projector = WristProjector(geometry, _intrinsics(), arm_slice=slice(0, 7))
    chunks = np.zeros((1, 30, 14))
    chunks[0, :, 1] = np.linspace(0.0, 0.2, 30)

    projector.set_pose(np.zeros(6))
    before = projector(chunks)[0]
    projector.set_pose(np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0]))
    after = projector(chunks)[0]

    assert np.nanmax(np.abs(after - before)) > 1.0


def test_points_behind_the_lens_come_back_as_nan(geometry):
    """Not as the finite pixel the divide would give them: a drawing routine that plots those
    puts a mirrored path on screen, which reads as a confident wrong prediction."""
    from yam_policy.viz import WristProjector

    projector = WristProjector(geometry, _intrinsics(), arm_slice=slice(0, 7))
    projector.set_pose(np.zeros(6))
    # A chunk that swings the arm hard behind the current camera pose.
    chunks = np.zeros((1, 30, 14))
    chunks[0, :, 1] = np.linspace(0.0, -2.5, 30)
    path = projector(chunks)[0]
    assert np.isnan(path).any(), "nothing was flagged as behind the lens"


def test_chunk_rank_is_checked(geometry):
    from yam_policy.viz import WristProjector

    projector = WristProjector(geometry, _intrinsics())
    with pytest.raises(ValueError, match="chunks"):
        projector(np.zeros((30, 14)))
