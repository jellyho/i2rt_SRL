"""Where a predicted action chunk would put the gripper, seen from the wrist camera.

The policy is a distribution over action chunks, not a single answer: flow matching maps
different noise to different chunks. Sampling several and drawing them all shows whether it is
confident (a tight bundle) or torn between options (a wide spray) — the thing a scalar loss
cannot tell you. ACRFT already does this for RoboCasa in ``examples/robocasa/action_overlay.py``.

Two things make the YAM version simpler and more honest than that one.

**The path is real.** RoboCasa's 12-d action carries an unnormalised OSC end-effector delta, so
that overlay cannot know how far a chunk actually reaches; it picks a scale per replan to make
the executed path a legible length and says so ("absolute magnitude is thus not to scale"). YAM
actions are joint positions, so running FK over a chunk gives the true path in metres. Nothing
is fudged, and two chunks that differ by a centimetre look like they differ by a centimetre.

**The camera pose is known, not calibrated.** i2rt publishes the arm with its RealSense D405
mount as one model (``i2rt/robot_models/arm/yam/v1/yam_linear_4310_d405.xml``), whose body chain
ends in a ``camera`` optical frame. So the wrist extrinsic comes from the manufacturer's CAD
rather than a hand-eye procedure -- see :data:`T_GRIPPER_CAMERA`.

The projection has to be redone every frame, because the camera is *on the wrist*: it moves with
the arm. RoboCasa recomputes its matrix every frame too, but for the opposite reason -- there the
camera is fixed to a mobile base. Same requirement, different half of the robot moving.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# gripper (flange) -> camera optical frame, for a YAM v1 with the linear_4310 gripper and the
# i2rt D405 mount. Composed from the body chain gripper -> camera_bracket -> camera_body ->
# camera in i2rt's own yam_linear_4310_d405.xml, and checked against the extrinsic its header
# states: pos [-0.07, 0, -0.077], quat_wxyz [0.153, 0.69, 0.69, 0.153], optical axis canted
# 25 deg off the flange -Z. The composition reproduces all three (25.00 deg), so this is the
# published transform at full precision rather than a re-derivation of it.
#
# The camera frame is +Z along the optical axis, X right, Y down -- the ROS/OpenCV convention a
# pinhole projection expects, so no axis juggling is needed downstream.
T_GRIPPER_CAMERA = np.array([
    [+0.000000000013, +0.906303416866, +0.422627633476, -0.070434978748],
    [+0.999999999993, +0.000001552386, -0.000003329044, -0.000000076568],
    [-0.000003673205, +0.422627633474, -0.906303416859, -0.077005929018],
    [+0.000000000000, +0.000000000000, +0.000000000000, +1.000000000000],
])

#: The body the extrinsic above is measured from, and the site that stands for "where the
#: gripper is". They are different frames: tcp_site is rotated 180 deg from the flange.
FLANGE_BODY = "gripper"
GRASP_SITE = "grasp_site"


class WristCameraGeometry:
    """FK for one arm, plus the projection into that arm's wrist camera.

    Built from the same combined arm+gripper MJCF the robot itself runs, so the kinematics
    cannot drift from the hardware's.
    """

    def __init__(self, xml_path: str, *, extrinsic: Optional[np.ndarray] = None) -> None:
        import mink
        import mujoco

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._config = mink.Configuration(self._model)
        self._nq = int(self._model.nq)
        self._extrinsic = np.asarray(T_GRIPPER_CAMERA if extrinsic is None else extrinsic, dtype=float)

    def _pad(self, q: Sequence[float]) -> np.ndarray:
        """Widen a joint vector to the model's nq.

        The model can have more joints than the robot reports -- a two-finger gripper against
        one normalised gripper value -- and the arm joints lead, so padding the tail leaves
        every frame from the flange outwards where it belongs.
        """
        out = np.zeros(self._nq, dtype=float)
        q = np.asarray(q, dtype=float).reshape(-1)
        n = min(q.size, self._nq)
        out[:n] = q[:n]
        return out

    def _frame(self, q: Sequence[float], name: str, kind: str) -> np.ndarray:
        self._config.update(self._pad(q))
        return self._config.get_transform_frame_to_world(name, kind).as_matrix()

    def grasp_pose(self, q: Sequence[float]) -> np.ndarray:
        """4x4 pose of the grasp point in the arm's base frame."""
        return self._frame(q, GRASP_SITE, "site")

    def camera_pose(self, q: Sequence[float]) -> np.ndarray:
        """4x4 pose of the wrist camera's optical frame in the arm's base frame."""
        return self._frame(q, FLANGE_BODY, "body") @ self._extrinsic

    def chunk_to_path(self, chunk: np.ndarray) -> np.ndarray:
        """An action chunk of joint targets -> [T, 3] grasp-point path in the base frame.

        ``chunk`` is [T, n_joints] for ONE arm, in the order the robot takes them. Any trailing
        gripper value is padded through and does not move the grasp point.
        """
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2:
            raise ValueError(f"expected a [T, joints] chunk, got shape {chunk.shape}")
        return np.stack([self.grasp_pose(q)[:3, 3] for q in chunk])

    def project(
        self,
        points_base: np.ndarray,
        q_now: Sequence[float],
        intrinsics: "CameraIntrinsics",
    ) -> np.ndarray:
        """[n, 3] points in the base frame -> [n, 2] (col, row) pixels, plus a validity mask.

        Returns ``[n, 3]`` where the last column is 1.0 for points genuinely in front of the
        camera and 0.0 otherwise. Points behind the lens still produce finite pixel coordinates
        through the divide, and drawing those puts a mirrored path on screen -- a failure that
        looks like a plausible prediction rather than an error.
        """
        points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
        world_to_cam = np.linalg.inv(self.camera_pose(q_now))
        homo = np.concatenate([points_base, np.ones((len(points_base), 1))], axis=1)
        cam = (world_to_cam @ homo.T).T[:, :3]

        z = cam[:, 2]
        in_front = z > 1e-6
        safe_z = np.where(in_front, z, 1.0)
        u = intrinsics.fx * cam[:, 0] / safe_z + intrinsics.cx
        v = intrinsics.fy * cam[:, 1] / safe_z + intrinsics.cy
        return np.stack([u, v, in_front.astype(float)], axis=1)


class CameraIntrinsics:
    """Pinhole intrinsics in pixels, for the resolution the frames actually arrive at."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """The same lens, reported for a resized frame.

        Frames are commonly resized before they reach a viewer, and intrinsics do not survive
        that untouched: projecting with the sensor's numbers onto a downscaled image puts every
        point too far from the centre, by exactly the resize factor.
        """
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    def __repr__(self) -> str:
        return (f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, {self.width}x{self.height})")


class WristProjector:
    """A callable that turns action chunks into image-pixel paths, for a HUD to draw.

    Matches the projector interface ACRFT's ``examples/robocasa/deploy_hud.py`` expects —
    ``chunks[N, H, A] -> list of [H, 2]`` — so it drops in where its ``SketchProjector`` sits.
    That one integrates two action dimensions into a corner minimap because, as it says, the
    fan has to be visible "before camera calibration/FK exist". Both now do: FK comes from the
    arm's own MJCF and the wrist extrinsic is published by i2rt, so this returns a real
    projection rather than a schematic.

    The pose matters and cannot be inferred from the chunk. The camera rides on the wrist, so
    where a future position lands on screen depends on where the arm is NOW; call
    :meth:`set_pose` each replan with the arm's current joints. Until it is called the projector
    uses the pose it was built with.

        from yam_policy.viz import WristCameraGeometry, CameraIntrinsics, WristProjector

        proj = WristProjector(WristCameraGeometry(mjcf), intrinsics, arm_slice=slice(0, 7))
        recorder = HudRecorder(mode="bon", horizon=H, projector=proj)
        ...
        proj.set_pose(state[0:7])                 # this arm's joints, this replan
        recorder.add(agent_rgb, wrist_rgb, response, step=t)

    Points behind the lens come back as NaN rather than as the finite pixel the perspective
    divide would give them: a drawing routine that plots those puts a mirrored path on screen,
    which reads as a confident wrong prediction rather than as missing data.
    """

    def __init__(self, geometry: "WristCameraGeometry", intrinsics: "CameraIntrinsics",
                 q_now: Optional[Sequence[float]] = None, *, arm_slice: Optional[slice] = None) -> None:
        self._geometry = geometry
        self._intrinsics = intrinsics
        self._arm_slice = arm_slice
        self._q = np.zeros(6) if q_now is None else np.asarray(q_now, dtype=float)

    def set_pose(self, q_now: Sequence[float]) -> None:
        """The arm's current joints. Call once per replan, before projecting."""
        self._q = np.asarray(q_now, dtype=float)

    def set_intrinsics(self, intrinsics: "CameraIntrinsics") -> None:
        self._intrinsics = intrinsics

    def __call__(self, chunks: np.ndarray) -> list:
        chunks = np.asarray(chunks, dtype=float)
        if chunks.ndim != 3:
            raise ValueError(f"expected [N, H, action_dim] chunks, got shape {chunks.shape}")
        if self._arm_slice is not None:
            chunks = chunks[:, :, self._arm_slice]

        out = []
        for chunk in chunks:
            projected = self._geometry.project(
                self._geometry.chunk_to_path(chunk), self._q, self._intrinsics
            )
            pixels = projected[:, :2].copy()
            pixels[projected[:, 2] < 0.5] = np.nan
            out.append(pixels)
        return out
