"""Forward kinematics and wrist-camera projection for YAM.

The geometry a viewer needs to draw what a policy predicted, without the viewer. Drawing lives
wherever the analysis lives -- this side only answers "where would these joint targets put the
gripper, and where does that land in the wrist camera".

    from yam_policy.viz import WristCameraGeometry, CameraIntrinsics, WristProjector

    geometry = WristCameraGeometry(mjcf_path)      # the arm+gripper model the robot runs
    path     = geometry.chunk_to_path(chunk)       # [T, joints] -> [T, 3] in the arm's base frame
    pixels   = geometry.project(path, q_now, intrinsics)

Two things worth knowing before building on it.

**The paths are in metres.** YAM actions are joint targets, so FK over a chunk gives the real
path -- a bundle that looks tight IS tight. Overlays built on end-effector *deltas* cannot say
that; openpi's RoboCasa one rescales each replan to a legible length and says so.

**The wrist extrinsic is published, not calibrated.** i2rt ships the arm with its D405 mount as
one model whose body chain ends in a `camera` optical frame, so :data:`T_GRIPPER_CAMERA` is the
manufacturer's transform. See its docstring for how it was checked.

`mujoco` and `mink` are imported lazily and declared as the `viz` extra, so a policy server
never loads them.
"""

from yam_policy.viz.geometry import (
    FLANGE_BODY,
    GRASP_SITE,
    T_GRIPPER_CAMERA,
    CameraIntrinsics,
    WristCameraGeometry,
    WristProjector,
)

__all__ = [
    "WristCameraGeometry",
    "WristProjector",
    "CameraIntrinsics",
    "T_GRIPPER_CAMERA",
    "FLANGE_BODY",
    "GRASP_SITE",
]
