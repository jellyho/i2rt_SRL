"""Draw what a YAM policy predicted, on the frames it predicted from.

A policy is a distribution over action chunks, not a single answer. Sampling several and
drawing them together is the only cheap way to see whether it is confident (a tight bundle) or
torn between options (a wide spray) — a scalar loss averages exactly that away.

This is the geometry and drawing half, packaged so a policy repo can import it and build its
own views. The three pieces stack:

    geometry     joint targets -> a metric end-effector path -> pixels in a wrist camera
    overlay      those pixels -> lines on a frame
    sample_log   the chunks a run actually sampled, recorded beside its dataset

Two things worth knowing before building on it.

**The paths are in metres.** YAM actions are joint targets, so FK over a chunk gives the real
path. Overlays built on end-effector *deltas* cannot do this — openpi's RoboCasa one rescales
each replan to a legible length and says so — which means a bundle that looks tight here IS
tight, and two chunks a centimetre apart look a centimetre apart.

**The wrist extrinsic is published, not calibrated.** i2rt ships the arm with its D405 mount as
one model whose body chain ends in a `camera` optical frame, so :data:`T_GRIPPER_CAMERA` is the
manufacturer's transform rather than something measured in a lab. See its docstring.

    from yam_policy.viz import WristCameraGeometry, CameraIntrinsics, overlay_samples

    geometry = WristCameraGeometry(mjcf_path)          # arm + gripper, as the robot runs it
    frame = overlay_samples(frame, geometry, samples, q_now, intrinsics)

`mujoco` and `mink` are needed for the kinematics and `pillow` for the drawing; they are
imported lazily, so importing this package costs nothing until something is actually drawn.
"""

from yam_policy.viz.geometry import (
    FLANGE_BODY,
    GRASP_SITE,
    T_GRIPPER_CAMERA,
    CameraIntrinsics,
    WristCameraGeometry,
)
from yam_policy.viz.overlay import (
    ARM_ACTION_SLICES,
    ARM_STATE_SLICES,
    WRIST_CAMERA_ARMS,
    WristOverlayRenderer,
    draw_chunk_paths,
    overlay_samples,
    project_samples,
)
from yam_policy.viz.sample_log import EpisodeSampleLog, episode_path, load, samples_at

__all__ = [
    # geometry
    "WristCameraGeometry",
    "CameraIntrinsics",
    "T_GRIPPER_CAMERA",
    "FLANGE_BODY",
    "GRASP_SITE",
    # drawing
    "overlay_samples",
    "project_samples",
    "draw_chunk_paths",
    "WristOverlayRenderer",
    "ARM_STATE_SLICES",
    "ARM_ACTION_SLICES",
    "WRIST_CAMERA_ARMS",
    # recorded samples
    "EpisodeSampleLog",
    "load",
    "samples_at",
    "episode_path",
]
