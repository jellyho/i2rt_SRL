"""Render the candidate-chunk fan for a real, recorded YAM deploy episode.

Offline analysis, not a live overlay -- on purpose. A live per-tick overlay was built once
(commit f615465) and dropped a day later (b4dee15/8c0e228): "watching the spread go past at
30fps turned out not to be worth much... visualisation belongs with the analysis, which reads
the dataset." This is that analysis, for real recorded data.

What it needs on disk:
  * A LeRobot dataset recorded with ``yam-data deploy --num-samples N`` against an ACRFT
    server started with ``serve_policy.py --num-samples N`` (the SAME N on both sides -- the
    dataset schema is fixed at handshake, see ``MultiSamplePolicy.extra_features``). That gives
    every recorded frame an ``action_samples`` column: one ``[N, 14]`` two-arm candidate-action
    snapshot per control tick, candidate 0 always the one actually executed.
  * ACRFT checked out somewhere reachable, to import its dashboard renderer
    (``examples/robocasa/hud.py``) -- pass ``--acrft-root`` or set ``$ACRFT_ROOT``.

What it does:
  1. Groups an episode's ticks into replans of ``--horizon`` consecutive frames -- the chunk
     size the server was started with. ActionChunkBroker re-queries the server only every
     ``action_horizon`` ticks, so that many consecutive recorded frames share one replan's
     candidates (see the module note on ``_replan_starts`` for what this assumes).
  2. Reassembles each replan's per-tick ``[N, 14]`` snapshots back into an ``[N, H, 14]``
     candidate chunk -- frame t's snapshot IS the candidates' predicted action at tick t of the
     chunk (see ``MultiSamplePolicy.infer``'s docstring, in ACRFT, for why the wire format is
     per-tick rather than candidate-major).
  3. Runs one arm's joint-target candidates through real forward kinematics
     (``WristCameraGeometry``) and projects into that arm's wrist camera (``WristProjector``),
     using the joint state at the start of the replan -- the camera rides the wrist, so a later
     pose does not describe where these candidates were projected from.
  4. Draws the fan on the recorded wrist frame with ``hud.Dashboard`` (``info=None`` -- there is
     no critic here, just the sampled spread) and writes an mp4.

    workstation/yam-data render-samples \\
        --repo-id my_deploy_run --episode 0 --arm left --horizon 30 \\
        --acrft-root ~/jellyho/ACRFT --out .scratch/deploy_samples.mp4

The camera intrinsics default to the same placeholder ``tests/test_wrist_view.py`` uses
("roughly a D405 at 640x480") -- pass ``--fx/--fy/--cx/--cy`` once the wrist camera is actually
calibrated; until then the fan's *shape* (tight vs. spread) is meaningful, its exact pixel
position is not.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from workstation.lerobot_recorder.dataset_reader import DatasetReader


def _replan_starts(n_frames: int, horizon: int) -> List[int]:
    """Tick indices where a fresh chunk starts, on a fixed grid from the episode's first frame.

    ActionChunkBroker re-queries the server exactly every ``horizon`` ticks starting from the
    first tick it drives, so replans land on a fixed grid *as long as streaming was continuous*.
    An intervention or a homing return forces an early replan (``_reset_policy_chunk``) that this
    does not know about -- a recorded episode that was interrupted mid-rollout will have its
    candidate reassembly drift out of alignment after the interruption. Fine for a first pass;
    revisit by reading ``observation.control_mode`` to re-sync on interruption if that matters.
    """
    return list(range(0, n_frames - horizon + 1, horizon))


def _load_replan_chunk(
    reader: "DatasetReader", episode: int, start: int, horizon: int, n_candidates: int
) -> Optional[np.ndarray]:
    """[N, H, 14] candidate chunk, reassembled from H consecutive per-tick [N, 14] snapshots."""
    steps = []
    for k in range(horizon):
        snap = reader.get_extra(episode, start + k, "action_samples", (n_candidates, 14))
        if snap is None:
            return None
        steps.append(snap)
    return np.stack(steps, axis=1)  # [N, H, 14]


def _to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Match the non-uniform squash ``hud.Dashboard.frame`` applies internally.

    Dashboard resizes whatever it is handed to a square ``_CAM x _CAM`` canvas with one scalar
    factor for both axes (``hud._draw_paths``'s ``k``), which is only correct if the source was
    already square -- the recorded wrist frame (D405, 640x480) is not. Doing the same squash here
    first, and scaling the intrinsics to match (see ``CameraIntrinsics.scaled_to``), makes the two
    non-uniform transforms cancel out, so a projected point still lands where the visible pixel
    it names actually is."""
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def render(args: argparse.Namespace) -> pathlib.Path:
    sys.path.insert(0, str(pathlib.Path(args.acrft_root).expanduser() / "examples" / "robocasa"))
    import hud
    from yam_policy.viz import CameraIntrinsics, WristCameraGeometry, WristProjector

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    from workstation.lerobot_recorder.dataset_reader import DatasetReader

    reader = DatasetReader(args.repo_id, args.root)
    reader.load()
    n_frames = reader.episode_length(args.episode)
    if not n_frames:
        raise SystemExit(f"episode {args.episode} not found (or empty) in {args.repo_id}")

    geometry = WristCameraGeometry(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
    # Native-resolution intrinsics, rescaled to the square canvas Dashboard/_draw_paths assumes
    # (see _to_square) -- not the recorded frame's own (non-square) resolution.
    camera_size = 256
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, width=640, height=480).scaled_to(
        camera_size, camera_size
    )
    # Which 7 of the 14 recorded action dims are this arm's (see workstation.lerobot_recorder
    # .config: joints 0..6 left, 7..13 right), and where in `observation.state`'s 42-d layout
    # this arm's joint POSITIONS sit (pos block only -- vel/eff do not describe the pose).
    arm_slice = slice(0, 7) if args.arm == "left" else slice(7, 14)
    state_slice = slice(0, 7) if args.arm == "left" else slice(21, 28)
    wrist_key = "wrist_left" if args.arm == "left" else "wrist_right"

    projector = WristProjector(geometry, intrinsics, arm_slice=arm_slice)
    dash = hud.Dashboard(mode="samples", horizon=args.horizon, camera_size=camera_size)

    starts = _replan_starts(n_frames, args.horizon)
    if not starts:
        raise SystemExit(f"episode has {n_frames} frames, shorter than --horizon {args.horizon}")

    frames = []
    step = 0
    for start in starts[: args.replans] if args.replans else starts:
        chunk = _load_replan_chunk(reader, args.episode, start, args.horizon, args.candidates)
        if chunk is None:
            print(f"replan at frame {start}: no action_samples recorded here, skipping", file=sys.stderr)
            continue
        state = reader.get_state(args.episode, start)
        if state is None:
            print(f"replan at frame {start}: no observation.state recorded here, skipping", file=sys.stderr)
            continue
        projector.set_pose(state[state_slice])
        paths = projector(chunk)  # [N] of [H, 2] pixel paths, this arm's candidates only

        images = reader.get_images(args.episode, start)
        agent = images.get("agentview")
        wrist = images.get(wrist_key)
        if agent is None:
            print(f"replan at frame {start}: no agentview image recorded here, skipping", file=sys.stderr)
            continue
        # Pre-squash the wrist frame ourselves (see _to_square) so it lines up with the
        # already-square-scaled intrinsics above; the agent view carries no overlay, so
        # Dashboard's own resize is fine for it as-is.
        wrist_sq = _to_square(wrist, camera_size) if wrist is not None else None

        for _ in range(max(1, args.hold)):
            frames.append(np.asarray(dash.frame(agent, wrist_sq, None, step, wrist_paths=paths, chosen=0)))
            step += 1

    if not frames:
        raise SystemExit("nothing to render -- no replan had both action_samples and images")

    import imageio

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {out} ({len(frames)} frames, {len(starts)} replans, {len(frames) // max(len(starts), 1)} held each)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="dataset repo id, as recorded (e.g. user/my_deploy_run)")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--horizon", type=int, required=True, help="action_horizon the server was started with")
    p.add_argument("--candidates", type=int, required=True, help="the --num-samples the server was started with")
    p.add_argument("--replans", type=int, default=0, help="max replans to render (0 = the whole episode)")
    p.add_argument("--hold", type=int, default=5, help="frames to hold each replan on screen")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default=".scratch/deploy_samples.mp4")
    p.add_argument(
        "--acrft-root", default=os.environ.get("ACRFT_ROOT", "~/jellyho/ACRFT"), help="checkout holding hud.py"
    )
    # Uncalibrated placeholder -- same numbers tests/test_wrist_view.py uses. The fan's shape is
    # meaningful before calibration; its exact pixel position is not (see module docstring).
    p.add_argument("--fx", type=float, default=430.0)
    p.add_argument("--fy", type=float, default=430.0)
    p.add_argument("--cx", type=float, default=320.0)
    p.add_argument("--cy", type=float, default=240.0)
    render(p.parse_args())


if __name__ == "__main__":
    main()
