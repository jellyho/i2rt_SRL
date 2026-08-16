"""Render the candidate-chunk fan for a real, recorded YAM deploy episode.

Offline analysis, not a live overlay -- on purpose. A live per-tick overlay was built once
(commit f615465) and dropped a day later (b4dee15/8c0e228): "watching the spread go past at
30fps turned out not to be worth much... visualisation belongs with the analysis, which reads
the dataset." This is that analysis, for real recorded data.

Self-contained: earlier versions imported ACRFT's ``examples/robocasa/hud.py`` for its camera
compositing, which pulled in a whole critic-decision dashboard (a matplotlib Q-grid panel, a
value-trace panel) that a plain multi-sample recording -- no critic, just the sampled spread --
never uses; ``hud.Dashboard.frame`` was always called with ``info=None`` here, which is the code
path that skips both panels. What actually got used was one thing: the path-overlay drawing
(``_draw_paths``, PIL-only, no matplotlib in it). ``_draw_fan`` below is that logic, vendored,
so this needs neither an ACRFT checkout nor matplotlib -- just PIL, which the recorder already
depends on. If a critic-scored dashboard is ever wanted here too, that is a reason to import the
real ``hud.Dashboard`` for THAT case, not to route this one through it.

Two sources of the overlaid path, ``--source``:
  * ``samples`` (default): the multi-candidate deploy FAN. Needs a LeRobot dataset recorded with
    ``yam-data deploy --num-samples N`` against an ACRFT server started with
    ``serve_policy.py --num-samples N`` (the SAME N on both sides -- schema fixed at handshake,
    see ``MultiSamplePolicy.extra_features``): every frame carries an ``action_samples`` column,
    one ``[N, 14]`` candidate-action snapshot per tick, candidate 0 the one actually executed.
  * ``action``: the ONE executed trajectory from the plain ``action`` column, drawn as a single
    path (no fan). This works on ANY LeRobot recording -- teleop, demo, replay -- so you can see
    what the robot did in a dataset without a multi-sample deploy. No ``--candidates`` needed.

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
  4. Also projects each arm's candidates into the FIXED agentview, through that arm's calibrated
     ``base_T_agentview`` (config, board-on-gripper solve): agentview does not ride the arm, so
     this needs no joint pose, no arm_offset, and no shared frame -- each arm's chunk lives in its
     own base frame and its own extrinsic places it (left fan green, right fan amber).
  5. Draws the fans on the recorded frames (``_draw_fan``) and writes an mp4.

    workstation/yam-data render-samples \\
        --repo-id my_deploy_run --episode 0 --arm left --horizon 30 --candidates 8 \\
        --out .scratch/deploy_samples.mp4
    # any dataset, executed trajectory only (no action_samples needed):
    workstation/yam-data render-samples \\
        --repo-id my_demo --episode 0 --arm left --source action --horizon 16 \\
        --out .scratch/executed_path.mp4

The WRIST intrinsics default to the same placeholder ``tests/test_wrist_view.py`` uses ("roughly
a D405 at 640x480"); the AGENTVIEW ones to a rough D455 placeholder (``--agent-fx`` etc). The
agentview EXTRINSIC is real (from config), so for its fan to line up exactly, pass the D455's own
factory intrinsics -- the same ones the board-on-gripper solve read. Until then the fan's *shape*
is meaningful, its exact pixel position is not. ``--agentview-arms`` picks which arms' fans to draw
(default both; an arm with no calibrated extrinsic is skipped with a note).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from PIL.ImageFont import FreeTypeFont, ImageFont
    from yam_policy.viz import CameraIntrinsics

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


def _load_action_chunk(reader: "DatasetReader", episode: int, start: int, horizon: int) -> Optional[np.ndarray]:
    """[1, H, 14] from the plain ``action`` column -- the ONE executed trajectory over the next H
    ticks, shaped as a single-candidate chunk so it flows through the same project/draw path.

    This is what makes the overlay work on ANY LeRobot recording (teleop, demo, replay), not just a
    multi-sample deploy: there is no fan, just the path the robot actually took from ``start``."""
    steps = []
    for k in range(horizon):
        a = reader.get_action(episode, start + k)
        if a is None:
            return None
        steps.append(np.asarray(a, dtype=float).reshape(-1))
    return np.stack(steps)[None]  # [1, H, 14]


def _project_fixed(points_base: np.ndarray, cam_t_base: np.ndarray, intr: "CameraIntrinsics") -> np.ndarray:
    """[H, 3] base-frame points -> [H, 2] pixels in a FIXED camera, NaN where behind the lens.

    The agentview counterpart of ``WristCameraGeometry.project``: that one derives the camera pose
    from the current joints (the camera rides the wrist), but agentview is bolted to the world, so
    the extrinsic is a constant ``cam_t_base`` (= ``agentview_T_base`` = ``inv(base_T_agentview)``,
    the calibrated pose from config) that does not depend on the arm pose at all. NaN (not the
    finite pixel the perspective divide would give) for points behind the lens, so a drawing
    routine skips them instead of plotting a mirrored path -- same contract as the wrist projector.
    """
    points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
    homo = np.concatenate([points_base, np.ones((len(points_base), 1))], axis=1)
    cam = (cam_t_base @ homo.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    u = intr.fx * cam[:, 0] / safe_z + intr.cx
    v = intr.fy * cam[:, 1] / safe_z + intr.cy
    px = np.stack([u, v], axis=1)
    px[~in_front] = np.nan
    return px


def _load_agentview_extrinsics(config_path: Optional[str], arms: List[str]) -> dict:
    """{arm: agentview_T_base (4x4)} for each requested arm that has a calibrated extrinsic in
    config.yaml (``cameras.agentview.extrinsic.<arm>.matrix`` is ``base_T_agentview``; inverted
    here to project base-frame points INTO agentview). Arms without one are omitted, with a note --
    the overlay just skips them rather than drawing a fan at a wrong pose."""
    from i2rt.serving.rig_config import find_rig, load_rig

    rig = load_rig(config_path)
    extr = ((rig.get("cameras") or {}).get("agentview") or {}).get("extrinsic") or {}
    out = {}
    for arm in arms:
        m = (extr.get(arm) or {}).get("matrix")
        if m is None:
            print(
                f"agentview: no calibrated extrinsic for '{arm}' in {find_rig(config_path)} "
                f"(run 'yam-data calibrate --board-on-gripper --arms {arm}') -- skipping its agentview fan",
                file=sys.stderr,
            )
            continue
        out[arm] = np.linalg.inv(np.asarray(m, dtype=float))
    return out


def _to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Squash a frame to ``size x size`` (non-uniform if the source is not already square).

    Doing this ourselves, once, and scaling the intrinsics to match with the same factor (see
    ``CameraIntrinsics.scaled_to``) makes the two non-uniform transforms cancel out, so a
    projected point still lands where the visible pixel it names actually is -- the recorded
    wrist frame (D405, 640x480) is not square, and projecting at one aspect ratio while drawing
    at another silently shifts every point."""
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def _finite_prefix(path: np.ndarray) -> Optional[list]:
    """A projected [H, 2] path, up to its first invisible point, as a list of (x, y) pairs.

    ``WristProjector`` marks a point NaN once it is behind the lens or off the sensor; drawing
    past that would either crash (PIL rejects NaN coordinates) or, worse, silently interpolate
    through it and invent geometry the camera never saw. None if the path starts invisible.
    """
    a = np.asarray(path, dtype=np.float64)
    finite = np.isfinite(a).all(axis=1)
    if not finite[0]:
        return None
    n = int(np.argmin(finite)) if not finite.all() else len(a)
    pts = [(float(x), float(y)) for x, y in a[:n]]
    return pts if len(pts) >= 2 else None


# Per-arm fan colours (chosen-candidate gradient start/end). Left keeps the original green; right
# gets an amber so both arms' fans read apart when drawn on the SAME agentview frame.
_FAN_COLORS = {
    "left": ((20, 140, 95), (190, 255, 225)),
    "right": ((170, 95, 20), (255, 205, 140)),
}


def _draw_fan(
    wrist_rgb: np.ndarray, paths: list, chosen: int = 0, *, colors: tuple = _FAN_COLORS["left"]
) -> np.ndarray:
    """Overlay every candidate's projected path onto one frame (wrist OR agentview).

    Ported from ACRFT's ``examples/robocasa/hud.py`` (``Dashboard._draw_paths``, PIL-only) and
    trimmed to what a plain multi-sample recording needs: no ``exec_steps``/committed-prefix
    split, since that is CriticSelectPolicy's partial-commit concept and MultiSamplePolicy always
    executes the whole chunk. The chosen candidate draws bright with a time gradient (so its
    direction of travel reads without an arrowhead); the rest draw translucent, drawn first so
    the chosen one stays on top. ``colors`` is ``(gradient_start, gradient_end)`` for the chosen
    candidate -- per arm, so a left and a right fan on one agentview frame stay distinguishable.

    Composited onto whatever is already on ``wrist_rgb``, so calling it twice (once per arm) layers
    the second arm's fan over the first rather than erasing it.
    """
    from PIL import Image, ImageDraw

    base = Image.fromarray(np.ascontiguousarray(wrist_rgb)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    for i, path in enumerate(paths):
        if i == chosen:
            continue
        pts = _finite_prefix(path)
        if pts is None:
            continue
        d.line(pts, fill=(150, 175, 200, 120), width=3)
        d.ellipse([pts[-1][0] - 2.5, pts[-1][1] - 2.5, pts[-1][0] + 2.5, pts[-1][1] + 2.5], fill=(150, 175, 200, 170))

    if 0 <= chosen < len(paths):
        pts = _finite_prefix(paths[chosen])
        if pts is not None:
            c0, c1 = colors
            n = len(pts) - 1
            for i in range(n):
                t = i / max(n - 1, 1)
                col = tuple(int(a + (b - a) * t) for a, b in zip(c0, c1, strict=True))
                d.line([pts[i], pts[i + 1]], fill=(*col, 255), width=5, joint="curve")
            ex, ey = pts[-1]
            d.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(*c1, 255))

    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def _font(size: int) -> "FreeTypeFont | ImageFont":
    """A monospace TTF if one is on disk, else PIL's built-in bitmap font.

    Same font-discovery *idea* as hf-utils' dataset render (``hfutil/dataset/video.find_font``)
    -- try common paths, degrade rather than fail -- reimplemented without matplotlib, which that
    one leans on as a last-resort source: pulling in matplotlib for a font path is the exact
    dependency this module exists to avoid (see the module docstring).
    """
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
    ):
        if pathlib.Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _label(
    canvas: "PILImage",
    xy: tuple,
    text: str,
    font: "FreeTypeFont | ImageFont",
    *,
    fill: tuple = (235, 238, 233),
    anchor: str = "la",
) -> None:
    """Draw ``text`` on a translucent black box, hf-utils' dataset-render look
    (``drawtext``'s ``boxcolor=black@0.5``) so a label reads on any footage behind it instead of
    only on the plain background this used to assume."""
    from PIL import Image, ImageDraw

    pad = 4
    probe = ImageDraw.Draw(canvas)
    l, t, r, b = probe.textbbox((0, 0), text, font=font, anchor="la")
    w, h = r - l, b - t
    x, y = xy
    if "r" in anchor:
        x -= w
    box = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(box).rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=(0, 0, 0, 140))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), box).convert("RGB"), (0, 0))
    ImageDraw.Draw(canvas).text((x, y), text, font=font, fill=fill, anchor="la")


def _compose_frame(
    agent_sq: np.ndarray, wrist_sq: np.ndarray, replan: int, step: int, cam: int, *, agent_label: str, wrist_label: str
) -> np.ndarray:
    """Agent view + fan-overlaid wrist view, hstacked flush (hf-utils' dataset-render layout),
    each panel bottom-left labelled on a translucent box instead of a separate header strip.

    No analytics column: there is no critic here to explain a decision with, only the sampled
    spread the fan already shows. ``hud.Dashboard``'s Q-grid/value-trace panels are exactly the
    part that does not apply -- see the module docstring for why they are not vendored too.
    """
    from PIL import Image

    canvas = Image.fromarray(np.concatenate([agent_sq, wrist_sq], axis=1))
    f_sm, f_md = _font(13), _font(15)
    _label(canvas, (8, 8), f"replan {replan}   step {step}", f_md)
    _label(canvas, (8, cam - 24), agent_label, f_sm)
    _label(canvas, (cam + 8, cam - 24), wrist_label, f_sm, fill=(178, 214, 197))
    return np.asarray(canvas)


def render(args: argparse.Namespace) -> pathlib.Path:
    from yam_policy.viz import CameraIntrinsics, WristCameraGeometry, WristProjector

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    from workstation.lerobot_recorder.dataset_reader import DatasetReader

    reader = DatasetReader(args.repo_id, args.root)
    reader.load()
    n_frames = reader.episode_length(args.episode)
    if not n_frames:
        raise SystemExit(f"episode {args.episode} not found (or empty) in {args.repo_id}")

    geometry = WristCameraGeometry(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
    # Native-resolution intrinsics, rescaled to the square canvas we draw on (see _to_square) --
    # not the recorded frame's own (non-square) resolution.
    camera_size = 256
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, width=640, height=480).scaled_to(
        camera_size, camera_size
    )
    # Which 7 of the 14 recorded action dims are each arm's (see workstation.lerobot_recorder
    # .config: joints 0..6 left, 7..13 right), and where in `observation.state`'s 42-d layout
    # each arm's joint POSITIONS sit (pos block only -- vel/eff do not describe the pose).
    action_slices = {"left": slice(0, 7), "right": slice(7, 14)}
    arm_slice = action_slices[args.arm]
    state_slice = slice(0, 7) if args.arm == "left" else slice(21, 28)
    wrist_key = "wrist_left" if args.arm == "left" else "wrist_right"

    projector = WristProjector(geometry, intrinsics, arm_slice=arm_slice)

    # Agentview overlay: a FIXED third-person camera, so each arm's fan projects through its
    # calibrated base_T_agentview (config, board-on-gripper solve) -- no wrist ride, no arm_offset,
    # no shared frame needed: each arm's chunk lives in its own base frame and its own extrinsic
    # places it. Arms without a calibrated extrinsic are skipped (see _load_agentview_extrinsics).
    agentview_arms = list(dict.fromkeys(args.agentview_arms))
    agent_extrinsics = _load_agentview_extrinsics(args.config, agentview_arms)
    agent_intr = CameraIntrinsics(
        fx=args.agent_fx, fy=args.agent_fy, cx=args.agent_cx, cy=args.agent_cy, width=640, height=480
    ).scaled_to(camera_size, camera_size)

    starts = _replan_starts(n_frames, args.horizon)
    if not starts:
        raise SystemExit(f"episode has {n_frames} frames, shorter than --horizon {args.horizon}")

    # "samples" = the multi-candidate deploy fan (action_samples column); "action" = the single
    # executed trajectory from the plain action column, so it works on ANY LeRobot recording.
    fan_word = "fan" if args.source == "samples" else "path"

    frames = []
    step = 0
    for replan_idx, start in enumerate(starts[: args.replans] if args.replans else starts):
        if args.source == "samples":
            chunk = _load_replan_chunk(reader, args.episode, start, args.horizon, args.candidates)
            missing = "no action_samples recorded here"
        else:
            chunk = _load_action_chunk(reader, args.episode, start, args.horizon)
            missing = "no action column here"
        if chunk is None:
            print(f"replan at frame {start}: {missing}, skipping", file=sys.stderr)
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
        if agent is None or wrist is None:
            print(f"replan at frame {start}: missing agentview/wrist image, skipping", file=sys.stderr)
            continue
        # Pre-squash both frames ourselves (see _to_square) so the wrist one lines up with the
        # already-square-scaled intrinsics above.
        agent_sq = _to_square(agent, camera_size)
        # Each calibrated arm's candidate fan, projected into the fixed agentview and layered on
        # (drawn per arm so left/right stay their own colour -- see _draw_fan / _FAN_COLORS).
        for arm, cam_t_base in agent_extrinsics.items():
            av_paths = [
                _project_fixed(geometry.chunk_to_path(cand[:, action_slices[arm]]), cam_t_base, agent_intr)
                for cand in chunk
            ]
            agent_sq = _draw_fan(agent_sq, av_paths, chosen=0, colors=_FAN_COLORS.get(arm, _FAN_COLORS["left"]))
        wrist_fan = _draw_fan(_to_square(wrist, camera_size), paths, chosen=0, colors=_FAN_COLORS[args.arm])
        agent_label = (
            f"agentview -- {'+'.join(agent_extrinsics)} {fan_word}" if agent_extrinsics else "agentview (no extrinsic)"
        )
        composed = _compose_frame(
            agent_sq,
            wrist_fan,
            replan_idx,
            step,
            camera_size,
            agent_label=agent_label,
            wrist_label=f"wrist {args.arm} -- candidate {fan_word}",
        )

        for _ in range(max(1, args.hold)):
            frames.append(composed)
            step += 1

    if not frames:
        raise SystemExit("nothing to render -- no replan had both action/action_samples and images")

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
    p.add_argument("--config", default=None, help="config.yaml for agentview extrinsics; auto-discovered by default")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--arm", choices=["left", "right"], default="left", help="which wrist camera panel to show")
    p.add_argument(
        "--agentview-arms",
        dest="agentview_arms",
        nargs="*",
        choices=["left", "right"],
        default=["left", "right"],
        help="which arms' candidate fans to overlay on agentview (default both; each uses its own "
        "calibrated base_T_agentview -- arms without one are skipped)",
    )
    p.add_argument(
        "--source",
        choices=["samples", "action"],
        default="samples",
        help="'samples' overlays the multi-candidate deploy fan (needs an action_samples column); "
        "'action' overlays the single executed trajectory from the plain action column, so it works "
        "on ANY LeRobot recording (teleop/demo/replay), not just a --num-samples deploy run",
    )
    p.add_argument("--horizon", type=int, required=True, help="how many future ticks to draw (deploy: action_horizon)")
    p.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="the --num-samples the server was started with (required for --source samples; ignored for action)",
    )
    p.add_argument("--replans", type=int, default=0, help="max replans to render (0 = the whole episode)")
    p.add_argument("--hold", type=int, default=5, help="frames to hold each replan on screen")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default=".scratch/deploy_samples.mp4")
    # Uncalibrated placeholder -- same numbers tests/test_wrist_view.py uses. The fan's shape is
    # meaningful before calibration; its exact pixel position is not (see module docstring).
    p.add_argument("--fx", type=float, default=430.0)
    p.add_argument("--fy", type=float, default=430.0)
    p.add_argument("--cx", type=float, default=320.0)
    p.add_argument("--cy", type=float, default=240.0)
    # Agentview (D455) intrinsics. The board-on-gripper solve USED the device's own factory
    # intrinsics, so for the agentview fan to line up exactly, pass THAT camera's numbers here
    # (rs-enumerate-devices, or the intrinsics the recorder logged). The defaults are a rough D455
    # placeholder at 640x480 -- the fan's shape is meaningful with them, its exact pixel position
    # is not, same caveat as the wrist intrinsics above.
    p.add_argument("--agent-fx", type=float, default=390.0)
    p.add_argument("--agent-fy", type=float, default=390.0)
    p.add_argument("--agent-cx", type=float, default=320.0)
    p.add_argument("--agent-cy", type=float, default=240.0)
    args = p.parse_args()
    if args.source == "samples" and args.candidates is None:
        p.error("--candidates is required with --source samples (the --num-samples the server used)")
    render(args)


if __name__ == "__main__":
    main()
