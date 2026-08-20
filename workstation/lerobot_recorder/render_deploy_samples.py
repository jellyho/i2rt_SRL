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
  1. Tiles an episode into chunks of ``--horizon`` consecutive frames on a fixed grid (the last
     one may be shorter, so the WHOLE trajectory is covered). This mirrors ActionChunkBroker,
     which re-queries the server every ``action_horizon`` ticks -- as long as streaming was
     continuous; an intervention/homing forces an early replan the grid does not know about, so an
     interrupted episode drifts out of alignment after the interruption (revisit via
     ``observation.control_mode`` if that matters).
  2. Reassembles each chunk's per-tick ``[N, 14]`` snapshots back into an ``[N, H, 14]`` candidate
     chunk -- frame t's snapshot IS the candidates' predicted action at tick t of the chunk (see
     ``MultiSamplePolicy.infer``'s docstring, in ACRFT, for the per-tick wire format).
  3. Renders EVERY tick of the chunk (not just its start): the joint-target paths run through real
     forward kinematics (``WristCameraGeometry``) once per chunk, and each wrist re-projects them
     at that tick's own pose -- so you watch the arm consume the chunk, then jump to the next.
  4. Also projects each arm's paths into the FIXED agentview, through that arm's calibrated
     ``base_T_agentview`` (config, board-on-gripper solve): agentview does not ride the arm, so
     this needs no joint pose, no arm_offset, and no shared frame -- each arm's chunk lives in its
     own base frame and its own extrinsic places it (left green, right amber).
  5. Draws the overlays on the recorded frames (``_draw_fan``) and writes an mp4.

    # deploy fan on all three cameras (agentview + both wrists):
    workstation/yam-data render-samples \\
        --repo-id my_deploy_run --episode 0 --horizon 30 --candidates 8 \\
        --out .scratch/deploy_samples.mp4
    # any dataset, executed trajectory only (no action_samples needed):
    workstation/yam-data render-samples \\
        --repo-id my_demo --episode 0 --source action --horizon 16 \\
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
    from yam_policy.viz import CameraIntrinsics, WristCameraGeometry

    from workstation.lerobot_recorder.dataset_reader import DatasetReader


def _recorded_chunk_starts(reader: "DatasetReader", episode: int, n_frames: int) -> Optional[List[int]]:
    """The REAL replan boundaries, read from the recording's `policy.chunk_index` column.

    A deploy run records which chunk each executed action came from, so the boundaries do not have
    to be guessed from a horizon the caller types in. That matters for more than convenience:

    * the horizon is adaptive -- a prefix-guided server returns only what is still worth executing,
      so a fixed grid mis-cuts every chunk after the first one whose length differs;
    * an intervention or a homing forces an early replan the grid knows nothing about, and every
      chunk after it is drawn shifted.

    None when the column is absent (any older recording, or a teleop/demo dataset) OR when it
    never changes -- a recording made while the provenance was being written as a constant zero
    has the column but no information in it, and taking it at face value would draw the whole
    episode as one enormous chunk. Either way the caller falls back to tiling by --horizon.
    """
    if not reader.has_feature("policy.chunk_index"):
        return None
    starts, previous = [], None
    for frame in range(n_frames):
        value = reader.get_scalar(episode, frame, "policy.chunk_index")
        if value is None:
            return None
        if previous is None or value != previous:
            starts.append(frame)
            previous = value
    return starts if len(starts) > 1 else None


def _recorded_candidates(reader: "DatasetReader") -> Optional[int]:
    """How many candidates the recording actually carries, from the action_samples feature shape.

    The count is the server's, fixed at handshake and written into the dataset schema; asking the
    caller to repeat it on the command line is one more number to get wrong (and a wrong one is a
    reshape error, not a picture)."""
    shape = reader.feature_shape("action_samples")
    return int(shape[0]) if shape and len(shape) >= 1 else None


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


def _project_wrist(
    geometry: "WristCameraGeometry", base_path: np.ndarray, q_now: np.ndarray, intr: "CameraIntrinsics"
) -> np.ndarray:
    """[H, 3] base-frame path -> [H, 2] pixels in the wrist camera at pose ``q_now``, NaN behind it.

    The wrist camera rides the arm, so its pose is ``WristCameraGeometry.camera_pose(q_now)`` --
    recomputed each tick as the arm executes, while the base-frame path itself is fixed. NaN for
    points behind the lens (same contract as ``_project_fixed`` / the old ``WristProjector``)."""
    proj = geometry.project(base_path, q_now, intr)  # [H, 3]; column 2 is the in-front mask
    px = proj[:, :2].copy()
    px[proj[:, 2] < 0.5] = np.nan
    return px


def _load_agentview_extrinsics(config_path: Optional[str], arms: List[str]) -> dict:
    """{arm: agentview_T_base (4x4)} for each requested arm that has a calibrated extrinsic in
    config.yaml (``cameras.agentview.extrinsic.<arm>.matrix`` is ``base_T_agentview``; inverted
    here to project base-frame points INTO agentview). Arms without one are omitted, with a note --
    the overlay just skips them rather than drawing a fan at a wrong pose."""
    from i2rt.serving.rig_config import find_rig, load_rig

    try:
        rig = load_rig(config_path)
    except Exception as e:  # no config found / unreadable -> no calibrated agentview, render raw
        print(f"agentview: no usable config ({e}) -- agentview shown without an overlay", file=sys.stderr)
        return {}
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


def _to_size(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to ``width x height`` (its native aspect kept when width/height match it).

    hf-utils' dataset render scales every camera to a COMMON HEIGHT and hstacks, so the footage
    keeps its 4:3 shape instead of being squashed square -- we do the same. Scaling the intrinsics
    to the SAME ``(width, height)`` (see ``CameraIntrinsics.scaled_to``, which takes the x and y
    factors separately) makes the two transforms cancel, so a projected point still lands on the
    visible pixel it names even when width/height differ -- projecting at one aspect and drawing at
    another otherwise silently shifts every point."""
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((width, height), Image.LANCZOS))


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
def _chunk_series(blocks: list, n_frames: int) -> np.ndarray:
    """How many steps the reply that owns each frame carried, one value per frame.

    The horizon is adaptive -- a prefix-guided server returns only what is still worth executing,
    and an intervention forces an early replan -- so the length is a signal in its own right, and
    it decides how often the policy is re-queried. Plotted on the same x axis as the value strip,
    the two read together: what the critic was choosing between, and how far ahead it committed.
    """
    lengths = np.zeros(n_frames, dtype=float)
    for start, end in zip(blocks, blocks[1:] + [n_frames], strict=False):
        lengths[start:end] = end - start
    return lengths


def _value_series(reader: "DatasetReader", episode: int, n_frames: int, n_candidates: int) -> "Optional[tuple]":
    """The critic value at every frame: what it gave the candidate it picked, and the spread.

    Read once for the whole episode rather than per chunk -- the curve is the point, and it only
    reads as a decision record when you can see where the current frame sits on it. Returns
    (chosen, low, high) or None when the run recorded no critic.
    """
    if not reader.has_feature("critic_scores"):
        return None
    chosen, low, high = [], [], []
    for frame in range(n_frames):
        scores = reader.get_extra(episode, frame, "critic_scores", (n_candidates,))
        if scores is None:
            return None
        scores = np.asarray(scores, dtype=float)
        pick = reader.get_scalar(episode, frame, "critic_choice")
        chosen.append(float(scores[int(pick or 0)]))
        low.append(float(scores.min()))
        high.append(float(scores.max()))
    return np.array(chosen), np.array(low), np.array(high)


def _value_panel_base(series: tuple, width: int, height: int, *, title: str = "", fmt: str = ".6g") -> tuple:
    """Paint a whole strip once; each frame only adds its cursor to a copy.

    Redrawing the curve per frame would multiply the cost of a 9000-frame render by the length of
    the episode for a picture that never changes.

    ``series`` is ``(line, low, high)``; pass ``low is high is None`` for a plain line with no
    band. Every strip shares the x axis (frame index) with the cameras above, so stacked strips
    read against each other and against the footage.
    """
    from PIL import Image, ImageDraw

    chosen, low, high = series
    banded = low is not None and high is not None
    if not banded:
        low = high = chosen
    img = Image.new("RGB", (width, height), (13, 17, 23))
    d = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 52, 10, 34, 26
    plot = (pad_l, pad_t, width - pad_r, height - pad_b)
    w, h = plot[2] - plot[0], plot[3] - plot[1]
    lo, hi = float(min(low.min(), chosen.min())), float(max(high.max(), chosen.max()))
    span = (hi - lo) or 1.0
    n = len(chosen)

    def xy(i: int, v: float) -> tuple:
        return (plot[0] + w * i / max(n - 1, 1), plot[3] - h * (v - lo) / span)

    # The spread across candidates: how much the critic thought was at stake at each decision.
    if banded:
        band = [xy(i, high[i]) for i in range(n)] + [xy(i, low[i]) for i in range(n - 1, -1, -1)]
        d.polygon(band, fill=(35, 55, 95))
    d.line([xy(i, chosen[i]) for i in range(n)], fill=(240, 210, 90), width=2)
    d.line([plot[0], plot[3], plot[2], plot[3]], fill=(48, 54, 61))
    f = _font(12)
    d.text((6, plot[1] - 6), f"{hi:.4g}", font=f, fill=(139, 148, 158))
    d.text((6, plot[3] - 6), f"{lo:.4g}", font=f, fill=(139, 148, 158))
    d.text((pad_l, 4), title, font=f, fill=(139, 148, 158))
    return img, plot, lo, span, fmt


def _value_panel(base: tuple, frame: int, n_frames: int, chosen_value: float) -> np.ndarray:
    """The precomputed strip plus a cursor at this frame."""
    from PIL import ImageDraw

    img, plot, lo, span, fmt = base
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = plot[2] - plot[0], plot[3] - plot[1]
    x = plot[0] + w * frame / max(n_frames - 1, 1)
    y = plot[3] - h * (chosen_value - lo) / span
    d.line([x, plot[1], x, plot[3]], fill=(90, 100, 115), width=1)
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(240, 210, 90))
    d.text((plot[0] + 4, plot[1] + 2), format(chosen_value, fmt), font=_font(12), fill=(240, 210, 90))
    return np.asarray(out)


def _load_critic(
    reader: "DatasetReader", episode: int, start: int, n_candidates: int
) -> "tuple[Optional[np.ndarray], int]":
    """This replan's critic scores and the candidate it picked, or (None, 0).

    A value-guided server records a score per candidate and the index it executed
    (`critic_scores` / `critic_choice`, declared at handshake). With them the fan stops being a
    spread of equally plausible options and becomes the value landscape the decision was made on:
    which paths the critic liked, which it rejected, and how far apart it thought they were.

    Read at the chunk's first frame: both columns are per step but constant across a replan (the
    decision is made once, then executed).
    """
    scores = reader.get_extra(episode, start, "critic_scores", (n_candidates,))
    if scores is None:
        return None, 0
    choice = reader.get_scalar(episode, start, "critic_choice")
    return np.asarray(scores, dtype=float), int(choice or 0)


def _value_color(score: float, low: float, high: float) -> tuple:
    """Rank a candidate's value on a cold-to-warm ramp.

    Normalised against THIS replan's own spread rather than a global scale: the absolute values a
    critic emits are arbitrary (cost-to-goal here runs to -2777), while the useful question at
    each decision is which of these candidates the critic preferred over the others.
    """
    t = 0.5 if high <= low else float(np.clip((score - low) / (high - low), 0.0, 1.0))
    cold, warm = (70, 105, 190), (240, 210, 90)  # rejected -> preferred
    return tuple(int(a + (b - a) * t) for a, b in zip(cold, warm, strict=True))


_FAN_COLORS = {
    "left": ((20, 140, 95), (190, 255, 225)),
    "right": ((170, 95, 20), (255, 205, 140)),
}


def _draw_fan(
    wrist_rgb: np.ndarray,
    paths: list,
    chosen: int = 0,
    *,
    colors: tuple = _FAN_COLORS["left"],
    scores: Optional[np.ndarray] = None,
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
        # Uniform blue when nothing scored them; otherwise the critic's own ranking (see
        # _value_color), so a rejected path and a near-miss do not look alike.
        if scores is not None and i < len(scores):
            col = _value_color(float(scores[i]), float(np.min(scores)), float(np.max(scores)))
        else:
            col = (95, 165, 235)
        d.line(pts, fill=(*col, 155), width=3)
        d.ellipse([pts[-1][0] - 2.5, pts[-1][1] - 2.5, pts[-1][0] + 2.5, pts[-1][1] + 2.5], fill=(*col, 200))

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
    panels: list, panel_w: int, panel_h: int, *, header: str, below: "np.ndarray | None" = None
) -> np.ndarray:
    """``[(image, label), ...]`` -> one hstacked-flush frame, hf-utils' dataset-render look: each
    camera scaled to a common height and butted together, a translucent-box header top-left and a
    per-panel camera label bottom-left (``drawtext ... box=1:boxcolor=black@0.5`` in hf-utils).

    Variable panel count so a run can show agentview + one wrist, or agentview + BOTH wrists, or
    any subset -- see ``--agentview-arms``/``--wrists``.

    ``below`` is stacked UNDER the cameras at the full frame width -- that is where an analytics
    strip belongs. Beside them it would be a fourth camera-sized column, making an already wide
    frame wider for a plot that reads better long and short.
    """
    from PIL import Image

    strip = np.concatenate([img for img, _ in panels], axis=1)
    canvas = Image.fromarray(strip if below is None else np.concatenate([strip, below], axis=0))
    f_sm, f_md = _font(14), _font(16)
    _label(canvas, (8, 8), header, f_md)
    for i, (_, label) in enumerate(panels):
        _label(canvas, (i * panel_w + 8, panel_h - 26), label, f_sm)
    return np.asarray(canvas)


def render(args: argparse.Namespace) -> pathlib.Path:
    from yam_policy.viz import CameraIntrinsics, WristCameraGeometry

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    from workstation.lerobot_recorder.dataset_reader import DatasetReader

    reader = DatasetReader(args.repo_id, args.root)
    reader.load()
    n_frames = reader.episode_length(args.episode)
    if not n_frames:
        raise SystemExit(f"episode {args.episode} not found (or empty) in {args.repo_id}")

    # Each panel keeps the footage's 4:3 (hf-utils scales cameras to a common height, not square --
    # see _to_size). Native-resolution intrinsics are rescaled to that panel size so a projected
    # point still lands on the visible pixel it names.
    panel_h = args.height
    panel_w = round(panel_h * 640 / 480)
    wrist_intr = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, width=640, height=480).scaled_to(
        panel_w, panel_h
    )
    agent_intr = CameraIntrinsics(
        fx=args.agent_fx, fy=args.agent_fy, cx=args.agent_cx, cy=args.agent_cy, width=640, height=480
    ).scaled_to(panel_w, panel_h)
    # Which 7 of the 14 recorded action dims are each arm's (see workstation.lerobot_recorder
    # .config: joints 0..6 left, 7..13 right), and where in `observation.state`'s 42-d layout
    # each arm's joint POSITIONS sit (pos block only -- vel/eff do not describe the pose).
    action_slices = {"left": slice(0, 7), "right": slice(7, 14)}
    state_slices = {"left": slice(0, 7), "right": slice(21, 28)}

    # One FK model per arm, built with THAT arm's calibrated wrist extrinsic when config has it
    # (gripper_T_camera; falls back to the CAD default). FK itself is arm-independent, so any of
    # these also serves the agentview projection (chunk_to_path is extrinsic-free).
    # The wrist camera mount is the CAD T_GRIPPER_CAMERA (WristCameraGeometry's default) -- the
    # rig uses CAD for the wrists, so there is no per-camera extrinsic to load from config.
    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    geometries = {arm: WristCameraGeometry(xml) for arm in ("left", "right")}

    # Agentview: a FIXED third-person camera, so each arm's overlay projects through its calibrated
    # base_T_agentview (config, board-on-gripper solve) -- no wrist ride, no arm_offset, no shared
    # frame needed. Each wrist camera rides its own arm and shows ONLY that arm's overlay (the other
    # arm's path would need arm_offset to place, which this deliberately avoids).
    agentview_arms = list(dict.fromkeys(args.agentview_arms))
    agent_extrinsics = _load_agentview_extrinsics(args.config, agentview_arms)
    wrists = list(dict.fromkeys(args.wrists))

    # Prefer the boundaries the run actually recorded; they are exact, they follow an adaptive
    # horizon, and they survive an intervention's early replan. Tiling by --horizon is the fallback
    # for recordings without the column (see _recorded_chunk_starts).
    horizon = args.horizon or int(reader.fps or 30)
    recorded = None if args.horizon else _recorded_chunk_starts(reader, args.episode, n_frames)
    if recorded:
        blocks = recorded
        lengths = [b - a for a, b in zip(blocks, blocks[1:] + [n_frames], strict=False)]
        print(
            f"chunk boundaries from the recording: {len(blocks)} replans, "
            f"length min/med/max = {min(lengths)}/{sorted(lengths)[len(lengths) // 2]}/{max(lengths)}",
            file=sys.stderr,
        )
    else:
        # Chunks tile the episode on a fixed grid of `horizon` ticks (the last one may be shorter,
        # so the WHOLE trajectory is covered, tail included -- not just the first n//H*H frames).
        blocks = list(range(0, n_frames, horizon))
    all_blocks = list(blocks)
    if args.replans:
        blocks = blocks[: args.replans]
    if not blocks:
        raise SystemExit(f"episode {args.episode} has no frames")

    # "samples" = the multi-candidate deploy fan (action_samples column); "action" = the single
    # executed trajectory from the plain action column, so it works on ANY LeRobot recording.
    if args.source == "samples" and args.candidates is None:
        args.candidates = _recorded_candidates(reader)
        if args.candidates is None:
            raise SystemExit(
                "this dataset declares no action_samples column, so there is no fan to draw. "
                "Record against a server started with --num-samples N (or a critic), or pass "
                "--source action to draw the executed trajectory instead."
            )
        print(f"candidates from the recording: {args.candidates}", file=sys.stderr)

    # The critic's value over the whole episode, if it recorded one: drawn as its own panel so the
    # decision is legible as a time series, not just as colour on the fan.
    value_series = None
    if args.source == "samples" and not args.no_value_plot:
        value_series = _value_series(reader, args.episode, n_frames, args.candidates)
        if value_series is not None:
            print(f"critic value curve: {n_frames} frames", file=sys.stderr)

    fan_word = "fan" if args.source == "samples" else "path"

    # Sized when the first frame is composed: the strip width depends on how many camera panels
    # actually render (a wrist can be missing from a recording).
    value_base = None
    # One value per frame: the length of the reply that frame came from (see _chunk_series). Only
    # when the boundaries are the RUN's own -- tiling by --horizon would plot the number the caller
    # typed in, held flat across the episode, which says nothing about the policy.
    chunk_lengths = _chunk_series(all_blocks, n_frames) if (recorded and not args.no_chunk_plot) else None
    chunk_base = None

    frames = []
    for block_idx, start in enumerate(blocks):
        # Each block runs to the next boundary (or the end), so a recorded chunk is drawn whole
        # whatever its length.
        block_end = blocks[block_idx + 1] if block_idx + 1 < len(blocks) else n_frames
        block_len = (block_end - start) if recorded else min(horizon, n_frames - start)
        if args.source == "samples":
            chunk = _load_replan_chunk(reader, args.episode, start, block_len, args.candidates)
            missing = "no action_samples recorded here"
        else:
            chunk = _load_action_chunk(reader, args.episode, start, block_len)
            missing = "no action column here"
        if chunk is None:
            print(f"chunk at frame {start}: {missing}, skipping", file=sys.stderr)
            continue

        # What the critic thought of these candidates, if the run recorded it. `chosen` is the
        # candidate actually executed -- candidate 0 for a plain multi-sample reply, but a critic
        # picks by value, and drawing 0 as the chosen one would then highlight the wrong path.
        scores, chosen = (None, 0)
        if args.source == "samples":
            scores, chosen = _load_critic(reader, args.episode, start, args.candidates)

        # The chunk's base-frame paths (FK of the joint targets) are the SAME for every tick in the
        # block -- only the wrist CAMERA moves as the arm executes -- so compute them once here and
        # only re-project per tick below. This is what makes rendering every frame cheap.
        base_paths = {
            arm: [geometries[arm].chunk_to_path(c[:, action_slices[arm]]) for c in chunk] for arm in ("left", "right")
        }
        agent_paths = {
            arm: [_project_fixed(bp, agent_extrinsics[arm], agent_intr) for bp in base_paths[arm]]
            for arm in agent_extrinsics
        }

        # Render EVERY tick in the block (not just its start): you watch the arm consume the chunk,
        # then it jumps to the next chunk. The overlay (the plan) holds across the block; the
        # footage advances and each wrist re-projects at that tick's pose.
        for offset in range(block_len):
            frame_idx = start + offset
            state = reader.get_state(args.episode, frame_idx)
            images = reader.get_images(args.episode, frame_idx)
            if state is None or not images:
                print(f"frame {frame_idx}: no state/images, skipping", file=sys.stderr)
                continue

            # left-to-right: wrist left, agentview, wrist right -- the scene camera in the middle,
            # each wrist on the side of the arm it rides.
            panels, agent_panel = [], None
            agent = images.get("agentview")
            if agent is not None and agentview_arms:
                agent_sq = _to_size(agent, panel_w, panel_h)
                for arm, apaths in agent_paths.items():
                    agent_sq = _draw_fan(agent_sq, apaths, chosen=chosen, colors=_FAN_COLORS[arm], scores=scores)
                label = (
                    f"agentview -- {'+'.join(agent_paths)} {fan_word}" if agent_paths else "agentview (no extrinsic)"
                )
                agent_panel = (agent_sq, label)

            for arm in wrists:
                wimg = images.get(f"wrist_{arm}")
                if wimg is None:
                    continue
                q_now = state[state_slices[arm]]
                wpaths = [_project_wrist(geometries[arm], bp, q_now, wrist_intr) for bp in base_paths[arm]]
                wrist_sq = _draw_fan(
                    _to_size(wimg, panel_w, panel_h), wpaths, chosen=chosen, colors=_FAN_COLORS[arm], scores=scores
                )
                panels.append((wrist_sq, f"wrist {arm} -- {fan_word}"))
                if arm == "left" and agent_panel is not None:
                    panels.append(agent_panel)  # agentview sits between the two wrists
                    agent_panel = None
            if agent_panel is not None:
                panels.append(agent_panel)  # no left wrist rendered: agentview still comes first

            if not panels:
                continue
            below = None
            if value_series is not None:
                if value_base is None:
                    # A wide, short strip under the cameras: a value curve reads along time, so it
                    # wants width, and giving it a camera's height would make the frame square.
                    # Even height: h264 with yuv420p subsamples chroma 2x2 and refuses an odd
                    # frame dimension, and this strip decides the frame's height together with the
                    # cameras (360 + 151 = 511 killed the encode).
                    value_base = _value_panel_base(
                        value_series,
                        panel_w * len(panels),
                        2 * round(panel_h * 0.42 / 2),
                        title="critic value -- picked (line) vs candidate spread (band)",
                    )
                below = _value_panel(value_base, frame_idx, n_frames, float(value_series[0][frame_idx]))
            if chunk_lengths is not None:
                if chunk_base is None:
                    chunk_base = _value_panel_base(
                        (chunk_lengths, None, None),
                        panel_w * len(panels),
                        2 * round(panel_h * 0.26 / 2),
                        title="chunk length -- steps the reply carried",
                        fmt=".0f",
                    )
                strip = _value_panel(chunk_base, frame_idx, n_frames, float(chunk_lengths[frame_idx]))
                below = strip if below is None else np.concatenate([below, strip], axis=0)

            header = (
                f"{args.repo_id} · ep {args.episode} · chunk {block_idx + 1}/{len(blocks)}  "
                f"tick {offset + 1}/{block_len}"
            )
            if scores is not None:
                # The decision, in the numbers it was made on: which candidate won, its value, and
                # how much better the critic thought it was than the worst on offer. A near-zero
                # spread means the critic saw nothing to choose between -- worth seeing.
                header += (
                    f"  ·  critic: #{chosen} of {len(scores)}  "
                    f"Q {float(scores[chosen]):.6g}  "
                    f"(best by {float(scores[chosen] - np.median(scores)):+.3g}, "
                    f"spread {float(np.max(scores) - np.min(scores)):.3g})"
                )
            composed = _compose_frame(panels, panel_w, panel_h, header=header, below=below)
            for _ in range(max(1, args.hold)):
                frames.append(composed)

    if not frames:
        raise SystemExit("nothing to render -- no frame had both action/action_samples and images")

    import imageio

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Browser-friendly h264: yuv420p is what every browser can decode (hf-utils encodes the same
    # way for its dataset previews); macro_block_size=1 keeps odd panel widths from being padded.
    imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1, codec="libx264", pixelformat="yuv420p")
    print(f"wrote {out} ({len(frames)} frames over {len(blocks)} chunk(s))")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="dataset repo id, as recorded (e.g. user/my_deploy_run)")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--config", default=None, help="config.yaml for agentview extrinsics; auto-discovered by default")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--wrists",
        nargs="*",
        choices=["left", "right"],
        default=["left", "right"],
        help="which wrist camera panels to show (default both); each overlays only its own arm's path",
    )
    p.add_argument(
        "--agentview-arms",
        dest="agentview_arms",
        nargs="*",
        choices=["left", "right"],
        default=["left", "right"],
        help="which arms' paths to overlay on the agentview panel (default both; each uses its own "
        "calibrated base_T_agentview -- arms without one are skipped). Pass nothing to drop agentview.",
    )
    p.add_argument(
        "--source",
        choices=["samples", "action"],
        default="samples",
        help="'samples' overlays the multi-candidate deploy fan (needs an action_samples column); "
        "'action' overlays the single executed trajectory from the plain action column, so it works "
        "on ANY LeRobot recording (teleop/demo/replay), not just a --num-samples deploy run",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="chunk size: how many future ticks to draw (default: the dataset fps = 1 second; "
        "for a deploy fan pass the server's action_horizon). Omit it and the REAL replan "
        "boundaries are read from the recording's policy.chunk_index when it has one",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="candidates per step; read from the dataset's action_samples column by default, "
        "so pass it only to override (ignored for --source action)",
    )
    p.add_argument(
        "--no-value-plot",
        action="store_true",
        help="omit the critic-value panel (drawn automatically when the recording has critic_scores)",
    )
    p.add_argument(
        "--no-chunk-plot",
        action="store_true",
        help="omit the chunk-length strip (drawn under the cameras, sharing their time axis)",
    )
    p.add_argument("--replans", type=int, default=0, help="max replans to render (0 = the whole episode)")
    p.add_argument("--hold", type=int, default=1, help="repeat each rendered tick N times (>1 = slow motion)")
    p.add_argument("--height", type=int, default=360, help="per-panel height in px (width follows the 4:3 footage)")
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
    # --candidates is filled in from the recording when it can be (see _recorded_candidates);
    # only a dataset whose schema does not declare action_samples still needs it spelled out.
    render(args)


if __name__ == "__main__":
    main()
