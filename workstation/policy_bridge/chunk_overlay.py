"""Draw sampled action chunks onto a wrist-camera frame.

One picture answers a question no metric does: is the policy sure? Several chunks drawn
together read immediately — a tight bundle is confidence, a wide spray is a policy torn
between options, and a single line wandering off is a policy that has lost the plot.

The paths come from :mod:`workstation.policy_bridge.wrist_view`, which turns joint targets into
a metric path via FK and projects it through the published D405 extrinsic. So unlike ACRFT's
RoboCasa overlay — which cannot know how far an unnormalised OSC delta reaches, and rescales
each replan to a legible length — the spread you see here is the spread in metres.

The same function serves the live view and a post-hoc render; only where the frame comes from
differs. Drawing twice, once per surface, is how the two quietly stop agreeing.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# The chunk that will actually run, vs the ones that were merely possible. The executed path is
# opaque and warm so it reads first; candidates are translucent and cool so a dozen of them
# still show their envelope instead of turning into a solid block.
EXECUTED_RGBA = (255, 210, 30, 255)
CANDIDATE_RGBA = (80, 150, 255, 130)
CANDIDATE_DOT_RGBA = (80, 150, 255, 200)


def draw_chunk_paths(
    frame: np.ndarray,
    paths_px: Sequence[np.ndarray],
    *,
    executed_index: int = 0,
    line_width: int = 2,
) -> np.ndarray:
    """Draw already-projected paths on ``frame`` (HxWx3 uint8) and return a new frame.

    ``paths_px`` is one ``[T, 3]`` array per chunk: columns are ``(u, v, in_front)`` as
    :meth:`WristCameraGeometry.project` returns them. Points that are not in front of the lens
    are dropped rather than drawn: the perspective divide yields a perfectly finite pixel for a
    point behind the camera, and the mirrored path that results looks like a confident wrong
    prediction rather than a bug.
    """
    from PIL import Image, ImageDraw

    base = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for index, path in enumerate(paths_px):
        runs = _visible_runs(path)
        if not runs:
            continue
        executed = index == executed_index
        colour = EXECUTED_RGBA if executed else CANDIDATE_RGBA
        width = line_width + 1 if executed else line_width
        for run in runs:
            if len(run) >= 2:
                draw.line(run, fill=colour, width=width)
        # A dot on the end, so a short path still shows where it was going. On the last
        # visible point, not the last point: a path that leaves the frame has no meaningful
        # endpoint on screen, and drawing one at a clipped coordinate invents a destination.
        x, y = runs[-1][-1]
        r = 4 if executed else 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour if executed else CANDIDATE_DOT_RGBA)

    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def _visible_runs(path: np.ndarray) -> list:
    """Split a projected path into runs of consecutive in-front points.

    Filtering the hidden points out instead would join the two points either side of the gap,
    drawing a straight line the gripper never travels — and a fabricated segment is worse than
    a missing one, because it is indistinguishable from a real prediction.

    Out-of-frame points are kept: PIL clips them, so a path that leaves the view and comes back
    draws the parts you can see and nothing in between, which is exactly right.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError(f"expected [T, 2|3] projected points, got shape {path.shape}")
    in_front = path[:, 2] > 0.5 if path.shape[1] >= 3 else np.ones(len(path), bool)

    runs, current = [], []
    for point, visible in zip(path[:, :2], in_front):
        if visible:
            current.append((float(point[0]), float(point[1])))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def project_samples(
    geometry,
    samples: np.ndarray,
    q_now: Sequence[float],
    intrinsics,
    *,
    arm_slice: Optional[slice] = None,
) -> list:
    """``[N, T, action_dim]`` chunks -> one ``[T, 3]`` projected path per chunk.

    ``arm_slice`` picks this arm's joints out of a bimanual action vector; the default takes the
    whole width, which is right when the caller has already split it. The projection uses the
    arm's CURRENT pose, because the camera rides on the wrist — the paths show where the gripper
    would go as seen from where the camera is right now, which is what the operator is looking
    at.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3:
        raise ValueError(f"expected [N, T, dim] samples, got shape {samples.shape}")
    if arm_slice is not None:
        samples = samples[:, :, arm_slice]

    out = []
    for chunk in samples:
        path_base = geometry.chunk_to_path(chunk)
        out.append(geometry.project(path_base, q_now, intrinsics))
    return out


def overlay_samples(
    frame: np.ndarray,
    geometry,
    samples: np.ndarray,
    q_now: Sequence[float],
    intrinsics,
    *,
    arm_slice: Optional[slice] = None,
    executed_index: int = 0,
) -> np.ndarray:
    """Project ``samples`` and draw them on ``frame``. The whole overlay in one call.

    Intrinsics are rescaled to the frame that was actually passed in. Frames get resized on the
    way to a viewer, and projecting with the sensor's numbers onto a resized image pushes every
    point away from the centre by exactly the resize factor — an error that looks like a
    plausible-but-wrong prediction rather than a mistake.
    """
    height, width = frame.shape[:2]
    if (width, height) != (intrinsics.width, intrinsics.height):
        intrinsics = intrinsics.scaled_to(width, height)
    paths = project_samples(geometry, samples, q_now, intrinsics, arm_slice=arm_slice)
    return draw_chunk_paths(frame, paths, executed_index=executed_index)
