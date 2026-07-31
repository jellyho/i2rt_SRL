"""Image composition helpers for the operator views (pure numpy, no cv2).

* :func:`compose_camera_strip` — left wrist, agentview, right wrist side-by-side
  so the teleoperator can see every camera without overlap.
* :func:`compose_agentview` — agentview as the base with the wrist views inset in
  the bottom corners, kept for any callers that still want the compact view.
* :func:`overlay` — alpha-blend two frames (e.g. an episode's first frame with the
  live camera) so the operator can match object placement before a replay.
* :func:`overlay_camera_views` — blend synchronized past-episode views into the
  matching live camera views without modifying the raw recorder images.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbor resize to (h, w) — dependency-free, fine for previews."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    yi = np.linspace(0, img.shape[0] - 1, h).astype(int)
    xi = np.linspace(0, img.shape[1] - 1, w).astype(int)
    return img[yi][:, xi]


def compose_camera_strip(
    images: Dict[str, np.ndarray],
    agent_key: str = "agentview",
    order: Tuple[str, ...] = ("wrist_left", "agentview", "wrist_right"),
) -> Optional[np.ndarray]:
    """Return a side-by-side live view ordered left camera, middle camera, right camera."""
    ordered_keys = tuple(agent_key if key == "agentview" else key for key in order)
    frames = [np.ascontiguousarray(images[key]) for key in ordered_keys if key in images]
    if not frames:
        frame = next(iter(images.values()), None)
        return None if frame is None else np.ascontiguousarray(frame)

    target_h = max(frame.shape[0] for frame in frames)
    resized = [
        _resize(frame, target_h, max(round(frame.shape[1] * target_h / frame.shape[0]), 1))
        for frame in frames
    ]
    return np.ascontiguousarray(np.hstack(resized))


def compose_agentview(
    images: Dict[str, np.ndarray],
    agent_key: str = "agentview",
    wrist_keys: Tuple[str, ...] = ("wrist_left", "wrist_right"),
    inset_frac: float = 0.33,
) -> Optional[np.ndarray]:
    """Agentview base with each wrist view inset into a bottom corner."""
    base = images.get(agent_key)
    if base is None:
        base = next(iter(images.values()), None)
    if base is None:
        return None
    out = np.ascontiguousarray(base).copy()
    h, w = out.shape[:2]
    ih, iw = max(int(h * inset_frac), 1), max(int(w * inset_frac), 1)
    corners = [(h - ih, 0), (h - ih, w - iw)]  # bottom-left, bottom-right
    for key, (y0, x0) in zip(wrist_keys, corners, strict=False):
        wim = images.get(key)
        if wim is None:
            continue
        small = _resize(np.ascontiguousarray(wim), ih, iw)
        out[y0 : y0 + ih, x0 : x0 + iw] = small
        out[y0 : y0 + ih, x0 : x0 + 2] = 255  # thin border so the inset reads clearly
        out[y0 : y0 + 2, x0 : x0 + iw] = 255
    return out


def overlay(base: Optional[np.ndarray], other: Optional[np.ndarray], alpha: float = 0.5) -> Optional[np.ndarray]:
    """Alpha-blend ``other`` (resized to ``base``) over ``base``."""
    if base is None:
        return other
    if other is None:
        return base
    o = _resize(np.ascontiguousarray(other), base.shape[0], base.shape[1])
    return (base.astype(np.float32) * (1.0 - alpha) + o.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def overlay_camera_views(
    live: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    live_alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Blend each live view over its matching reference view.

    ``live_alpha=1`` is an unchanged live preview and ``live_alpha=0`` is the
    past episode only.  A new mapping and new blended arrays are returned; this
    is important because the original frames are also consumed by the dataset
    writer and deployment policy.
    """

    alpha = min(max(float(live_alpha), 0.0), 1.0)
    out = dict(live)
    for key in live.keys() & reference.keys():
        blended = overlay(reference[key], live[key], alpha=alpha)
        if blended is not None:
            out[key] = blended
    return out
