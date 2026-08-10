"""Render a recorded episode with the policy's sampled chunks drawn on the wrist views.

The live overlay shows the spread as it happens, which is the right way to catch a policy
hesitating in the moment. This is the other half: replay an episode afterwards, ask the policy
for N chunks at each frame, and write a video. Nobody has to be standing at the robot, the same
episode can be re-rendered against a different checkpoint, and a moment worth arguing about can
be paused on.

The drawing is :class:`WristOverlayRenderer`, the same object the live view uses — writing it
twice is how two surfaces quietly stop agreeing about what the policy predicted.

    workstation/yam-data render-samples --dataset yam_lego_taxi --episode 3 --num-samples 8

Frames come from the dataset, and the policy is asked fresh at each one, so this shows what the
CURRENT checkpoint would have done at each moment the robot was in — not what it did at the
time. That is usually the interesting comparison, but it is not a replay of the original run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WRIST_KEYS = ("wrist_left", "wrist_right")


def _episode_frames(dataset_dir: str, episode: int) -> List[dict]:
    """Load one episode as a list of ``{"state": [42], "images": {key: HxWx3}}``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/render", root=dataset_dir)
    episodes = ds.meta.episodes
    row = episodes[episode] if hasattr(episodes, "__getitem__") else None
    if row is None:
        raise ValueError(f"episode {episode} not found in {dataset_dir}")

    def _scalar(v):
        return v[0] if isinstance(v, (list, tuple, np.ndarray)) else v

    start = int(_scalar(row["dataset_from_index"]))
    end = int(_scalar(row["dataset_to_index"]))

    out = []
    for i in range(start, end):
        item = ds[i]
        images = {}
        for key in WRIST_KEYS:
            frame = item.get(f"observation.images.{key}")
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):     # CHW float -> HWC uint8
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
            images[key] = arr
        out.append({"state": np.asarray(item["observation.state"], dtype=float), "images": images})
    return out


def render(
    dataset_dir: str,
    episode: int,
    out_path: str,
    *,
    policy_host: str,
    policy_port: int,
    num_samples: int,
    prompt: str,
    fps: int,
    intrinsics_by_key: Optional[dict] = None,
    limit: Optional[int] = None,
    replay_policy: bool = False,
) -> str:
    from workstation.policy_bridge.chunk_overlay import WristOverlayRenderer
    from workstation.policy_bridge.wrist_view import CameraIntrinsics

    frames = _episode_frames(dataset_dir, episode)
    if limit:
        frames = frames[:limit]
    if not frames:
        raise ValueError("episode has no frames")

    renderer = WristOverlayRenderer()
    if not renderer.available:
        raise RuntimeError(f"wrist geometry unavailable: {renderer.error}")

    # Prefer what the policy actually predicted at the time, if the run recorded it. Asking a
    # server instead answers a different question -- what the CURRENT checkpoint would do at
    # each moment -- which is useful, but is not what happened.
    from workstation.policy_bridge import sample_log as _sample_log

    recorded = None if replay_policy else _sample_log.load(dataset_dir, episode)
    if recorded is not None:
        logger.info("using the %d sample set(s) recorded with the episode", len(recorded["frame_index"]))
        rendered = []
        from workstation.lerobot_recorder.views import compose_camera_strip

        for index, frame in enumerate(frames):
            samples = _sample_log.samples_at(recorded, index)
            decorated = renderer.draw(frame["images"], samples, frame["state"], _intrinsics_fallback(frames, intrinsics_by_key))
            rendered.append(compose_camera_strip(decorated))
        return _write(rendered, out_path, fps)

    from yam_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=policy_host, port=policy_port)
    meta = client.get_server_metadata() or {}
    if not meta.get("supports_multi_sample"):
        # Better to say so than to render a video of single lines and let someone conclude the
        # policy is extremely confident.
        raise RuntimeError(
            f"policy server at {policy_host}:{policy_port} does not support multi-sample "
            "(no `supports_multi_sample` in its metadata) — update it before rendering"
        )

    intrinsics_for = _intrinsics_fallback(frames, intrinsics_by_key)

    from workstation.lerobot_recorder.views import compose_camera_strip

    rendered = []
    for index, frame in enumerate(frames):
        obs = {"observation/state": frame["state"], "prompt": prompt, "num_samples": int(num_samples)}
        for key, image in frame["images"].items():
            obs[f"observation/{key}"] = image
        try:
            samples = client.infer(obs).get("action_samples")
        except Exception as e:
            logger.error("inference failed at frame %d: %s", index, e)
            samples = None
        decorated = renderer.draw(frame["images"], samples, frame["state"], intrinsics_for)
        rendered.append(compose_camera_strip(decorated))
        if index % 30 == 0:
            logger.info("rendered %d/%d frames", index, len(frames))

    return _write(rendered, out_path, fps)


def _write(rendered: List[np.ndarray], out_path: str, fps: int) -> str:
    import imageio.v3 as iio

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    iio.imwrite(out_path, np.stack(rendered), fps=fps)
    logger.info("wrote %s (%d frames)", out_path, len(rendered))
    return out_path


def _intrinsics_fallback(frames: List[dict], intrinsics_by_key: Optional[dict]):
    from workstation.policy_bridge.wrist_view import CameraIntrinsics

    def intrinsics_for(key: str):
        if intrinsics_by_key and key in intrinsics_by_key:
            return intrinsics_by_key[key]
        sample = frames[0]["images"].get(key)
        if sample is None:
            return None
        # No stored intrinsics: fall back to the D405's nominal figures for this frame size.
        # Approximate, and said out loud, because a silently-wrong focal length makes the paths
        # wrong in a way that still looks plausible.
        height, width = sample.shape[:2]
        logger.warning("no intrinsics for %s; assuming a nominal D405 at %dx%d", key, width, height)
        return CameraIntrinsics(fx=0.9 * width, fy=0.9 * width, cx=width / 2, cy=height / 2,
                                width=width, height=height)

    return intrinsics_for


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=None, help="dataset folder under the recorder root")
    p.add_argument("--root", default=None)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", default=None, help="output mp4 (default: <dataset>_renders/samples-ep<N>.mp4)")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--policy-host", default=None)
    p.add_argument("--policy-port", type=int, default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="stop after this many frames (0 = whole episode)")
    p.add_argument("--replay-policy", action="store_true",
                   help="ignore the samples recorded with the episode and ask a live policy "
                        "server instead — what the CURRENT checkpoint would do, not what happened")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from i2rt.serving.rig_config import Resolver, load_rig
    from workstation.lerobot_recorder.dataset_writer import dataset_dir

    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    pol = Resolver(args, p, rig.get("policy", {}))
    root = args.root or rec.get("root")
    name = (args.dataset or rec.get("repo_id")).strip("/").split("/")[-1]
    ds_dir = dataset_dir(root, name)

    out = args.out or os.path.join(f"{ds_dir}_renders", f"samples-ep{args.episode}.mp4")
    try:
        render(
            ds_dir, args.episode, out,
            policy_host=args.policy_host or pol.get("policy_host", key="host"),
            policy_port=int(args.policy_port or pol.get("policy_port", key="port")),
            num_samples=args.num_samples,
            prompt=args.prompt or rec.get("task"),
            fps=args.fps,
            limit=args.limit or None,
            replay_policy=args.replay_policy,
        )
    except Exception as e:
        print(f"render failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
