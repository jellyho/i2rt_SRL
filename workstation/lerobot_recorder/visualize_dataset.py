"""Render a fast-forward MP4 of a recorded episode's camera views side by side.

Concatenates the three cameras (wrist_left | agentview | wrist_right, fixed order)
of a LeRobot v3.0 dataset written by ``yam-data record`` into a single MP4, at a
chosen fast-forward ratio.

All episodes of a camera share one MP4; per-episode trim windows come from
``meta/episodes/**.parquet`` (``videos/<key>/{from,to}_timestamp``). We hand those
windows to ffmpeg (input-level ``-ss/-to``), draw labels + a counter, ``hstack`` the
panels, and ``setpts`` for speed -- no torch/lerobot decode needed.

Run from the repo root with uv (resolves the env; base conda lacks pyarrow)::

    uv run python workstation/lerobot_recorder/visualize_dataset.py \
        --dataset yam_lego_yellow_taxi --speed 4 --episode 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fixed panel order, left to right.
CAMERAS = ["wrist_left", "agentview", "wrist_right"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# One episode's per-camera source: (mp4_path, from_timestamp, to_timestamp).
Windows = Dict[str, Tuple[Path, float, float]]


def find_root(cli_root: Optional[str]) -> Path:
    """Dataset root: CLI value, else ``recorder.root`` from config.yaml, else default."""
    if cli_root:
        return Path(os.path.expanduser(cli_root))
    cfg = Path(__file__).resolve().parents[2] / "config.yaml"
    if cfg.exists():
        # Cheap targeted parse -- avoids a hard yaml dependency for one field.
        in_recorder = False
        for line in cfg.read_text().splitlines():
            if re.match(r"^\w", line):
                in_recorder = line.startswith("recorder:")
            m = re.match(r"\s+root:\s*(\S+)", line)
            if in_recorder and m:
                return Path(os.path.expanduser(m.group(1)))
    return Path(os.path.expanduser("~/lerobot_data"))


def load_dataset(dataset_dir: Path) -> Tuple[int, str, Dict[int, Windows]]:
    """Return (fps, video_path_template, {episode_index: Windows}) for a dataset dir."""
    import pyarrow.parquet as pq

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 30))
    video_path = info["video_path"]  # e.g. videos/{video_key}/chunk-{chunk_index:03d}/...

    ep_files = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not ep_files:
        raise FileNotFoundError(f"no episode metadata under {dataset_dir/'meta'/'episodes'}")

    episodes: Dict[int, Windows] = {}
    for f in ep_files:
        table = pq.read_table(f)
        cols = set(table.column_names)
        rows = table.to_pylist()
        for row in rows:
            ep = int(row["episode_index"])
            win: Windows = {}
            for cam in CAMERAS:
                key = f"observation.images.{cam}"
                ci = f"videos/{key}/chunk_index"
                fi = f"videos/{key}/file_index"
                ft = f"videos/{key}/from_timestamp"
                tt = f"videos/{key}/to_timestamp"
                if not {ci, fi, ft, tt} <= cols:
                    continue  # camera not present in this dataset
                mp4 = dataset_dir / video_path.format(
                    video_key=key, chunk_index=int(row[ci]), file_index=int(row[fi])
                )
                win[cam] = (mp4, float(row[ft]), float(row[tt]))
            episodes[ep] = win
    return fps, video_path, episodes


def _drawtext(text: str, x: str, y: str, size: int) -> str:
    """A drawtext filter with a translucent box; text is pre-escaped by the caller."""
    return (
        f"drawtext=fontfile={FONT}:text='{text}':x={x}:y={y}:"
        f"fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6"
    )


def build_ffmpeg_cmd(
    windows: Windows,
    dataset: str,
    episode: int,
    speed: float,
    out_path: Path,
    fps_cap: int,
) -> List[str]:
    """Assemble the ffmpeg command: trim each camera, label, hstack, speed up."""
    present = [c for c in CAMERAS if c in windows]
    if not present:
        raise ValueError(f"episode {episode}: none of {CAMERAS} present in dataset")

    cmd: List[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for cam in present:
        mp4, t0, t1 = windows[cam]
        if not mp4.exists():
            raise FileNotFoundError(f"missing video for {cam}: {mp4}")
        # Input-level seek: accurate enough here and avoids decoding the whole file.
        cmd += ["-ss", f"{t0:.6f}", "-to", f"{t1:.6f}", "-i", str(mp4)]

    # Per-panel camera-name label (bottom-left).
    parts = []
    labels = []
    for i, cam in enumerate(present):
        lbl = _drawtext(cam, x="8", y="h-th-8", size=22)
        parts.append(f"[{i}:v]{lbl}[p{i}]")
        labels.append(f"[p{i}]")

    # Stack, then a header on the full strip, then fast-forward.
    header = _drawtext(
        rf"{dataset}  ep{episode}  t=%{{pts\:hms}}  frame %{{n}}",
        x="8", y="8", size=22,
    )
    stack = f"{''.join(labels)}hstack=inputs={len(present)},{header},setpts=PTS/{speed}[out]"
    filtergraph = ";".join(parts + [stack])

    cmd += [
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-an",
        "-r", str(fps_cap),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(out_path),
    ]
    return cmd


def render_episode(
    episodes: Dict[int, Windows],
    dataset: str,
    episode: int,
    speed: float,
    out_dir: Path,
    fps_cap: int,
) -> Optional[Path]:
    if episode not in episodes:
        valid = ", ".join(str(e) for e in sorted(episodes))
        print(f"[skip] episode {episode} not found (valid: {valid})", file=sys.stderr)
        return None
    windows = episodes[episode]
    missing = [c for c in CAMERAS if c not in windows]
    if missing:
        print(f"[warn] episode {episode}: cameras absent, dropping panels: {missing}", file=sys.stderr)

    speed_tag = f"{speed:g}"
    out_path = out_dir / f"{dataset}_ep{episode}_{speed_tag}x.mp4"
    cmd = build_ffmpeg_cmd(windows, dataset, episode, speed, out_path, fps_cap)
    print(f"[render] episode {episode} -> {out_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[error] ffmpeg failed for episode {episode}:\n{proc.stderr}", file=sys.stderr)
        return None
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset name (directory under --root)")
    p.add_argument("--speed", type=float, default=1.0, help="fast-forward ratio, >= 1 (default 1)")
    p.add_argument("--episode", nargs="+", default=["all"], help="episode indices or 'all' (default all)")
    p.add_argument("--root", default=None, help="dataset root (default: config.yaml recorder.root)")
    p.add_argument("--out", default=None, help="output dir (default <root>/viz, outside the dataset folder)")
    p.add_argument("--fps-cap", type=int, default=60, help="max output fps after speed-up (default 60)")
    args = p.parse_args(argv)
    if args.speed < 1:
        p.error("--speed must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = find_root(args.root)
    dataset_dir = root / args.dataset
    if not dataset_dir.exists():
        print(f"[error] dataset not found: {dataset_dir}", file=sys.stderr)
        return 1

    fps, _, episodes = load_dataset(dataset_dir)
    print(f"[info] {args.dataset}: {len(episodes)} episodes, {fps} fps, root={root}")

    if len(args.episode) == 1 and args.episode[0].lower() == "all":
        wanted = sorted(episodes)
    else:
        try:
            wanted = [int(e) for e in args.episode]
        except ValueError:
            print(f"[error] --episode must be integers or 'all', got {args.episode}", file=sys.stderr)
            return 1

    # Keep renders OUT of the dataset dir so they never ride along on a dataset upload.
    out_dir = Path(os.path.expanduser(args.out)) if args.out else root / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = [render_episode(episodes, args.dataset, ep, args.speed, out_dir, args.fps_cap) for ep in wanted]
    ok = [p for p in done if p]
    print(f"[done] {len(ok)}/{len(wanted)} episodes rendered into {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
