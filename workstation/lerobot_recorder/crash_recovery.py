"""Recover a LeRobot dataset from a recorder crash mid-episode.

An episode is committed in stages: frames are staged as PNGs under ``images/``, encoded
into per-camera mp4s, appended to the shared camera files, written into a data parquet, and
only then added to ``meta/episodes`` and ``info.json``. A crash between any two of those
leaves files on disk that the metadata never learned about.

Those leftovers are not merely untidy. A **half-written data parquet has no footer**, and
LeRobot globs ``data/`` rather than reading the file list out of the metadata — so one
truncated file makes the whole dataset fail to open, including the episodes that are
perfectly fine:

    DatasetGenerationError: An error occurred while generating the dataset

The committed episodes are almost always intact: the metadata is written last, so anything
it references was complete before the crash. Recovery is therefore not a repair of the
data but the removal of what the metadata does not claim.

:func:`find_crash_leftovers` reports them; :func:`recover` moves them out of the dataset —
never deletes — into a sibling ``<dataset>.crash-orphans.<timestamp>/`` so a mistake is
reversible and a genuinely valuable partial episode can still be dug out by hand.

Nothing here touches a file any episode references, so it cannot lose committed data.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _episode_table(ds_dir: str):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)


def _referenced_indices(eps, prefix: str) -> set:
    """The ``file_index`` values the metadata claims for ``prefix`` (e.g. ``data``)."""
    col = f"{prefix}/file_index"
    if eps is None or col not in eps.columns:
        return set()
    return {int(v) for v in eps[col].dropna().unique()}


def _parquet_readable(path: str) -> bool:
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def find_crash_leftovers(ds_dir: str) -> Dict[str, List[str]]:
    """Files on disk that no episode in ``meta/episodes`` refers to.

    Returns ``{"reason": [paths]}`` — grouped so the caller can say *why* each one is
    going, rather than presenting an undifferentiated list of things to delete:

    ``unreferenced_data``   data parquet whose ``file_index`` no episode claims
    ``corrupt_data``        data parquet that cannot be opened at all (this is the one
                            that takes the whole dataset down with it)
    ``unreferenced_video``  per-camera mp4 no episode claims
    ``staged_images``       ``images/`` PNG scratch from an episode that never encoded
    ``temp_dirs``           the writer's ``tmp*`` scratch directories

    Empty when the dataset is clean.
    """
    out: Dict[str, List[str]] = {k: [] for k in
                                 ("corrupt_data", "unreferenced_data", "unreferenced_video",
                                  "staged_images", "temp_dirs")}
    if not os.path.isdir(os.path.join(ds_dir, "meta")):
        return {k: v for k, v in out.items() if v}
    eps = _episode_table(ds_dir)

    # data parquet
    claimed = _referenced_indices(eps, "data")
    for path in sorted(glob.glob(os.path.join(ds_dir, "data", "**", "*.parquet"), recursive=True)):
        try:
            idx = int(os.path.basename(path).split("file-")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if idx in claimed:
            # Referenced AND unreadable would mean committed data is gone -- report it, but
            # never move it: that is a real loss the operator has to know about.
            if not _parquet_readable(path):
                logger.error("REFERENCED data file is unreadable, not touching it: %s", path)
            continue
        out["corrupt_data" if not _parquet_readable(path) else "unreferenced_data"].append(path)

    # videos, per camera key
    for cam_dir in sorted(glob.glob(os.path.join(ds_dir, "videos", "*"))):
        key = os.path.basename(cam_dir)
        claimed_v = _referenced_indices(eps, f"videos/{key}")
        for path in sorted(glob.glob(os.path.join(cam_dir, "**", "*.mp4"), recursive=True)):
            try:
                idx = int(os.path.basename(path).split("file-")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if idx not in claimed_v:
                out["unreferenced_video"].append(path)

    # PNG staging and temp dirs are scratch by construction -- the writer only keeps them
    # while an episode is in flight.
    images = os.path.join(ds_dir, "images")
    if os.path.isdir(images):
        out["staged_images"].append(images)
    for path in sorted(glob.glob(os.path.join(ds_dir, "tmp*"))):
        if os.path.isdir(path):
            out["temp_dirs"].append(path)

    return {k: v for k, v in out.items() if v}


def describe(ds_dir: str) -> str:
    """One-line-per-item summary of what recovery would move (or 'clean')."""
    found = find_crash_leftovers(ds_dir)
    if not found:
        return "no crash leftovers"
    lines = []
    for reason, paths in found.items():
        for p in paths:
            size = _size(p)
            lines.append(f"  {reason:20s} {size:>9s}  {os.path.relpath(p, ds_dir)}")
    return "\n".join(lines)


def _size(path: str) -> str:
    try:
        if os.path.isdir(path):
            total = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs
            )
        else:
            total = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024.0
    return "?"


def recover(ds_dir: str, *, dry_run: bool = False, quarantine: Optional[str] = None) -> Dict:
    """Move everything :func:`find_crash_leftovers` reports out of the dataset.

    The quarantine is a **sibling** directory, not a subdirectory: LeRobot globs ``data/``
    and ``videos/``, so parking a broken parquet anywhere inside would leave the dataset
    just as unopenable as before.

    Returns ``{"moved": [...], "quarantine": path, "reasons": {...}}``; ``moved`` is empty
    when the dataset was already clean.
    """
    found = find_crash_leftovers(ds_dir)
    result: Dict = {"moved": [], "quarantine": None, "reasons": found}
    if not found:
        return result

    if dry_run:
        return result

    ds_dir = os.path.abspath(ds_dir.rstrip("/"))
    if quarantine is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        quarantine = f"{ds_dir}.crash-orphans.{stamp}"
    os.makedirs(quarantine, exist_ok=True)
    result["quarantine"] = quarantine

    for reason, paths in found.items():
        for path in paths:
            # Flatten into the quarantine, keeping enough of the path to stay unambiguous
            # (two cameras both have a file-019.mp4).
            rel = os.path.relpath(path, ds_dir).replace(os.sep, "__")
            dst = os.path.join(quarantine, rel)
            shutil.move(path, dst)
            result["moved"].append(dst)
            logger.warning("quarantined %s (%s) -> %s", os.path.relpath(path, ds_dir), reason, dst)

    return result


def verify(ds_dir: str) -> Dict:
    """Open the dataset the way a training run would. ``{"ok": bool, "error": str, ...}``.

    Recovery that is not checked by loading is not recovery: the failure this fixes only
    shows up when something globs the data directory.
    """
    out: Dict = {"ok": False, "error": "", "episodes": None, "frames": None}
    try:
        info = json.load(open(os.path.join(ds_dir, "meta", "info.json")))
        out["episodes"] = info.get("total_episodes")
        out["frames"] = info.get("total_frames")
    except Exception as e:
        out["error"] = f"cannot read meta/info.json: {e}"
        return out
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset("local/verify", root=ds_dir)
        ds[0]
        ds[len(ds) - 1]
        out["ok"] = True
        out["episodes"] = ds.meta.total_episodes
        out["frames"] = ds.meta.total_frames
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out
