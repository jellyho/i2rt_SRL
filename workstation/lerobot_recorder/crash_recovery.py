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

A second, quieter crash signature has no leftover files at all. LeRobot buffers episode
metadata in memory (``metadata_buffer_size``, 10 by default) and only writes ``data/`` and
``meta/episodes`` parquet footers on ``finalize()`` -- but it rewrites ``meta/info.json``
after *every* episode. Ctrl-C therefore leaves ``info.json`` counting episodes that never
reached the disk, and the next open fails somewhere far from the cause:

    FileNotFoundError: Cached dataset doesn't contain all requested episodes
    ... RepositoryNotFoundError: 404 ... huggingface.co/api/datasets/<repo>/refs

(LeRobot reads that shortfall as "the local copy is incomplete" and tries to download the
rest from the Hub, which for a local-only dataset is a 404.) The buffered episodes are
gone -- they never left memory -- so the repair is to roll the *counters* back to what
``meta/episodes`` actually holds: :func:`find_uncommitted_metadata` reports it and
:func:`truncate_to_committed` rewrites ``info.json``, re-aggregates ``meta/stats.json``
from the surviving per-episode stats, and drops the matching ``outcomes.jsonl`` rows.

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
    out: Dict[str, List[str]] = {
        k: [] for k in ("corrupt_data", "unreferenced_data", "unreferenced_video", "staged_images", "temp_dirs")
    }
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
    drift = find_uncommitted_metadata(ds_dir)
    if not found and drift is None:
        return "no crash leftovers"
    lines = []
    for reason, paths in found.items():
        for p in paths:
            size = _size(p)
            lines.append(f"  {reason:20s} {size:>9s}  {os.path.relpath(p, ds_dir)}")
    if drift is not None:
        lines.append(
            f"  {'uncommitted_meta':20s} {'':>9s}  meta/info.json claims "
            f"{drift['info_episodes']} episodes / {drift['info_frames']} frames, "
            f"meta/episodes holds {drift['committed_episodes']} / {drift['committed_frames']} "
            f"— {drift['lost_episodes']} episode(s) never left memory"
        )
    return "\n".join(lines)


def _size(path: str) -> str:
    try:
        if os.path.isdir(path):
            total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
        else:
            total = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024.0
    return "?"


def _committed_totals(ds_dir: str):
    """``(episodes, frames)`` that ``meta/episodes`` actually holds, or ``None``."""
    eps = _episode_table(ds_dir)
    if eps is None or len(eps) == 0:
        return None
    frames = int(eps["dataset_to_index"].max()) if "dataset_to_index" in eps.columns else int(eps["length"].sum())
    return len(eps), frames


def find_uncommitted_metadata(ds_dir: str) -> Optional[Dict]:
    """``info.json`` counters running ahead of what ``meta/episodes`` committed.

    LeRobot rewrites ``info.json`` per episode but flushes the episode/data parquets in
    batches, so a Ctrl-C mid-batch leaves the counters claiming episodes that were only
    ever in memory. Returns ``None`` when the two agree (or when there is nothing to
    compare); otherwise the counts, so the caller can say exactly how many episodes the
    interrupt cost.
    """
    info_path = os.path.join(ds_dir, "meta", "info.json")
    if not os.path.isfile(info_path):
        return None
    try:
        with open(info_path) as fh:
            info = json.load(fh)
        info_eps = int(info["total_episodes"])
        info_frames = int(info["total_frames"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    committed = _committed_totals(ds_dir)
    if committed is None:
        return None
    eps, frames = committed
    if info_eps <= eps and info_frames <= frames:
        return None
    return {
        "info_episodes": info_eps,
        "info_frames": info_frames,
        "committed_episodes": eps,
        "committed_frames": frames,
        "lost_episodes": info_eps - eps,
        "lost_frames": info_frames - frames,
    }


def _stats_shapes(ds_dir: str) -> Dict[str, Dict[str, tuple]]:
    """``{feature: {stat: shape}}`` from the existing ``meta/stats.json``.

    Round-tripping through parquet drops trailing 1-dims -- an image ``min`` comes back
    ``(3, 1)`` where LeRobot's validator insists on ``(3, 1, 1)`` -- and the shape is not
    derivable from the feature spec, so the file LeRobot itself wrote is the reference.
    """
    import numpy as np

    try:
        with open(os.path.join(ds_dir, "meta", "stats.json")) as fh:
            stats = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        feature: {stat: np.asarray(value).shape for stat, value in per_stat.items()}
        for feature, per_stat in stats.items()
        if isinstance(per_stat, dict)
    }


def _episode_stats(eps, index: int, shapes: Dict[str, Dict[str, tuple]]) -> Dict[str, Dict]:
    """One episode's row of ``stats/<feature>/<stat>`` columns, back in nested form."""
    import numpy as np

    row = eps.iloc[index]
    out: Dict[str, Dict] = {}
    for col in eps.columns:
        if not col.startswith("stats/"):
            continue
        _, feature, stat = col.split("/", 2)
        value = row[col]
        # ``count`` stays integral -- that is what LeRobot writes, and it is a frame count.
        dtype = np.int64 if stat == "count" else np.float64
        arr = np.array(value.tolist() if hasattr(value, "tolist") else value, dtype=dtype)
        want = shapes.get(feature, {}).get(stat)
        if want is not None and arr.shape != want and arr.size == int(np.prod(want)):
            arr = arr.reshape(want)
        out.setdefault(feature, {})[stat] = arr
    return out


def reaggregate_stats(ds_dir: str) -> bool:
    """Rewrite ``meta/stats.json`` from the per-episode stats in ``meta/episodes``.

    ``stats.json`` is rewritten after every episode, so it also carries the episodes an
    interrupt lost. Returns False (and leaves the file alone) if anything goes wrong --
    stale stats are a normalization inaccuracy, not a dataset that will not open.
    """
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import serialize_dict

    eps = _episode_table(ds_dir)
    if eps is None or len(eps) == 0:
        return False
    shapes = _stats_shapes(ds_dir)
    try:
        stats = aggregate_stats([_episode_stats(eps, i, shapes) for i in range(len(eps))])
        with open(os.path.join(ds_dir, "meta", "stats.json"), "w") as fh:
            json.dump(serialize_dict(stats), fh, indent=4)
    except Exception as e:
        logger.error("could not re-aggregate meta/stats.json (left as-is): %s", e)
        return False
    return True


def truncate_to_committed(ds_dir: str, *, quarantine: str) -> Optional[Dict]:
    """Roll ``info.json`` / ``stats.json`` / ``outcomes.jsonl`` back to the committed episodes.

    The originals are copied into ``quarantine`` first, so the pre-repair counters stay
    readable. ``stats.json`` is re-aggregated from the surviving per-episode stats rather
    than left alone: it is written per episode too, so it also carries the lost ones.
    Returns the :func:`find_uncommitted_metadata` report, or ``None`` if nothing was wrong.
    """
    drift = find_uncommitted_metadata(ds_dir)
    if drift is None:
        return None
    n_eps, n_frames = drift["committed_episodes"], drift["committed_frames"]
    os.makedirs(quarantine, exist_ok=True)

    info_path = os.path.join(ds_dir, "meta", "info.json")
    stats_path = os.path.join(ds_dir, "meta", "stats.json")
    outcomes_path = os.path.join(ds_dir, "outcomes.jsonl")
    for path in (info_path, stats_path, outcomes_path):
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(quarantine, os.path.relpath(path, ds_dir).replace(os.sep, "__")))

    with open(info_path) as fh:
        info = json.load(fh)
    info["total_episodes"] = n_eps
    info["total_frames"] = n_frames
    if isinstance(info.get("splits"), dict) and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{n_eps}"
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=4)

    if os.path.isfile(stats_path):
        reaggregate_stats(ds_dir)

    # The sidecar is appended per episode too, so it holds rows for episodes that are gone.
    if os.path.isfile(outcomes_path):
        kept = []
        for line in open(outcomes_path):
            if not line.strip():
                continue
            try:
                if int(json.loads(line)["episode"]) < n_eps:
                    kept.append(line if line.endswith("\n") else line + "\n")
            except (ValueError, KeyError, json.JSONDecodeError):
                kept.append(line if line.endswith("\n") else line + "\n")
        with open(outcomes_path, "w") as fh:
            fh.writelines(kept)

    logger.warning(
        "rolled metadata back to the %d committed episode(s): the interrupt lost %d episode(s) "
        "(%d frames) that were still buffered in memory; originals copied to %s",
        n_eps,
        drift["lost_episodes"],
        drift["lost_frames"],
        quarantine,
    )
    return drift


def recover(ds_dir: str, *, dry_run: bool = False, quarantine: Optional[str] = None) -> Dict:
    """Move everything :func:`find_crash_leftovers` reports out of the dataset.

    The quarantine is a **sibling** directory, not a subdirectory: LeRobot globs ``data/``
    and ``videos/``, so parking a broken parquet anywhere inside would leave the dataset
    just as unopenable as before.

    Also rolls ``info.json`` back onto the committed episodes -- last, because quarantining
    an unreferenced data parquet can itself lower what counts as committed.

    Returns ``{"moved": [...], "quarantine": path, "reasons": {...}, "uncommitted": {...}}``;
    ``moved`` is empty when there were no leftover files, ``uncommitted`` is ``None`` when
    the counters already agreed with the metadata.
    """
    found = find_crash_leftovers(ds_dir)
    drift = find_uncommitted_metadata(ds_dir)
    result: Dict = {"moved": [], "quarantine": None, "reasons": found, "uncommitted": drift}
    if not found and drift is None:
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

    result["uncommitted"] = truncate_to_committed(ds_dir, quarantine=quarantine)
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
