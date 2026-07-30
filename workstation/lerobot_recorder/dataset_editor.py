"""Safe in-place editing of a recorded LeRobot **v3.0** dataset.

The viewer/editor GUI (``editor_gui.py``) drives this backend. The heavy structural
operations (deleting episodes, changing tasks) go through LeRobot's official
``lerobot.datasets.dataset_tools`` so the parquet data, videos, and ``meta/`` stay
consistent (episodes are re-indexed, ``info.json`` counts updated, etc.). Around
those we also keep the recorder's own ``outcomes.jsonl`` sidecar in sync.

Every structural edit **backs the dataset up first** — the original folder is
renamed aside to ``<name>.backup-<op>.<timestamp>`` (the convention already used on
disk) so a bad edit is always recoverable. Lightweight edits (relabelling an
episode outcome) only touch the sidecar and need no backup.

``delete_episodes`` writes a brand-new dataset in a temp dir and then swaps it into
place; the other files (``outcomes.jsonl``, ``rl_config.json``) are remapped/copied
across. Nothing here requires the robot or cameras — only ``lerobot`` + ``pyarrow``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List

from workstation.lerobot_recorder.dataset_writer import dataset_dir

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- pure helpers
def remap_outcome_entries(entries: List[dict], deleted: List[int], total_episodes: int) -> List[dict]:
    """Drop the deleted episodes' outcome rows and re-index the survivors to 0..k-1.

    ``entries`` is the parsed ``outcomes.jsonl``; ``deleted`` the episode indices being
    removed; ``total_episodes`` the count *before* deletion. Survivors keep their
    relative order and are renumbered exactly the way LeRobot re-indexes the data.
    Rows without a valid ``episode`` field are dropped (they can't be placed).
    """
    deleted_set = set(deleted)
    keep_old = [i for i in range(total_episodes) if i not in deleted_set]
    mapping = {old: new for new, old in enumerate(keep_old)}

    by_ep: Dict[int, dict] = {}
    for e in entries:
        try:
            ep = int(e["episode"])
        except (KeyError, TypeError, ValueError):
            continue
        if ep in mapping:
            by_ep[ep] = e

    out: List[dict] = []
    for old in keep_old:
        e = by_ep.get(old)
        if e is None:
            continue
        e = dict(e)
        e["episode"] = mapping[old]
        out.append(e)
    return out


def video_length_mismatches(ds_dir: str) -> List[dict]:
    """Find episodes whose stored video window disagrees with the frame ``length``.

    A GPU/streaming encoder can drop an episode's trailing frame, so a camera's video
    ends up 1 frame shorter than the recorded ``length`` (data + other cameras stay
    full). Returns one dict per offending ``(episode, camera)``:
    ``{"episode", "camera", "length", "derived", "file"}``. Empty when consistent.
    Metadata-only (no video decode) — cheap enough to run after every recording.
    """
    import glob as _glob

    import pandas as pd

    info_path = os.path.join(ds_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return []
    fps = int(json.load(open(info_path))["fps"])
    files = sorted(_glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))
    out: List[dict] = []
    for f in files:
        df = pd.read_parquet(f)
        vkeys = [c.split("/", 2)[1] for c in df.columns if c.endswith("/from_timestamp")]
        for i in range(len(df)):
            length = int(df.iloc[i]["length"])
            ep = int(df.iloc[i]["episode_index"])
            for vk in vkeys:
                derived = round(float(df.iloc[i][f"videos/{vk}/to_timestamp"]) * fps) - round(
                    float(df.iloc[i][f"videos/{vk}/from_timestamp"]) * fps
                )
                if derived != length:
                    out.append({"episode": ep, "camera": vk, "length": length, "derived": derived, "file": f})
    return out


def repair_length_consistency(ds_dir: str) -> int:
    """Make ``meta/episodes`` self-consistent so lerobot's editors accept the dataset.

    For every ``(episode, camera)`` whose stored ``to_timestamp - from_timestamp`` disagrees
    with ``length`` (see :func:`video_length_mismatches`), snaps ``to_timestamp`` so the window
    spans exactly ``length`` frames. Repairs *metadata only* — data and video bytes are
    untouched. Returns the number of fields repaired.
    """
    import glob as _glob

    import pandas as pd

    info_path = os.path.join(ds_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return 0
    fps = int(json.load(open(info_path))["fps"])
    files = sorted(_glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))
    repaired = 0
    for f in files:
        df = pd.read_parquet(f)
        vkeys = [c.split("/", 2)[1] for c in df.columns if c.endswith("/from_timestamp")]
        changed = False
        for i in range(len(df)):
            length = int(df.iloc[i]["length"])
            for vk in vkeys:
                fcol, tcol = f"videos/{vk}/from_timestamp", f"videos/{vk}/to_timestamp"
                from_frame = round(float(df.iloc[i][fcol]) * fps)
                if round(float(df.iloc[i][tcol]) * fps) - from_frame != length:
                    df.iat[i, df.columns.get_loc(tcol)] = (from_frame + length) / fps
                    repaired += 1
                    changed = True
        if changed:
            df.to_parquet(f, index=False)
    return repaired


def _read_jsonl(path: str) -> List[dict]:
    entries: List[dict] = []
    if not os.path.exists(path):
        return entries
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed outcome row in %s", path)
    return entries


def _write_jsonl(path: str, entries: List[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- editor
class DatasetEditor:
    """Metadata reader + structural editor for one dataset at ``<root>/<name>``."""

    def __init__(self, repo_id: str, root: str) -> None:
        self.repo_id = repo_id
        self.root = root
        self.ds_dir = dataset_dir(root, repo_id)
        self._ds = None  # loaded LeRobotDataset (lazily)

    # ------------------------------------------------------------------ paths
    @property
    def outcomes_path(self) -> str:
        return os.path.join(self.ds_dir, "outcomes.jsonl")

    # ------------------------------------------------------------------ load / inspect
    def load(self) -> None:
        """Open the underlying LeRobotDataset (needed for delete/task edits)."""
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self._ds = LeRobotDataset(self.repo_id, root=self.ds_dir)

    def attach(self, ds: object) -> None:
        """Reuse an already-open ``LeRobotDataset`` (e.g. the reader's) to avoid a second
        full open — both point at the same ``<root>/<name>`` folder."""
        self._ds = ds

    @property
    def total_episodes(self) -> int:
        if self._ds is not None:
            meta = getattr(self._ds, "meta", None)
            return int(getattr(self._ds, "num_episodes", getattr(meta, "total_episodes", 0)) or 0)
        # metadata-only fallback
        info = os.path.join(self.ds_dir, "meta", "info.json")
        if os.path.exists(info):
            try:
                return int(json.load(open(info)).get("total_episodes", 0) or 0)
            except Exception:
                return 0
        return 0

    def outcomes_by_episode(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for e in _read_jsonl(self.outcomes_path):
            try:
                out[int(e["episode"])] = str(e["outcome"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def sources_by_episode(self) -> Dict[int, str]:
        """Per-episode record source from the sidecar: ``teleop`` / ``dagger`` / ``eval``."""
        out: Dict[int, str] = {}
        for e in _read_jsonl(self.outcomes_path):
            src = e.get("source")
            if e.get("episode") is not None and src is not None:
                try:
                    out[int(e["episode"])] = str(src)
                except (TypeError, ValueError):
                    continue
        return out

    def frames_by_episode(self) -> Dict[int, int]:
        """Per-episode recorded frame count from the sidecar (best-effort)."""
        out: Dict[int, int] = {}
        for e in _read_jsonl(self.outcomes_path):
            if e.get("episode") is not None and e.get("frames") is not None:
                try:
                    out[int(e["episode"])] = int(e["frames"])
                except (TypeError, ValueError):
                    continue
        return out

    def tasks_by_episode(self) -> Dict[int, str]:
        """Per-episode task string, read from the LeRobot metadata (best-effort)."""
        out: Dict[int, str] = {}
        meta = getattr(self._ds, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is None:
            return out
        try:
            iterator = iter(episodes)
        except TypeError:
            return out
        for ep in iterator:
            try:
                idx = int(ep["episode_index"])
            except Exception:
                continue
            tasks = ep.get("tasks")
            if isinstance(tasks, str):
                out[idx] = tasks
            elif tasks is not None and len(tasks):
                out[idx] = str(tasks[0])
        return out

    # ------------------------------------------------------------------ lightweight edit
    def relabel(self, episode: int, outcome: str) -> None:
        """Set an episode's outcome in the sidecar (no LeRobot rewrite; cheap & instant).

        Outcomes are the recorder's own annotation (``success`` / ``fail`` / ``discard``);
        they live only in ``outcomes.jsonl`` and don't touch the training data.
        """
        entries = _read_jsonl(self.outcomes_path)
        found = False
        for e in entries:
            try:
                if int(e.get("episode")) == int(episode):
                    e["outcome"] = outcome
                    found = True
            except (TypeError, ValueError):
                continue
        if not found:
            entries.append({"episode": int(episode), "outcome": outcome})
            entries.sort(key=lambda e: int(e.get("episode", 0)))
        os.makedirs(self.ds_dir, exist_ok=True)
        _write_jsonl(self.outcomes_path, entries)
        logger.info("relabelled episode %d -> %s", episode, outcome)

    # ------------------------------------------------------------------ backup
    def _backup(self, op: str) -> str:
        """Rename the dataset dir aside to ``<name>.backup-<op>.<timestamp>`` and return it."""
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = f"{self.ds_dir}.backup-{op}.{stamp}"
        shutil.copytree(self.ds_dir, backup, symlinks=True)
        logger.info("backed up dataset to %s", backup)
        return backup

    # ------------------------------------------------------------------ structural edits
    def delete_episodes(self, indices: List[int], backup: bool = True) -> str:
        """Delete episodes and re-index the dataset in place.

        Builds a fresh, re-indexed dataset with ``lerobot.datasets.dataset_tools`` in a
        temp dir, remaps the ``outcomes.jsonl`` sidecar, then atomically swaps it into
        place (the original is copied to a ``.backup-...`` dir first when ``backup``).
        Returns the backup path (or "" if not backed up).
        """
        from lerobot.datasets.dataset_tools import delete_episodes as _delete_episodes

        indices = sorted(set(int(i) for i in indices))
        if not indices:
            raise ValueError("no episodes selected to delete")
        if self._ds is None:
            self.load()
        total = self.total_episodes
        if len(indices) >= total:
            raise ValueError("cannot delete all episodes from the dataset")

        parent = os.path.dirname(self.ds_dir)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        tmp_dir = os.path.join(parent, f".{os.path.basename(self.ds_dir)}.edit-{stamp}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        backup_path = ""
        try:
            # Re-indexed dataset written to a temp dir (same repo_id so meta is identical).
            # If lerobot rejects a length/timestamp inconsistency (GPU-encoder dropped a
            # trailing frame), repair the metadata once and retry — see repair_length_consistency.
            try:
                _delete_episodes(self._ds, episode_indices=indices, output_dir=tmp_dir, repo_id=self.repo_id)
            except AssertionError as e:
                if "length mismatch" not in str(e).lower():
                    raise
                shutil.rmtree(tmp_dir, ignore_errors=True)
                n = self.repair_length_consistency()
                logger.warning("repaired %d length/timestamp inconsistency(ies); retrying delete", n)
                self.load()  # reopen against the repaired metadata
                _delete_episodes(self._ds, episode_indices=indices, output_dir=tmp_dir, repo_id=self.repo_id)

            # Remap our sidecars into the new dataset before the swap.
            new_outcomes = remap_outcome_entries(_read_jsonl(self.outcomes_path), indices, total)
            if new_outcomes:
                _write_jsonl(os.path.join(tmp_dir, "outcomes.jsonl"), new_outcomes)
            rl_cfg = os.path.join(self.ds_dir, "rl_config.json")
            if os.path.exists(rl_cfg):
                shutil.copy2(rl_cfg, os.path.join(tmp_dir, "rl_config.json"))

            # The new dataset already lives in tmp_dir, so we move the original *aside*
            # (instant, no extra disk) rather than copytree-ing it — then swap tmp in.
            if backup:
                backup_path = f"{self.ds_dir}.backup-delete-{len(indices)}ep.{stamp}"
                os.replace(self.ds_dir, backup_path)
                logger.info("moved original dataset aside to %s", backup_path)
            else:
                shutil.rmtree(self.ds_dir)
            os.replace(tmp_dir, self.ds_dir)
        except BaseException:
            # Any failure (including the repair path) must not leak the temp dir; the
            # original dataset is only ever moved in the final swap above, so it is safe.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        self._ds = None  # force a reload against the new on-disk layout
        logger.info("deleted %d episode(s) %s; dataset now has %d", len(indices), indices, self.total_episodes)
        return backup_path

    def repair_length_consistency(self) -> int:
        """Snap ``meta/episodes`` timestamps to match ``length`` so lerobot's editors
        accept the dataset. Delegates to the module-level
        :func:`repair_length_consistency`; returns the number of fields repaired."""
        return repair_length_consistency(self.ds_dir)

    def set_task(self, episodes: List[int], task: str, backup: bool = True) -> str:
        """Set the language instruction (task) for the given episodes, in place.

        Uses ``dataset_tools.modify_tasks`` (rewrites ``tasks.parquet`` + the
        ``task_index`` column) and updates the sidecar's ``task`` field to match.
        Returns the backup path (or "").
        """
        from lerobot.datasets.dataset_tools import modify_tasks

        episodes = sorted(set(int(i) for i in episodes))
        task = task.strip()
        if not episodes:
            raise ValueError("no episodes selected")
        if not task:
            raise ValueError("task text is empty")
        if self._ds is None:
            self.load()

        backup_path = self._backup("set-task") if backup else ""
        try:
            modify_tasks(self._ds, episode_tasks={e: task for e in episodes})
        except Exception:
            raise

        entries = _read_jsonl(self.outcomes_path)
        ep_set = set(episodes)
        for e in entries:
            try:
                if int(e.get("episode")) in ep_set:
                    e["task"] = task
            except (TypeError, ValueError):
                continue
        if entries:
            _write_jsonl(self.outcomes_path, entries)
        self._ds = None  # metadata changed; reload on next use
        logger.info("set task for %d episode(s) -> %r", len(episodes), task)
        return backup_path
