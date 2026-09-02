"""Safe in-place editing of a recorded LeRobot **v3.0** dataset.

The viewer/editor GUI that used to drive this now lives in `hf-utils` (see the README).
What remains here is what the *recorder* needs: the video/length consistency check run by
``dataset_writer.finalize()`` and the homing-tail marking behind ``yam-data mark-homing``,
both of which encode YAM specifics a general tool should not. The heavy structural
operations (deleting episodes, changing tasks) go through LeRobot's official
``lerobot.datasets.dataset_tools`` so the parquet data, videos, and ``meta/`` stay
consistent (episodes are re-indexed, ``info.json`` counts updated, etc.). The episode
verdict is two ordinary features (``next.success`` / ``next.done``, see ``outcomes.py``), so
those tools carry it along for free.

Every structural edit **backs the dataset up first** — the original folder is
renamed aside to ``<name>.backup-<op>.<timestamp>`` (the convention already used on
disk) so a bad edit is always recoverable. Lightweight edits (relabelling an episode's
verdict) rewrite two bool columns of one episode and its stats, and need no backup.

``delete_episodes`` writes a brand-new dataset in a temp dir and then swaps it into
place; ``rl_config.json`` is copied across. Nothing here requires the robot or cameras — only ``lerobot`` + ``pyarrow``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder import outcomes as _outcomes
from workstation.lerobot_recorder.config import ARM_DOF
from workstation.lerobot_recorder.dataset_writer import dataset_dir

logger = logging.getLogger(__name__)

# Gripper joint positions inside observation.state = [L.pos(7), L.vel, L.eff, R.pos(7), ...];
# the gripper is the last DOF of each arm's pos block.
_L_GRIP = ARM_DOF - 1
_R_GRIP = 3 * ARM_DOF + (ARM_DOF - 1)


def detect_homing_start(
    gripper: "np.ndarray", *, closed_ceiling: float = 0.15, close_margin: float = 0.06, min_drop: float = 0.15
) -> Optional[int]:
    """Find where the automatic HOMING return begins, from the gripper signal.

    Every episode ends with the gripper closing to a floor and holding there while the
    arms return home. This returns the frame index where that **final close** begins (the
    top of the last descent into the sustained closed hold), or ``None`` if the episode
    doesn't end with the gripper actually closed (so nothing is guessed). Gripper is
    normalized ~1 open / ~0 closed. Pure + smoothed → robust to sensor noise; unit-tested.
    """
    g = np.asarray(gripper, dtype=np.float64).reshape(-1)
    n = g.size
    if n < 20:
        return None
    # light moving-average smoothing
    w = 5
    g = np.convolve(g, np.ones(w) / w, mode="same")
    floor = float(np.median(g[-10:]))
    if floor > closed_ceiling:
        return None  # the episode doesn't END with the gripper actually closed → don't guess
    close_thr = floor + close_margin
    if g[-1] > close_thr:
        return None  # didn't settle at the floor → don't guess
    # sustained closed hold at the very end
    i = n - 1
    while i > 0 and g[i] <= close_thr:
        i -= 1
    hold_start = i + 1
    if hold_start >= n or (n - hold_start) < 3:
        return None
    # walk back through the final descent (closing forward ⇒ rising as we step back)
    j = hold_start
    while j - 1 >= 0 and g[j - 1] > g[j] + 1e-4:
        j -= 1
    if g[j] - floor < min_drop:
        return None  # not a real close (too small a drop) → skip
    return int(j)


# --------------------------------------------------------------------------- pure helpers
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


# --------------------------------------------------------------------------- editor
class DatasetEditor:
    """Metadata reader + structural editor for one dataset at ``<root>/<name>``."""

    def __init__(self, repo_id: str, root: str) -> None:
        self.repo_id = repo_id
        self.root = root
        self.ds_dir = dataset_dir(root, repo_id)
        self._ds = None  # loaded LeRobotDataset (lazily)

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
        """Per-episode verdict (``success`` / ``fail`` / ``unknown``) from the dataset's own
        metadata. Empty for a dataset that predates the outcome features."""
        return _outcomes.episode_outcomes(self.ds_dir, strict=False)

    def frames_by_episode(self) -> Dict[int, int]:
        """Per-episode frame count, from ``meta/episodes``."""
        return _outcomes.episode_lengths(self.ds_dir)

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
    # ---------------------------------------------------- non-destructive homing trim
    def set_homing_tail(self, episode: int, from_frame: int) -> int:
        """Mark this episode's frames ``>= from_frame`` as homing (``observation.control_mode
        = homing``), so training can drop the homing tail — e.g. treat a failed episode as
        ending at the failure. Non-destructive: no frames removed, no video re-encode; it
        only rewrites the scalar ``control_mode`` column (present in every recorded dataset).
        Returns the number of frames marked."""
        from workstation.lerobot_recorder.config import CONTROL_MODE

        return self._set_control_mode(
            lambda df: (df["episode_index"] == episode) & (df["frame_index"] >= int(from_frame)),
            float(CONTROL_MODE["homing"]),
        )

    def clear_homing(self, episode: int) -> int:
        """Reset this episode's homing-marked frames back to teleop. Returns frames cleared."""
        from workstation.lerobot_recorder.config import CONTROL_MODE

        homing, teleop = float(CONTROL_MODE["homing"]), float(CONTROL_MODE["teleop"])
        return self._set_control_mode(
            lambda df: (df["episode_index"] == episode) & (df["observation.control_mode"] == homing),
            teleop,
        )

    def episode_homing_starts(self, episodes: Optional[List[int]] = None) -> Dict[int, int]:
        """Auto-detect the homing-return start (within-episode frame index) for each episode
        from the gripper signal — see :func:`detect_homing_start`. Read-only: returns
        ``{episode: from_frame}`` only for episodes that clearly end with the gripper closing
        (others are omitted, never guessed). ``episodes`` limits the scan."""
        import glob as _glob

        import pandas as pd

        files = sorted(_glob.glob(os.path.join(self.ds_dir, "data", "**", "*.parquet"), recursive=True))
        if not files:
            return {}
        df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                        for f in files])
        want = set(episodes) if episodes is not None else None
        starts: Dict[int, int] = {}
        for ep, e in df.groupby("episode_index"):
            ep = int(ep)
            if want is not None and ep not in want:
                continue
            e = e.sort_values("frame_index")
            state = np.stack(e["observation.state"].to_numpy())
            if state.shape[1] <= _R_GRIP:
                continue
            gripper = 0.5 * (state[:, _L_GRIP] + state[:, _R_GRIP])
            hs = detect_homing_start(gripper)
            if hs is not None:
                starts[ep] = int(e["frame_index"].to_numpy()[hs])
        return starts

    def auto_mark_homing(self, episodes: Optional[List[int]] = None) -> Dict[int, int]:
        """Detect and mark the homing tail of every episode in one pass (non-destructive).

        Sets ``observation.control_mode = homing`` from each detected start to the episode
        end, in a single batched parquet rewrite. Returns ``{episode: frames_marked}``.
        """
        from workstation.lerobot_recorder.config import CONTROL_MODE

        starts = self.episode_homing_starts(episodes)
        if not starts:
            return {}

        def predicate(df: "object") -> "np.ndarray":  # rows at/after each episode's homing start
            ei = df["episode_index"].to_numpy()
            fi = df["frame_index"].to_numpy()
            mask = np.zeros(len(df), dtype=bool)
            for ep, frm in starts.items():
                mask |= (ei == ep) & (fi >= frm)
            return mask

        self._set_control_mode(predicate, float(CONTROL_MODE["homing"]))
        # frames marked per episode = length - start (length = max frame_index + 1)
        return starts

    def _set_control_mode(self, predicate: "Callable", value: float) -> int:
        """Set ``observation.control_mode = value`` for rows matching ``predicate(df)`` across
        the data parquet files, then refresh that feature's stats. Metadata-only edit."""
        import glob as _glob

        import pandas as pd

        col = "observation.control_mode"
        files = sorted(_glob.glob(os.path.join(self.ds_dir, "data", "**", "*.parquet"), recursive=True))
        n = 0
        for f in files:
            df = pd.read_parquet(f)
            if col not in df.columns:
                continue
            mask = predicate(df)
            count = int(mask.sum())
            if count:
                df.loc[mask, col] = value
                df[col] = df[col].astype("float32")
                # Atomic replace (write new inode, then rename) — never overwrite the parquet
                # in place: a reader may still have the old file memory-mapped, and truncating
                # it under them corrupts their view. os.replace leaves the old inode intact.
                tmp = f"{f}.tmp"
                df.to_parquet(tmp, index=False)
                os.replace(tmp, f)
                n += count
        if n:
            self._refresh_control_mode_stats(files)
            self._ds = None  # data changed on disk; reopen on next use
        return n

    def _refresh_control_mode_stats(self, files: List[str]) -> None:
        """Recompute meta/stats.json for observation.control_mode from the edited column."""
        import numpy as np
        import pandas as pd

        stats_path = os.path.join(self.ds_dir, "meta", "stats.json")
        if not os.path.exists(stats_path):
            return
        try:
            vals = np.concatenate([
                pd.read_parquet(f, columns=["observation.control_mode"])["observation.control_mode"]
                .to_numpy(dtype=np.float64)
                for f in files
            ])
            stats = json.load(open(stats_path))
            stats["observation.control_mode"] = {
                "min": [float(vals.min())], "max": [float(vals.max())],
                "mean": [float(vals.mean())], "std": [float(vals.std())],
                "count": [int(vals.size)],
                **{f"q{q:02d}": [float(np.percentile(vals, q))] for q in (1, 10, 50, 90, 99)},
            }
            tmp = f"{stats_path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(stats, fh)
            os.replace(tmp, stats_path)
        except Exception as e:
            logger.warning("could not refresh control_mode stats: %s", e)

    def relabel(self, episode: int, outcome: str) -> None:
        """Set an episode's verdict: rewrites its ``next.success`` / ``next.done`` terminal
        frame and the episode's stats for those two features, then re-aggregates
        ``meta/stats.json``. Videos and every other column are untouched."""
        _outcomes.relabel(self.ds_dir, episode, outcome)
        self._ds = None  # data changed on disk; reopen on next use
        logger.info("relabelled episode %d -> %s", episode, _outcomes.normalize(outcome))

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
        temp dir (the verdict columns travel with the frames), then atomically swaps it into
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
        ``task_index`` column).
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

        self._ds = None  # metadata changed; reload on next use
        logger.info("set task for %d episode(s) -> %r", len(episodes), task)
        return backup_path
