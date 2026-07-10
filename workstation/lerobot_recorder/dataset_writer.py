"""Async, queued ``LeRobotDataset`` (v3.0) writer for bimanual YAM episodes.

The recorder buffers each episode's frames in memory and **submits the whole
episode** to this writer. A single background worker thread then encodes/saves
episodes **one at a time** off a queue — so LeRobot's per-trajectory processing
(``save_episode``: parquet + video encode) never blocks the next collection.

Targets the official LeRobot Dataset **v3.0** API (``lerobot >= 0.4.0``):

* ``LeRobotDataset.create(...)`` / load existing for ``--resume``
* ``add_frame(frame)`` — per-frame ``task`` is a key inside the frame dict
* ``save_episode()`` / ``clear_episode_buffer()`` / ``finalize()``

Each frame is a dict of ``{feature_key: np.ndarray}`` plus ``"images"`` (a
``{cam: HxWx3}`` dict) and ``"task"``. The feature schema is built from a sample
frame so new fields (leader pose, eef, control_mode, …) flow through with no
schema edits here. ``mock=True`` skips ``lerobot`` and just counts.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig

logger = logging.getLogger(__name__)


def _import_lerobot_dataset() -> type:
    try:
        from lerobot.datasets import LeRobotDataset  # lerobot >= 0.4.0 (v3.0)
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def dataset_dir(root: str, repo_id: str) -> str:
    """The actual dataset folder: ``<root>/<name>`` where ``name`` is the last segment
    of ``repo_id`` (e.g. root=~/lerobot_data, repo_id=hello/pick_and_place ->
    ~/lerobot_data/pick_and_place). ``root`` is a PARENT dir holding one folder per
    dataset, so several datasets can live side by side."""
    name = repo_id.strip("/").split("/")[-1] or "dataset"
    return os.path.join(os.path.expanduser(root), name)


def list_datasets(root: str) -> List[str]:
    """Dataset folder names under the parent ``root`` (for the setup-page picker).

    Every non-hidden subdirectory counts — a fresh dataset has no metadata yet.
    Sorted; [] when the root doesn't exist."""
    path = os.path.expanduser(root)
    try:
        return sorted(
            d for d in os.listdir(path)
            if not d.startswith(".") and os.path.isdir(os.path.join(path, d))
        )
    except OSError:
        return []


def dataset_tasks(ds_dir: str) -> List[str]:
    """Task strings already used by the dataset at ``ds_dir`` (best-effort).

    Reads LeRobot v3 ``meta/tasks.parquet`` (tasks are the index) or the v2
    ``meta/tasks.jsonl`` fallback. [] when neither exists or parsing fails."""
    meta = os.path.join(os.path.expanduser(ds_dir), "meta")
    parquet = os.path.join(meta, "tasks.parquet")
    if os.path.exists(parquet):
        try:
            import pandas as pd

            return [str(t) for t in pd.read_parquet(parquet).index]
        except Exception:
            return []
    jsonl = os.path.join(meta, "tasks.jsonl")
    if os.path.exists(jsonl):
        try:
            with open(jsonl) as fh:
                return [json.loads(line)["task"] for line in fh if line.strip()]
        except Exception:
            return []
    return []


def dataset_info(root: str) -> Dict:
    """Inspect the dataset dir at ``root`` for the setup page — no lerobot import.

    ``{"exists": bool, "episodes": int|None}``. Episode count is a best-effort read
    of LeRobot metadata, falling back to the ``outcomes.jsonl`` sidecar."""
    path = os.path.expanduser(root)
    if not os.path.isdir(path) or not os.listdir(path):
        return {"exists": False, "episodes": None}
    episodes: Optional[int] = None
    info_path = os.path.join(path, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as fh:
                meta_episodes = json.load(fh).get("total_episodes")
            if meta_episodes is not None:
                episodes = int(meta_episodes)
        except Exception:
            episodes = None
    sidecar = os.path.join(path, "outcomes.jsonl")
    if episodes is None and os.path.exists(sidecar):
        try:
            with open(sidecar) as fh:
                episodes = sum(1 for line in fh if line.strip())
        except Exception:
            episodes = None
    return {"exists": True, "episodes": episodes}


def remove_dataset_root(root: str) -> None:
    """Delete the dataset dir at ``root`` (used by the GUI's confirmed overwrite)."""
    path = os.path.expanduser(root)
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info("removed existing dataset dir %s", path)


class AsyncDatasetWriter:
    """Queued episode writer. ``submit()`` returns immediately; a worker saves."""

    def __init__(self, cfg: RecorderConfig, image_keys: List[str], image_shapes: Dict[str, tuple]) -> None:
        self.cfg = cfg
        self.image_keys = image_keys
        self.image_shapes = image_shapes
        self._mock = cfg.mock
        # Output format: "lerobot" (LeRobotDataset) or "abcdl" (the abcdl MP4+binary
        # training cache; one episode dir per submit, written via abcdl.EpisodeWriter).
        self._abcdl = str(getattr(cfg, "record_format", "lerobot")).lower() == "abcdl"
        self._ds = None
        self._features: Optional[dict] = None
        # The dataset lives in <root>/<name>; root is just the parent directory.
        self._root = dataset_dir(cfg.root, cfg.repo_id)
        self._outcomes_path = os.path.join(self._root, "outcomes.jsonl")
        # Episodes already in the dataset before this session (resume); re-synced to
        # the authoritative count in open(). total/new_episodes build on this.
        self._initial_episodes = int(dataset_info(self._root)["episodes"] or 0) if cfg.resume else 0

        self._queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._n_episodes = 0  # saved episodes (incremented by the worker)
        self._n_submitted = 0
        self._saving = False  # True while the worker is encoding/writing one episode
        self._saving_index: Optional[int] = None  # episode index currently being saved
        self._saving_frames = 0  # frame count of the episode currently being saved
        self._finalized = False
        self._finalize_lock = threading.Lock()
        self.low_disk = False  # set when an episode was refused for lack of free space

    def _free_gb(self) -> float:
        path = self._root if os.path.isdir(self._root) else (os.path.dirname(self._root) or ".")
        try:
            return shutil.disk_usage(path).free / 1e9
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _vector_names(key: str, dim: int) -> List[str]:
        from workstation.lerobot_recorder.config import action_names, eef_names, leader_names, state_names

        if key == "observation.state":
            return state_names()
        if key == "action":
            return action_names()
        if key == "observation.leader":
            return leader_names()
        if key == "observation.eef":
            return eef_names()
        return [f"{key.rsplit('.', 1)[-1]}.{i}" for i in range(dim)]

    def _build_features(self, sample: dict) -> dict:
        img_dtype = "video" if self.cfg.use_videos else "image"
        feats: dict = {}
        for key in self.image_keys:
            feats[f"observation.images.{key}"] = {
                "dtype": img_dtype,
                "shape": self.image_shapes[key],
                "names": ["height", "width", "channels"],
            }
        for key, val in sample.items():
            if key in ("images", "task"):
                continue
            vec = np.asarray(val, dtype=np.float32).reshape(-1)
            feats[key] = {"dtype": "float32", "shape": (vec.size,), "names": self._vector_names(key, vec.size)}
        if getattr(self.cfg, "rl_features", False):
            for key in ("success", "reward", "mc_return"):  # per-frame RL signals
                feats[key] = {"dtype": "float32", "shape": (1,), "names": None}
        return feats

    def _dataset_encoding_kwargs(self) -> dict:
        enc = {
            "vcodec": self.cfg.vcodec,
            "batch_encoding_size": max(1, int(self.cfg.batch_encoding_size)),
        }
        if int(self.cfg.encoder_threads) > 0:
            enc["encoder_threads"] = int(self.cfg.encoder_threads)
        # Direct-to-encoder streaming (no temp-PNG round-trip). Only passed when
        # enabled so stock lerobot builds without the kwarg keep working.
        if bool(getattr(self.cfg, "streaming_encoding", False)):
            enc["streaming_encoding"] = True
        return enc

    def _create_encoding_kwargs(self) -> dict:
        enc = self._dataset_encoding_kwargs()
        # Async PNG writer (parallelizes the slow pre-encode step). Only pass when
        # enabled so older LeRobot versions without these kwargs keep working.
        if int(self.cfg.image_writer_threads) > 0 or int(self.cfg.image_writer_processes) > 0:
            enc["image_writer_threads"] = int(self.cfg.image_writer_threads)
            enc["image_writer_processes"] = int(self.cfg.image_writer_processes)
        return enc

    def _dataset_episode_entries(self) -> List[dict]:
        episodes = getattr(getattr(self._ds, "meta", None), "episodes", None)
        if episodes is None:
            return []
        entries: List[dict] = []
        try:
            iterator = iter(episodes)
        except TypeError:
            return []
        for ep in iterator:
            try:
                episode_index = int(ep["episode_index"])
            except Exception:
                continue
            tasks = ep.get("tasks") or []
            if isinstance(tasks, str):
                task = tasks
            else:
                task = str(tasks[0]) if tasks else self.cfg.task
            length = ep.get("length")
            if length is None:
                length = int(ep.get("dataset_to_index", 0)) - int(ep.get("dataset_from_index", 0))
            entries.append(
                {
                    "episode": episode_index,
                    "outcome": "unknown",
                    "task": task,
                    "frames": int(length),
                    "source": "recovered",
                    "t": None,
                    "recovered": True,
                }
            )
        return entries

    def _read_outcome_entries(self) -> List[dict]:
        if not os.path.exists(self._outcomes_path):
            return []
        entries: List[dict] = []
        try:
            with open(self._outcomes_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed outcome row in %s", self._outcomes_path)
        except Exception as e:
            logger.error("could not read outcome sidecar: %s", e)
        return entries

    def _rewrite_outcome_entries(self, entries: List[dict]) -> None:
        os.makedirs(self._root, exist_ok=True)
        tmp_path = f"{self._outcomes_path}.tmp"
        with open(tmp_path, "w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
        os.replace(tmp_path, self._outcomes_path)

    def _ensure_outcome_sidecar_complete(self) -> None:
        recovered = {int(e["episode"]): e for e in self._dataset_episode_entries()}
        if not recovered:
            return
        entries_by_episode: Dict[int, dict] = {}
        for entry in self._read_outcome_entries():
            try:
                entries_by_episode[int(entry["episode"])] = entry
            except Exception:
                logger.warning("skipping outcome row without a valid episode index in %s", self._outcomes_path)
        missing = sorted(set(recovered) - set(entries_by_episode))
        if not missing:
            return
        for episode_index in missing:
            entries_by_episode[episode_index] = recovered[episode_index]
        ordered = [entries_by_episode[i] for i in sorted(entries_by_episode)]
        try:
            self._rewrite_outcome_entries(ordered)
            logger.warning(
                "recovered %d missing outcome sidecar row(s) from dataset metadata: episodes %s",
                len(missing), missing,
            )
        except Exception as e:
            logger.error("could not repair outcome sidecar: %s", e)

    # ------------------------------------------------------- RL per-frame features
    def _rl_config_path(self) -> str:
        return os.path.join(self._root, "rl_config.json")

    def _read_rl_config_on_resume(self) -> None:
        """Continuing an existing dataset must reuse its reward scheme — load the
        sidecar and override the (possibly different) current config."""
        path = self._rl_config_path()
        if self.cfg.resume and os.path.exists(path):
            try:
                rc = json.load(open(path))
                self.cfg.rl_features = bool(rc.get("rl_features", self.cfg.rl_features))
                self.cfg.reward_mode = str(rc.get("reward_mode", self.cfg.reward_mode))
                self.cfg.discount_factor = float(rc.get("discount_factor", self.cfg.discount_factor))
                logger.info("resume: inheriting RL settings %s", rc)
            except Exception as e:
                logger.error("could not read rl_config.json on resume: %s", e)

    def _write_rl_config_once(self) -> None:
        """Persist the reward scheme next to the dataset so resume can inherit it."""
        path = self._rl_config_path()
        if os.path.exists(path):
            return
        try:
            os.makedirs(self._root, exist_ok=True)
            json.dump({"rl_features": bool(getattr(self.cfg, "rl_features", False)),
                       "reward_mode": getattr(self.cfg, "reward_mode", "sparse"),
                       "discount_factor": float(getattr(self.cfg, "discount_factor", 0.99))},
                      open(path, "w"), indent=2)
        except Exception as e:
            logger.error("could not write rl_config.json: %s", e)

    def _frame_features(self, n_frames: int, outcome: Optional[str]) -> Optional[dict]:
        """Per-frame success/reward/mc_return for one episode, or None when disabled."""
        if not getattr(self.cfg, "rl_features", False):
            return None
        try:
            from abcdl.rewards import compute_frame_features
        except ImportError as e:
            raise RuntimeError("rl_features needs the abcdl package — pip install -e '.[abcdl]'") from e
        return compute_frame_features(
            n_frames, success=(outcome == "success"),
            mode=getattr(self.cfg, "reward_mode", "sparse"),
            discount=float(getattr(self.cfg, "discount_factor", 0.99)),
        )

    # ------------------------------------------------------------------ lifecycle
    def open(self, sample_frame: dict) -> None:
        """Open the dataset using ``sample_frame`` to derive the feature schema."""
        # On resume, inherit the dataset's original RL settings so the per-frame
        # reward/return scheme stays consistent across a continued collection.
        self._read_rl_config_on_resume()
        self._features = self._build_features(sample_frame)
        if self._abcdl:
            os.makedirs(self._root, exist_ok=True)
            if self.cfg.resume and os.path.isdir(self._root):
                self._n_episodes = sum(
                    1 for d in os.listdir(self._root)
                    if d.startswith("episode_")
                    and os.path.exists(os.path.join(self._root, d, "states_actions.bin"))
                )
                logger.info("abcdl dataset resuming at %s (%d episodes)", self._root, self._n_episodes)
            else:
                logger.info("abcdl dataset at %s (format=abcdl, size=%d)",
                            self._root, int(getattr(self.cfg, "abcdl_size", 224)))
            self._initial_episodes = self._n_episodes  # authoritative pre-session count
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            return
        if not self._mock:
            LeRobotDataset = _import_lerobot_dataset()
            if self.cfg.resume and os.path.isdir(self._root):
                enc = self._dataset_encoding_kwargs()
                try:
                    self._ds = LeRobotDataset(self.cfg.repo_id, root=self._root, **enc)
                except TypeError:
                    logger.warning("LeRobotDataset() rejected encoding kwargs %s; using defaults", sorted(enc))
                    self._ds = LeRobotDataset(self.cfg.repo_id, root=self._root)
                self._n_episodes = int(
                    getattr(self._ds, "num_episodes", getattr(getattr(self._ds, "meta", None), "total_episodes", 0))
                )
                self._ensure_outcome_sidecar_complete()
                logger.info(
                    "dataset resuming at %s (%d existing episodes, vcodec=%s, batch=%s)",
                    self._root, self._n_episodes, self.cfg.vcodec, self.cfg.batch_encoding_size,
                )
            else:
                enc = self._create_encoding_kwargs()
                try:
                    self._ds = LeRobotDataset.create(
                        repo_id=self.cfg.repo_id,
                        fps=self.cfg.fps,
                        features=self._features,
                        root=self._root,
                        robot_type=self.cfg.robot_type,
                        use_videos=self.cfg.use_videos,
                        **enc,
                    )
                except TypeError:  # older/newer LeRobot without these kwargs — fall back
                    logger.warning("LeRobot.create() rejected encoding kwargs %s; using defaults", sorted(enc))
                    self._ds = LeRobotDataset.create(
                        repo_id=self.cfg.repo_id,
                        fps=self.cfg.fps,
                        features=self._features,
                        root=self._root,
                        robot_type=self.cfg.robot_type,
                        use_videos=self.cfg.use_videos,
                    )
                logger.info(
                    "dataset created at %s (repo_id=%s, vcodec=%s, batch=%s)",
                    self._root, self.cfg.repo_id, self.cfg.vcodec, self.cfg.batch_encoding_size,
                )
        else:
            if self.cfg.resume:
                self._n_episodes = self._existing_outcome_count()
            logger.info("MOCK writer (repo_id=%s); features=%s", self.cfg.repo_id, sorted(self._features))

        self._initial_episodes = self._n_episodes  # authoritative pre-session count
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _existing_outcome_count(self) -> int:
        try:
            with open(self._outcomes_path) as fh:
                return sum(1 for line in fh if line.strip())
        except FileNotFoundError:
            return 0
        except Exception as e:
            logger.warning("could not count existing outcomes at %s: %s", self._outcomes_path, e)
            return 0

    # ------------------------------------------------------------------ submit
    def submit(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        """Enqueue a complete episode (list of frame dicts) for background saving."""
        if self._finalized:
            raise RuntimeError("dataset writer has already been finalized")
        if frames:
            self._queue.put((frames, outcome, task))
            with self._lock:
                self._n_submitted += 1

    @property
    def queue_depth(self) -> int:
        """Episodes WAITING in the queue (not counting the one being saved)."""
        return self._queue.qsize()

    @property
    def saving(self) -> bool:
        """True while the worker is encoding/writing an episode (qsize is 0 then)."""
        with self._lock:
            return self._saving

    @property
    def pending_total(self) -> int:
        """Episodes not yet on disk: queued + the one currently being saved."""
        with self._lock:
            return self._queue.qsize() + (1 if self._saving else 0)

    @property
    def progress(self) -> dict:
        """Detailed writer state for the GUI: how much is saved, what's encoding now,
        and how much is waiting. One worker by design (a LeRobotDataset is single-writer)."""
        with self._lock:
            return {
                "workers": 1,
                "saved": self._n_episodes,
                "submitted": self._n_submitted,
                "saving": self._saving,
                "saving_index": self._saving_index,
                "saving_frames": self._saving_frames,
                "queued": self._queue.qsize(),
                "finalized": self._finalized,
            }

    @property
    def num_episodes(self) -> int:
        with self._lock:
            return self._n_episodes

    @property
    def total_episodes(self) -> int:
        """Episodes in the WHOLE dataset: pre-session (resume) + saved this session."""
        with self._lock:
            return max(self._n_episodes, self._initial_episodes)

    @property
    def new_episodes(self) -> int:
        """Episodes saved THIS session (excludes what a resumed dataset already had)."""
        with self._lock:
            return max(0, self._n_episodes - self._initial_episodes)

    @property
    def finalized(self) -> bool:
        return self._finalized

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            frames, outcome, task = item
            with self._lock:
                self._saving = True
                self._saving_index = self._n_episodes
                self._saving_frames = len(frames)
            try:
                self._save_episode(frames, outcome, task)
            except Exception as e:
                logger.error("episode save failed: %s", e)
            finally:
                with self._lock:
                    self._saving = False
                    self._saving_index = None
                    self._saving_frames = 0
                self._queue.task_done()

    def _save_episode(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        free = self._free_gb()
        if free < self.cfg.min_free_gb:
            self.low_disk = True
            logger.warning("LOW DISK: %.1f GB free (< %s GB) — episode NOT saved", free, self.cfg.min_free_gb)
            return
        self.low_disk = False
        # Per-frame RL signals from the episode outcome (None when disabled).
        ff = self._frame_features(len(frames), outcome)
        if not self._mock:
            if self._abcdl:
                self._save_episode_abcdl(frames, task, self._n_episodes, ff)
            else:
                for i, f in enumerate(frames):
                    frame = {f"observation.images.{k}": v for k, v in f["images"].items()}
                    for key, val in f.items():
                        if key in ("images", "task"):
                            continue
                        frame[key] = np.asarray(val, dtype=np.float32)
                    if ff:
                        for k, v in ff.items():
                            frame[k] = np.asarray([v[i]], dtype=np.float32)
                    frame["task"] = task
                    self._ds.add_frame(frame)
                self._ds.save_episode()
        self._write_rl_config_once()
        with self._lock:
            episode_index = self._n_episodes
            self._n_episodes += 1
        self._record_outcome(episode_index, outcome, task, len(frames))
        logger.info("saved episode #%d (%d frames, outcome=%s)", episode_index, len(frames), outcome)

    def _save_episode_abcdl(self, frames: List[dict], task: str, ep_index: int,
                            frame_features: Optional[dict] = None) -> None:
        """Write one episode as an abcdl dir via abcdl.EpisodeWriter (images square-resized)."""
        import cv2

        try:
            from abcdl.writer import EpisodeWriter
        except ImportError as e:
            raise RuntimeError("format: abcdl needs the abcdl package — pip install -e '.[abcdl]'") from e

        size = int(getattr(self.cfg, "abcdl_size", 224))
        out_dir = os.path.join(self._root, f"episode_{ep_index:06d}")
        tick = int(1e9 / max(1, int(self.cfg.fps)))
        w = EpisodeWriter(out_dir, formats=("abcdl",), fps=int(self.cfg.fps),
                          cameras=list(self.image_keys))
        for i, f in enumerate(frames):
            imgs = {
                k: cv2.resize(np.asarray(v), (size, size), interpolation=cv2.INTER_AREA)
                for k, v in f["images"].items()
            }
            w.add_frame(i * tick, np.asarray(f["observation.state"], np.float64),
                        np.asarray(f["action"], np.float64), imgs)
        w.save(task=task, frame_features=frame_features)

    def _record_outcome(self, episode_index: int, outcome: Optional[str], task: str, n_frames: int) -> None:
        entry = {
            "episode": episode_index,
            "outcome": outcome,
            "task": task,
            "frames": n_frames,
            "source": self.cfg.record_source,
            "t": time.time(),
        }
        try:
            os.makedirs(self._root, exist_ok=True)
            with open(self._outcomes_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("could not write outcome sidecar: %s", e)

    # ------------------------------------------------------------------ shutdown
    def finalize(self) -> None:
        """Drain the queue, stop the worker, then close the LeRobot dataset."""
        with self._finalize_lock:
            if self._finalized:
                return
            self._stop.set()
            if self._worker is not None:
                self._worker.join(timeout=600.0)
            if not self._mock and self._ds is not None and self._n_episodes > 0:
                self._flush_pending_batch()
                try:
                    self._ds.finalize()
                    logger.info("dataset finalized (parquet/metadata closed)")
                except Exception as e:
                    logger.error("dataset finalize failed: %s", e)
            self._finalized = True

    def _flush_pending_batch(self) -> None:
        """With batch_encoding_size > 1, LeRobot defers video encoding and its finalize()
        does NOT flush the trailing (< batch) episodes — they'd stay as temp PNGs and the
        dataset would be part-video/part-images. Encode that remainder here before closing."""
        ds = self._ds
        pending = int(getattr(ds, "episodes_since_last_encoding", 0) or 0)
        if pending <= 0 or not hasattr(ds, "_batch_save_episode_video"):
            return
        try:
            end = int(ds.num_episodes)
            ds._batch_save_episode_video(end - pending, end)
            ds.episodes_since_last_encoding = 0
            logger.info("flushed %d trailing episode(s) to video before finalize", pending)
        except Exception as e:
            logger.error("could not flush pending batch encode (%d episodes): %s", pending, e)
