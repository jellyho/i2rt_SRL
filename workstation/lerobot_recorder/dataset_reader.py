"""Read a recorded LeRobot dataset for replay.

Loads a LeRobot **v3.0** dataset and exposes, per episode, the per-frame action
(14-d, both arms) for driving the robot and an optional image for the GUI. The
few version-sensitive accessors are isolated here (like ``dataset_writer.py``).

**Only ONE episode's frames are held in memory at a time.** ``load()`` reads just the
lightweight metadata (episode count, per-episode length, fps, feature schema) -- not the
frames. The frame accessors then lazily open the *single* episode they are asked for
(``LeRobotDataset(..., episodes=[e])``) and cache it, replacing whatever episode was open
before. A 347-episode set costs ~120 MB this way (metadata + one episode) instead of ~1.1 GB
for the whole thing -- replay/render only ever look at one episode, so there is no reason to
load the rest.

``mock=True`` synthesizes a few episodes so the replay UI/logic runs without
``lerobot`` or a real dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder.config import ACTION_DIM

_MOCK_EPISODES = 3
_MOCK_LENGTH = 120


def _to_uint8_hwc(t: Any) -> Optional[np.ndarray]:
    """Coerce a torch/np image (CHW or HWC, float [0,1] or uint8) to a contiguous HxWx3 uint8."""
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class DatasetReader:
    def __init__(self, repo_id: str, root: str, display_cam: str = "agentview", mock: bool = False) -> None:
        self.repo_id = repo_id
        self.root = root
        self.display_cam = display_cam
        self.mock = mock
        self._meta = None  # LeRobotDatasetMetadata -- frame-free (episode lengths, fps, features)
        self._ds = None  # the ONE currently-loaded episode's LeRobotDataset (episodes=[_loaded_ep])
        self._loaded_ep: Optional[int] = None
        self._fps = 60
        self._total_eps = 0
        self._ep_lengths: Dict[int, int] = {}
        self._features: Dict[str, Any] = {}

    # ------------------------------------------------------------------ load (metadata only)
    def load(self) -> None:
        if self.mock:
            self._fps = 60
            self._total_eps = _MOCK_EPISODES
            self._ep_lengths = {e: _MOCK_LENGTH for e in range(_MOCK_EPISODES)}
            return
        try:
            from lerobot.datasets import LeRobotDatasetMetadata
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from workstation.lerobot_recorder.dataset_writer import dataset_dir

        # The dataset lives in <root>/<name> (same rule the recorder writes with). Metadata only --
        # no frames, no video index for every episode; the frame accessors load one episode lazily.
        self._root_dir = dataset_dir(self.root, self.repo_id)
        self._meta = LeRobotDatasetMetadata(self.repo_id, root=self._root_dir)
        self._fps = int(getattr(self._meta, "fps", 60) or 60)
        self._total_eps = int(getattr(self._meta, "total_episodes", 0) or 0)
        self._features = dict(getattr(self._meta, "features", {}) or {})
        # Per-episode lengths from the metadata's episodes table (small: one row per episode).
        eps = getattr(self._meta, "episodes", None)
        self._ep_lengths = {}
        if eps is not None:
            try:
                for e, ln in zip(eps["episode_index"], eps["length"], strict=False):
                    self._ep_lengths[int(e)] = int(ln)
            except (KeyError, TypeError):
                # Older/newer layouts: fall back to indexing rows.
                for i in range(self._total_eps):
                    row = eps[i]
                    self._ep_lengths[int(row.get("episode_index", i))] = int(row.get("length", 0))

    def _ensure_episode(self, episode: int) -> None:
        """Make ``self._ds`` hold ONLY ``episode``'s frames (0-indexed), loading it if a different
        episode (or none) is currently open. Replacing ``self._ds`` frees the previous episode."""
        if self._loaded_ep == episode and self._ds is not None:
            return
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self._ds = None  # drop the previous episode before opening the next (halves peak memory)
        self._ds = LeRobotDataset(self.repo_id, root=self._root_dir, episodes=[episode])
        self._loaded_ep = episode

    # ------------------------------------------------------------------ queries
    @property
    def num_episodes(self) -> int:
        return self._total_eps

    @property
    def fps(self) -> int:
        return self._fps

    def episode_length(self, episode: int) -> int:
        return int(self._ep_lengths.get(episode, 0))

    def get_action(self, episode: int, frame: int) -> np.ndarray:
        if self.mock:
            t = frame / 30.0
            return (0.3 * np.sin(t + np.arange(ACTION_DIM))).astype(np.float32)
        self._ensure_episode(episode)
        act = self._ds.hf_dataset[frame]["action"]  # avoids video decode
        return np.asarray(act, dtype=np.float32).reshape(-1)

    def camera_keys(self) -> List[str]:
        """Camera names available in the dataset (the ``observation.images.<name>`` suffixes)."""
        if self.mock:
            return ["agentview"]
        return sorted(
            k.split("observation.images.", 1)[1] for k in self._features if k.startswith("observation.images.")
        )

    def get_image(self, episode: int, frame: int) -> Optional[np.ndarray]:
        """Return an HxWx3 uint8 image (the configured display camera) for display, or None."""
        return self.get_images(episode, frame).get(self.display_cam)

    def get_images(self, episode: int, frame: int) -> Dict[str, np.ndarray]:
        """Return ``{camera_name: HxWx3 uint8}`` for every camera at this frame (one video decode)."""
        if self.mock:
            img = np.zeros((120, 160, 3), dtype=np.uint8)
            x = (frame * 3) % 160
            img[:, max(0, x - 6) : x + 6, :] = 200
            return {"agentview": img}
        self._ensure_episode(episode)
        out: Dict[str, np.ndarray] = {}
        try:
            row = self._ds[frame]  # decodes video frames
        except Exception:
            return out
        for key, val in row.items():
            if not key.startswith("observation.images."):
                continue
            img = _to_uint8_hwc(val)
            if img is not None and img.ndim == 3:
                out[key.split("observation.images.", 1)[1]] = img
        return out

    def get_state(self, episode: int, frame: int) -> Optional[np.ndarray]:
        """Return the frame's ``observation.state`` vector (no video decode), or None."""
        if self.mock:
            return None
        self._ensure_episode(episode)
        try:
            st = self._ds.hf_dataset[frame].get("observation.state")
        except Exception:
            return None
        return None if st is None else np.asarray(st, dtype=np.float32).reshape(-1)

    def get_scalar(self, episode: int, frame: int, key: str) -> Optional[float]:
        """Return a per-frame scalar feature (e.g. ``observation.control_mode``, ``homing``),
        or None if the feature is absent. No video decode."""
        if self.mock:
            return None
        self._ensure_episode(episode)
        try:
            val = self._ds.hf_dataset[frame].get(key)
        except Exception:
            return None
        if val is None:
            return None
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        return float(arr[0]) if arr.size else None

    def get_extra(self, episode: int, frame: int, key: str, shape: tuple) -> Optional[np.ndarray]:
        """Return a declared extra-feature column (e.g. ``action_samples``) at its declared
        shape (no video decode), or None if the frame/column is absent.

        Mirrors ``get_state``: the dataset writer stores declared extras flattened (see
        ``dataset_writer._to_lerobot_frame``), so the caller's own shape is what un-flattens it.
        """
        if self.mock:
            return None
        self._ensure_episode(episode)
        try:
            val = self._ds.hf_dataset[frame].get(key)
        except Exception:
            return None
        return None if val is None else np.asarray(val, dtype=np.float32).reshape(shape)

    def has_feature(self, key: str) -> bool:
        if self.mock:
            return False
        return key in self._features

    def get_control_mode_series(self, episode: int) -> Optional[np.ndarray]:
        """Return the per-frame ``observation.control_mode`` for a whole episode (1-D int-ish
        float array), or None if absent. One in-memory column read -- no video decode."""
        if self.mock:
            return None
        self._ensure_episode(episode)
        try:
            col = self._ds.hf_dataset["observation.control_mode"]  # this episode's frames only
        except Exception:
            return None
        out = np.empty(len(col), dtype=np.float32)
        for k, v in enumerate(col):
            a = np.asarray(v, dtype=np.float32).reshape(-1)
            out[k] = a[0] if a.size else 0.0
        return out
