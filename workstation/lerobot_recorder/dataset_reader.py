"""Read a recorded LeRobot dataset for replay.

Loads a LeRobot **v3.0** dataset and exposes, per episode, the per-frame action
(14-d, both arms) for driving the robot and an optional image for the GUI. The
few version-sensitive accessors are isolated here (like ``dataset_writer.py``).

``mock=True`` synthesizes a few episodes so the replay UI/logic runs without
``lerobot`` or a real dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder.config import ACTION_DIM


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
        self._ds = None
        self._ep_index: Dict[int, List[int]] = {}  # episode -> global frame indices
        self._fps = 60

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.mock:
            self._fps = 60
            self._ep_index = {e: list(range(e * 1000, e * 1000 + 120)) for e in range(3)}  # 3 episodes x 120 frames
            return
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from workstation.lerobot_recorder.dataset_writer import dataset_dir

        # The dataset lives in <root>/<name> (same rule the recorder writes with).
        self._ds = LeRobotDataset(self.repo_id, root=dataset_dir(self.root, self.repo_id))
        self._fps = int(getattr(self._ds, "fps", getattr(getattr(self._ds, "meta", None), "fps", 60)))
        # group global frame indices by episode (cheap column read; no video decode)
        col = self._ds.hf_dataset["episode_index"]
        self._ep_index = {}
        for gi, e in enumerate(col):
            self._ep_index.setdefault(int(e), []).append(gi)

    # ------------------------------------------------------------------ queries
    @property
    def num_episodes(self) -> int:
        return len(self._ep_index)

    @property
    def fps(self) -> int:
        return self._fps

    def episode_length(self, episode: int) -> int:
        return len(self._ep_index.get(episode, []))

    def get_action(self, episode: int, frame: int) -> np.ndarray:
        if self.mock:
            t = frame / 30.0
            return (0.3 * np.sin(t + np.arange(ACTION_DIM))).astype(np.float32)
        gi = self._ep_index[episode][frame]
        act = self._ds.hf_dataset[gi]["action"]  # avoids video decode
        return np.asarray(act, dtype=np.float32).reshape(-1)

    def camera_keys(self) -> List[str]:
        """Camera names available in the dataset (the ``observation.images.<name>`` suffixes)."""
        if self.mock:
            return ["agentview"]
        feats = getattr(self._ds, "features", None) or getattr(getattr(self._ds, "meta", None), "features", {}) or {}
        return sorted(k.split("observation.images.", 1)[1] for k in feats if k.startswith("observation.images."))

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
        gi = self._ep_index[episode][frame]
        out: Dict[str, np.ndarray] = {}
        try:
            row = self._ds[gi]  # decodes video frames
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
        gi = self._ep_index[episode][frame]
        try:
            st = self._ds.hf_dataset[gi].get("observation.state")
        except Exception:
            return None
        return None if st is None else np.asarray(st, dtype=np.float32).reshape(-1)
