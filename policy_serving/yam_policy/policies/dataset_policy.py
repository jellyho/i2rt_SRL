"""Replay a recorded episode by serving it as a policy.

Replay used to be its own stack: a second robot client, a second command path
(``RobotClient.command`` against ``run_robot_server wrapper``), its own ramp-to-first-frame,
and a GUI with none of the deployment safeguards. But replay differs from deployment in
exactly one thing — where the actions come from. Everything else it needs, deploy already
does better.

So the actions come from a dataset and nothing else changes:

    python -m yam_policy.serve \
        --policy yam_policy.policies.dataset_policy:DatasetPolicy \
        --config root=~/lerobot_data/yam_cable_tie_v4 --config episode=3

    robot/yam deploy                 # the SAME server deployment uses -- no `wrapper`
    workstation/yam-data deploy      # the same UI, live cameras, e-stop, takeover

What that inherits, none of which the old replay had: the follower smoother and joint-speed
clamp (so there is no hand-rolled ramp), human takeover on a handle button, the network
e-stop, the link-loss watchdog, leader mirroring, and the option to record the replayed run
as a dataset of its own.

**Only the action column is read.** Replay does not need the video, so this opens the parquet
files directly rather than a `LeRobotDataset` — no decoding, no lerobot dependency, and a
50-episode dataset loads in well under a second.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..base_policy import BasePolicy

logger = logging.getLogger(__name__)


class DatasetPolicy(BasePolicy):
    """Serves one episode's recorded actions, in order, as action chunks.

    Args:
        root: The dataset directory (the one holding `meta/` and `data/`).
        episode: Which `episode_index` to replay.
        chunk: Actions per reply. The client executes one per control tick and asks again,
            so this is only how often the two talk, not a horizon the robot commits to.
        speed: >1 replays faster by dropping frames, <1 slower by repeating them. The client
            ticks at a fixed rate, so changing the stream is what changes the speed.
        loop: Start again at frame 0 instead of holding at the end.
    """

    def __init__(
        self,
        root: str,
        episode: int = 0,
        chunk: int = 30,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self._root = Path(root).expanduser()
        self._episode = int(episode)
        self._chunk = max(1, int(chunk))
        self._speed = float(speed)
        self._loop = bool(loop)
        if self._speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")

        self._actions = self._load_episode(self._root, self._episode)
        if self._speed != 1.0:
            self._actions = self._resample(self._actions, self._speed)
        self._cursor = 0
        self._exhausted = False

        self._fps = self._dataset_fps(self._root) * self._speed
        self.action_horizon = self._chunk
        #: Merged into the handshake by serve.py. Beyond naming what is driving, the replay
        #: fields let the client line its past-demonstration overlay up with what the arm is
        #: doing: same dataset, same episode, same rate, without anyone selecting it by hand.
        self.policy_info = {
            "framework": "dataset-replay",
            "policy_name": f"{self._root.name}#{self._episode}",
            "checkpoint": str(self._root),
            "replay_dataset": self._root.name,
            "replay_episode": self._episode,
            "replay_fps": self._fps,
        }
        logger.info(
            "replaying %s episode %d: %d frames%s at %.2fx",
            self._root.name,
            self._episode,
            len(self._actions),
            " (looping)" if self._loop else "",
            self._speed,
        )

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_episode(root: Path, episode: int) -> np.ndarray:
        """Every `action` row of one episode, in frame order.

        Reads the parquet directly: a v3.0 dataset stores many episodes per file, so this
        filters on `episode_index` and sorts on `frame_index` rather than trusting row order.
        """
        import pyarrow.dataset as ds

        data_dir = root / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"{root} has no data/ directory — is it a LeRobot v3.0 dataset?")

        table = ds.dataset(data_dir, format="parquet").to_table(
            columns=["episode_index", "frame_index", "action"],
            filter=ds.field("episode_index") == episode,
        )
        if table.num_rows == 0:
            raise ValueError(f"episode {episode} is not in {root} (it has {DatasetPolicy._episodes(root)})")

        order = np.argsort(np.asarray(table.column("frame_index")))
        rows = table.column("action").to_pylist()
        return np.asarray([rows[i] for i in order], dtype=np.float32)

    @staticmethod
    def _dataset_fps(root: Path) -> float:
        """The rate the episode was recorded at, so the client can play its overlay to match."""
        try:
            return float(json.loads((root / "meta" / "info.json").read_text())["fps"])
        except Exception:
            logger.warning("could not read fps from %s/meta/info.json; assuming 30", root)
            return 30.0

    @staticmethod
    def _episodes(root: Path) -> str:
        try:
            total = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
            return f"episodes 0..{int(total) - 1}"
        except Exception:
            return "an unknown number of episodes"

    @staticmethod
    def _resample(actions: np.ndarray, speed: float) -> np.ndarray:
        """Nearest-frame resample. Slowing down repeats frames, which holds the arm rather
        than interpolating toward a pose the robot never recorded."""
        n_out = max(1, round(len(actions) / speed))
        idx = np.minimum((np.arange(n_out) * speed).astype(int), len(actions) - 1)
        return actions[idx]

    # ------------------------------------------------------------------ serving
    def infer(self, obs: Dict) -> Dict:
        """The next `chunk` actions. The observation is ignored — that is the whole point.

        At the end the last pose is repeated rather than the stream stopping: the client is
        driving a robot and needs something to hold, and the alternative (a short or empty
        chunk) would leave the followers on a stale command.
        """
        del obs
        remaining = len(self._actions) - self._cursor
        if remaining <= 0:
            if self._loop:
                self._cursor = 0
                remaining = len(self._actions)
            else:
                if not self._exhausted:
                    logger.info("episode %d finished; holding the final pose", self._episode)
                    self._exhausted = True
                held = np.repeat(self._actions[-1][None, :], self._chunk, axis=0)
                return {"actions": held, "replay_done": np.ones((self._chunk, 1), np.float32)}

        take = min(self._chunk, remaining)
        out = self._actions[self._cursor : self._cursor + take]
        self._cursor += take
        if take < self._chunk:  # pad the tail so every reply is the same length
            out = np.concatenate([out, np.repeat(out[-1][None, :], self._chunk - take, axis=0)])
        return {"actions": out.astype(np.float32), "replay_done": np.zeros((self._chunk, 1), np.float32)}

    def extra_features(self) -> Dict[str, List[int]]:
        """Recorded alongside the actions when the run is being written to a dataset, so a
        replayed episode can be told from where it ran out."""
        return {"replay_done": [1]}

    def reset(self) -> None:
        """Rewind. The deploy client calls this when a rollout starts, so each start replays
        the episode from the beginning without anything having to restart the server."""
        self._cursor = 0
        self._exhausted = False

    # ------------------------------------------------------------------ introspection
    @property
    def frames(self) -> int:
        return len(self._actions)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def obs_spec(self) -> Dict:
        """No cameras and no state: replay reads none of the observation, and saying so keeps
        the client from resizing frames for a policy that will not look at them."""
        return {"image_keys": {}}

    def __repr__(self) -> str:
        return f"DatasetPolicy({self._root.name}, episode={self._episode}, frames={self.frames})"
