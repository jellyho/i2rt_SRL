"""Record the action chunks a policy sampled, alongside the dataset.

The live overlay shows the policy's spread as it happens and then it is gone. This keeps the
numbers, so "it hesitated there" can be reopened, re-drawn, and argued with later — and so a
rendered video can show what the policy ACTUALLY predicted at the time rather than what the
current checkpoint would predict now, which is a different question.

Two facts shape the format.

**The samples change once per chunk, not once per frame.** The broker infers when the chunk it
is holding runs out, so at a 30-step horizon and 30 Hz that is once a second; the thirty frames
in between share one identical set. Storing per frame would be thirty times the bytes for the
same information, so a row is written when the samples CHANGE and carries the frame index it
started at.

**They are a diagnostic, not a target.** float16 halves the file for a quantity nobody trains
on and whose interesting property — how far apart the chunks are — survives three decimal
places comfortably.

This is a sidecar rather than a LeRobot feature on purpose. The dataset's schema is fixed
before recording starts (see ``Recorder._sample_frame``), and an ``[N, horizon, action_dim]``
column would have to know the policy's horizon at that point — reintroducing exactly the
number-to-keep-in-sync that the broker was changed to stop needing. The sidecar learns the
shape from the first row it is given.

    <dataset>/action_samples/episode-000003.npz
        frame_index  int32   [K]           frame the sample set became current at
        samples      float16 [K, N, H, A]  the chunks the policy drew there
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DIR_NAME = "action_samples"


def episode_path(dataset_dir: str, episode_index: int) -> str:
    return os.path.join(dataset_dir, DIR_NAME, f"episode-{episode_index:06d}.npz")


class EpisodeSampleLog:
    """Collects one episode's sample sets, skipping the repeats.

    Fed every frame; keeps a row only when the samples differ from the last one kept. Identity
    is checked on the array itself rather than on a counter, because the runner republishes the
    same object for every frame of a chunk and a counter would be one more thing to keep in
    step with it.
    """

    def __init__(self) -> None:
        self._frames: List[int] = []
        self._samples: List[np.ndarray] = []
        self._chosen: List[int] = []
        self._scores: List[Optional[np.ndarray]] = []
        self._last: Optional[np.ndarray] = None

    def add(self, frame_index: int, samples: Optional[np.ndarray], *,
            chosen: int = 0, scores: Optional[np.ndarray] = None) -> None:
        """Record a sample set, if it differs from the last one kept.

        ``chosen`` is which candidate was executed. It is 0 for plain multi-sampling, where the
        executed chunk is drawn first by construction, but NOT for a critic that picks
        best-of-N -- and drawing the wrong one as "what the robot did" is the sort of error that
        looks like a correct picture. ``scores`` is the critic's Q per candidate when there is
        one, so a view can shade the fan by value instead of treating every candidate alike.
        """
        if samples is None:
            return
        samples = np.asarray(samples)
        if samples.ndim != 3:
            logger.debug("ignoring action samples of shape %s", samples.shape)
            return
        if self._last is not None and samples.shape == self._last.shape and np.array_equal(samples, self._last):
            return
        self._frames.append(int(frame_index))
        self._samples.append(samples.astype(np.float16))
        self._chosen.append(int(chosen))
        self._scores.append(None if scores is None else np.asarray(scores, dtype=np.float16).reshape(-1))
        self._last = samples

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def rows(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames, self._samples, self._last = [], [], None
        self._chosen, self._scores = [], []

    def save(self, dataset_dir: str, episode_index: int) -> Optional[str]:
        """Write the episode's samples; returns the path, or None when there were none.

        Never raises: losing a diagnostic must not cost the episode it describes.
        """
        if not self._frames:
            return None
        shapes = {s.shape for s in self._samples}
        if len(shapes) > 1:
            # A changed sample count or horizon mid-episode. Keep the majority shape rather
            # than write something no reader can stack.
            logger.warning("action samples changed shape mid-episode (%s); keeping the first", shapes)
            first = self._samples[0].shape
            keep = [i for i, s in enumerate(self._samples) if s.shape == first]
            self._frames = [self._frames[i] for i in keep]
            self._samples = [self._samples[i] for i in keep]
            self._chosen = [self._chosen[i] for i in keep]
            self._scores = [self._scores[i] for i in keep]
        path = episode_path(dataset_dir, episode_index)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            arrays = {
                "frame_index": np.asarray(self._frames, dtype=np.int32),
                "samples": np.stack(self._samples),
                "chosen": np.asarray(self._chosen, dtype=np.int32),
            }
            # Only when a critic actually scored them -- an array of zeros would read as
            # "every candidate is worthless" rather than "nobody was asked".
            if all(sc is not None for sc in self._scores) and self._scores:
                arrays["scores"] = np.stack(self._scores)
            np.savez_compressed(path, **arrays)
            return path
        except Exception as e:
            logger.error("could not write action samples for episode %d: %s", episode_index, e)
            return None


def load(dataset_dir: str, episode_index: int) -> Optional[dict]:
    """``{"frame_index": [K], "samples": [K, N, H, A]}`` for an episode, or None."""
    path = episode_path(dataset_dir, episode_index)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            out = {"frame_index": data["frame_index"], "samples": data["samples"]}
            # Older logs predate these; a reader must not have to know which vintage it has.
            out["chosen"] = (data["chosen"] if "chosen" in data.files
                             else np.zeros(len(out["frame_index"]), np.int32))
            out["scores"] = data["scores"] if "scores" in data.files else None
            return out
    except Exception as e:
        logger.error("could not read action samples at %s: %s", path, e)
        return None


def samples_at(log: dict, frame_index: int) -> Optional[np.ndarray]:
    """The sample set in force at ``frame_index``.

    Rows mark where a set became current, so the one in force is the last row at or before the
    frame — searching for an exact match would find nothing for 29 frames out of 30.
    """
    if not log:
        return None
    frames = np.asarray(log["frame_index"])
    position = int(np.searchsorted(frames, frame_index, side="right")) - 1
    if position < 0:
        return None
    return np.asarray(log["samples"][position], dtype=np.float32)


def row_at(log: dict, frame_index: int) -> Optional[dict]:
    """Everything recorded for the set in force at ``frame_index``.

    ``{"samples": [N, H, A], "chosen": int, "scores": [N] | None}`` -- the chosen index matters
    because a critic picking best-of-N does not put its choice first, and drawing candidate 0
    as "what the robot did" produces a picture that looks right and is not.
    """
    if not log:
        return None
    frames = np.asarray(log["frame_index"])
    position = int(np.searchsorted(frames, frame_index, side="right")) - 1
    if position < 0:
        return None
    scores = log.get("scores")
    return {
        "samples": np.asarray(log["samples"][position], dtype=np.float32),
        "chosen": int(np.asarray(log.get("chosen", np.zeros(len(frames))))[position]),
        "scores": None if scores is None else np.asarray(scores[position], dtype=np.float32),
    }
