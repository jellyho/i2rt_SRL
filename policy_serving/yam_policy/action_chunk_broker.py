"""ActionChunkBroker — serve a predicted action chunk one step at a time.

Mirrors ``openpi_client.action_chunk_broker``. Wrap any :class:`BasePolicy` (e.g.
a :class:`WebsocketClientPolicy`); the broker calls the inner policy only every
``action_horizon`` steps and returns one action per step from the cached chunk,
so you query the (possibly remote) policy ~once per N control ticks.

The inner policy must return ``{"actions": ndarray(action_horizon, action_dim), ...}``.
Each ``infer`` returns the same dict with array fields sliced to the current step.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .yam_contract import validate_action_chunk


def _slice_step(results: Dict, step: int, created_at: float) -> Dict:
    sliced = {k: (v[step, ...] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in results.items()}
    sliced["broker_timing"] = {
        "chunk_age_ms": (time.monotonic() - created_at) * 1000.0,
        "chunk_step": step,
    }
    return sliced


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._cur_step = 0
        self._last_results: Dict | None = None
        self._chunk_created_at = 0.0

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = validate_action_chunk(self._policy.infer(obs), execution_horizon=self._action_horizon)
            self._cur_step = 0
            self._chunk_created_at = time.monotonic()

        results = _slice_step(self._last_results, self._cur_step, self._chunk_created_at)

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._last_results = None
        self._cur_step = 0
        self._chunk_created_at = 0.0
        self._policy.reset()


class AsyncActionChunkBroker(BasePolicy):
    """Action-chunk broker that **prefetches the next chunk** in a background thread.

    Same one-step-at-a-time interface as :class:`ActionChunkBroker`, but when the
    current chunk is ``prefetch_at`` steps in, it kicks off the next ``infer`` on a
    worker thread (using the most recent obs). By the time the chunk is exhausted the
    next one is usually ready, so the (possibly remote) inference latency is hidden
    instead of stalling the control loop at every chunk boundary.
    """

    def __init__(self, policy: BasePolicy, action_horizon: int, prefetch_at: Optional[int] = None) -> None:
        self._policy = policy
        self._h = int(action_horizon)
        self._prefetch_at = int(prefetch_at) if prefetch_at is not None else max(self._h - 2, 1)
        self._exec = ThreadPoolExecutor(max_workers=1)  # serializes inference calls
        self._cur: Dict | None = None
        self._step = 0
        self._next = None  # Future for the upcoming chunk
        self._last_obs: Dict | None = None
        self._generation = 0
        self._chunk_created_at = 0.0
        self._infer_lock = threading.Lock()

    def _infer_chunk(self, obs: Dict, generation: int) -> tuple[int, Dict, float]:
        # A reset can invalidate a running prefetch. Serialize access to the
        # underlying websocket/model so the fresh post-reset request cannot race it.
        with self._infer_lock:
            result = validate_action_chunk(self._policy.infer(obs), execution_horizon=self._h)
        return generation, result, time.monotonic()

    def infer(self, obs: Dict) -> Dict:
        self._last_obs = obs
        if self._cur is None:
            _, self._cur, self._chunk_created_at = self._infer_chunk(obs, self._generation)
            self._step = 0

        out = _slice_step(self._cur, self._step, self._chunk_created_at)
        self._step += 1

        if self._step == self._prefetch_at and self._next is None:
            self._next = self._exec.submit(self._infer_chunk, self._last_obs, self._generation)

        if self._step >= self._h:
            prefetched = self._next.result() if self._next is not None else None
            self._next = None
            if prefetched is not None and prefetched[0] == self._generation:
                _, self._cur, self._chunk_created_at = prefetched
            else:
                _, self._cur, self._chunk_created_at = self._infer_chunk(self._last_obs, self._generation)
            self._step = 0

        return out

    def reset(self) -> None:
        self._generation += 1
        if self._next is not None:
            self._next.cancel()
            self._next = None
        self._cur = None
        self._step = 0
        self._chunk_created_at = 0.0
        self._policy.reset()

    def close(self) -> None:
        self.reset()
        self._exec.shutdown(wait=False, cancel_futures=True)
