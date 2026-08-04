"""ActionChunkBroker — serve a predicted action chunk one step at a time.

Wrap any :class:`BasePolicy` (e.g. a :class:`WebsocketClientPolicy`); the broker calls the
inner policy only when the cached chunk runs out and returns one action per step from it,
so you query the (possibly remote) policy once per chunk instead of every control tick.

**The horizon is not configured — it is whatever the policy returned.** ``openpi_client``'s
broker takes ``action_horizon`` as a constructor argument, which means the number has to be
known on both sides and kept in sync by hand. It never is: a checkpoint trained with
``action_horizon=30`` served to a client defaulting to 16 raises nothing, it just throws
away the back half of every chunk and re-infers twice as often. Reading the length off
``actions.shape[0]`` removes the setting, and with it the whole class of mismatch — the same
client then drives any policy, whatever chunk size it was trained with.

The inner policy must return ``{"actions": ndarray(chunk, action_dim), ...}``. Each
``infer`` returns that dict with array fields sliced to the current step.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


def _slice_step(results: Dict, step: int) -> Dict:
    return {k: (v[step, ...] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in results.items()}


def chunk_len(results: Dict) -> int:
    """How many steps the policy just handed us — the leading dim of ``actions``.

    1 for a policy that returns a single, unchunked action, so such a policy still works
    (it just re-infers every tick, which is exactly what it means)."""
    actions = results.get("actions")
    if isinstance(actions, np.ndarray) and actions.ndim > 1:
        return int(actions.shape[0])
    return 1


def is_chunked(results: Dict) -> bool:
    """Whether ``actions`` has a leading step axis to index into.

    An unchunked ``(action_dim,)`` response must be passed through whole — slicing it by
    step would hand the robot a single scalar joint target."""
    actions = results.get("actions")
    return isinstance(actions, np.ndarray) and actions.ndim > 1


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy) -> None:
        self._policy = policy
        self._cur_step = 0
        self._chunk = 0  # length of the chunk in hand; 0 = nothing cached
        self._chunked = True
        self._last_results: Dict | None = None

    @property
    def action_horizon(self) -> int:
        """The chunk size last observed (0 before the first inference)."""
        return self._chunk

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._chunk = chunk_len(self._last_results)
            self._chunked = is_chunked(self._last_results)
            self._cur_step = 0

        results = _slice_step(self._last_results, self._cur_step) if self._chunked else self._last_results

        self._cur_step += 1
        if self._cur_step >= self._chunk:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._last_results = None
        self._cur_step = 0
        self._policy.reset()


class AsyncActionChunkBroker(BasePolicy):
    """Action-chunk broker that **prefetches the next chunk** in a background thread.

    Same one-step-at-a-time interface as :class:`ActionChunkBroker`, but ``prefetch_lead``
    steps before the current chunk runs out it kicks off the next ``infer`` on a worker
    thread (using the most recent obs). By the time the chunk is exhausted the next one is
    usually ready, so the (possibly remote) inference latency is hidden instead of stalling
    the control loop at every chunk boundary.

    The lead is counted back from the end of whatever chunk arrived, so this adapts to the
    policy's chunk size like the synchronous broker does.
    """

    def __init__(self, policy: BasePolicy, prefetch_lead: int = 2) -> None:
        self._policy = policy
        self._lead = max(1, int(prefetch_lead))
        self._exec = ThreadPoolExecutor(max_workers=1)  # serializes inference calls
        self._cur: Dict | None = None
        self._chunk = 0
        self._chunked = True
        self._step = 0
        self._next = None  # Future for the upcoming chunk
        self._last_obs: Dict | None = None

    @property
    def action_horizon(self) -> int:
        return self._chunk

    def _adopt(self, results: Dict) -> None:
        self._cur = results
        self._chunk = chunk_len(results)
        self._chunked = is_chunked(results)
        self._step = 0

    def infer(self, obs: Dict) -> Dict:
        self._last_obs = obs
        if self._cur is None:
            self._adopt(self._policy.infer(obs))

        out = _slice_step(self._cur, self._step) if self._chunked else self._cur
        self._step += 1

        # Start the next inference `lead` steps from the end of THIS chunk (never before
        # step 1, so a 1- or 2-step chunk still prefetches exactly once).
        if self._next is None and self._step >= max(1, self._chunk - self._lead):
            self._next = self._exec.submit(self._policy.infer, self._last_obs)

        if self._step >= self._chunk:
            try:
                nxt = self._next.result() if self._next is not None else self._policy.infer(self._last_obs)
            finally:
                self._next = None
            self._adopt(nxt)

        return out

    def reset(self) -> None:
        if self._next is not None:
            try:
                self._next.result(timeout=0)
            except Exception:
                pass
            self._next = None
        self._cur = None
        self._chunk = 0
        self._step = 0
        self._policy.reset()

    def close(self) -> None:
        self._exec.shutdown(wait=False)
