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

**Extra per-step data rides along.** A policy may return more than actions -- a critic's Q per
step, the candidate chunks it chose between, anything it wants recorded. Those arrays are
sliced by the same rule: leading axis equal to the chunk means per-step, so step `i` gets row
`i`. Nothing here knows their names; the server declares those at handshake (see
``extra_features`` in the server metadata) and the recorder turns them into dataset columns.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


def _slice_step(results: Dict, step: int) -> Dict:
    """One step out of a chunked reply.

    Every array whose leading axis is the chunk is sliced, ``actions`` and whatever else the
    policy sent along -- Q values per step, candidate chunks, anything. That is the contract:
    an extra array is per-step data, indexed like the actions it accompanies. An array whose
    leading axis is NOT the chunk is passed through whole rather than mis-indexed, because
    slicing it would return a different thing entirely and look like data.
    """
    chunk = chunk_len(results)
    return {
        k: (v[step, ...] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == chunk else v)
        for k, v in results.items()
    }


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


def _warn_ignored_horizon(action_horizon: Optional[int]) -> None:
    """Accept — and ignore — a caller-supplied horizon.

    Older call sites (and the exploratory RTC branch) pass ``action_horizon=``. Raising a
    TypeError on them would only force a flag-day rename; the number is simply not used
    any more, because the chunk decides its own length. Say so once instead of silently
    accepting an argument that no longer means anything."""
    if action_horizon is not None:
        logger.warning(
            "ActionChunkBroker: action_horizon=%s is ignored — the chunk size now comes from "
            "the policy's response, so it cannot disagree with the checkpoint.",
            action_horizon,
        )


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy, action_horizon: Optional[int] = None) -> None:
        _warn_ignored_horizon(action_horizon)
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
