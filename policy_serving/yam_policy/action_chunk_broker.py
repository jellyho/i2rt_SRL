"""ActionChunkBroker — serve a predicted action chunk one step at a time.

Mirrors ``openpi_client.action_chunk_broker``. Wrap any :class:`BasePolicy` (e.g.
a :class:`WebsocketClientPolicy`); the broker calls the inner policy only every
``action_horizon`` steps and returns one action per step from the cached chunk,
so you query the (possibly remote) policy ~once per N control ticks.

The inner policy must return ``{"actions": ndarray(action_horizon, action_dim), ...}``.
Each ``infer`` returns the same dict with array fields sliced to the current step.
"""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy


def _slice_step(results: Dict, step: int) -> Dict:
    return {k: (v[step, ...] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in results.items()}


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._cur_step = 0
        self._last_results: Dict | None = None

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = _slice_step(self._last_results, self._cur_step)

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._last_results = None
        self._cur_step = 0
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

    def infer(self, obs: Dict) -> Dict:
        self._last_obs = obs
        if self._cur is None:
            self._cur = self._policy.infer(obs)
            self._step = 0

        out = _slice_step(self._cur, self._step)
        self._step += 1

        if self._step == self._prefetch_at and self._next is None:
            self._next = self._exec.submit(self._policy.infer, self._last_obs)

        if self._step >= self._h:
            self._cur = self._next.result() if self._next is not None else self._policy.infer(self._last_obs)
            self._next = None
            self._step = 0

        return out

    def reset(self) -> None:
        if self._next is not None:
            try:
                # A websocket connection cannot service a new request while the
                # prefetch recv is still in progress. Finish and discard it before
                # declaring the action buffer empty.
                self._next.result()
            except Exception:
                pass
            self._next = None
        self._cur = None
        self._step = 0
        self._last_obs = None
        self._policy.reset()

    def close(self) -> None:
        self._exec.shutdown(wait=False)


class RTCActionChunkBroker(BasePolicy):
    """Workstation-side scheduler for Real-Time Chunking.

    The control loop calls :meth:`infer` every tick. This broker keeps executing
    the current action chunk, publishes a new request after
    ``min_execute_steps``, and replaces the queue as soon as the guided result is
    ready. The policy server receives the normalized unexecuted tail and a
    conservative delay estimate; it never owns the robot clock.
    """

    REQUEST_KEY = "_yam_rtc"

    def __init__(
        self,
        policy: BasePolicy,
        action_horizon: int,
        *,
        min_execute_steps: int = 8,
        rate_hz: float = 30.0,
        n_obs_steps: int = 1,
        delay_history_size: int = 10,
    ) -> None:
        self._policy = policy
        self._horizon = int(action_horizon)
        self._min_execute_steps = int(min_execute_steps)
        self._rate_hz = float(rate_hz)
        if self._rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._period = 1.0 / self._rate_hz
        self._history = deque(maxlen=max(1, int(n_obs_steps)))
        self._delay_estimates = deque(maxlen=max(1, int(delay_history_size)))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yam-rtc")

        if self._horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not 0 < self._min_execute_steps < self._horizon:
            raise ValueError("min_execute_steps must be greater than zero and shorter than action_horizon")
        if 2 * self._min_execute_steps > self._horizon:
            raise ValueError(
                "RTC timing is infeasible: min_execute_steps must not exceed half "
                f"the action horizon ({self._min_execute_steps} > {self._horizon}/2)"
            )

        self._current: Dict | None = None
        self._index = 0
        self._future = None
        self._request_started_at = 0.0
        self._request_start_action_count = 0
        self._total_actions = 0
        self._steps_since_request = 0
        self._discard_future = False

    @staticmethod
    def _copy_observation(obs: Dict) -> Dict:
        copied = {}
        for key, value in obs.items():
            if key == RTCActionChunkBroker.REQUEST_KEY:
                continue
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _validate_chunk(self, results: Dict) -> None:
        for key in ("actions", "model_actions"):
            value = results.get(key)
            if not isinstance(value, np.ndarray) or value.ndim < 2:
                raise ValueError(f"RTC policy response requires a [T, A] {key!r} array")
        if len(results["actions"]) != len(results["model_actions"]):
            raise ValueError("actions and model_actions must have the same horizon")
        if len(results["actions"]) != self._horizon:
            raise ValueError(
                "RTC policy response horizon does not match the configured action horizon: "
                f"{len(results['actions'])} != {self._horizon}"
            )

    @staticmethod
    def _drop_steps(results: Dict, steps: int) -> Dict:
        horizon = len(results["actions"])
        dropped = max(0, min(int(steps), horizon))
        return {
            key: (
                value[dropped:].copy()
                if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == horizon
                else value
            )
            for key, value in results.items()
        }

    def _predicted_delay(self) -> int:
        # Algorithm 1 initializes its delay buffer with d_init.  Before the first
        # asynchronous request we conservatively use s_min; the synchronous cold
        # start is a warmup and is deliberately excluded from the rolling buffer.
        return max(self._delay_estimates, default=self._min_execute_steps)

    def _record_wall_delay(self, elapsed: float, consumed_steps: int) -> int:
        # A ceiling is deliberately conservative for the next request's frozen
        # prefix. The number of actions actually consumed remains authoritative
        # if the caller is running faster than wall-clock rate in a test or replay.
        elapsed_steps = math.ceil(max(0.0, elapsed) / self._period)
        observed_delay = max(int(consumed_steps), 0, elapsed_steps)
        self._delay_estimates.append(observed_delay)
        return observed_delay

    def _make_request(
        self,
        obs: Dict,
        previous_tail: np.ndarray | None,
        processed_tail: np.ndarray | None = None,
        inference_delay: int = 0,
    ) -> Dict:
        request = self._copy_observation(obs)
        request[self.REQUEST_KEY] = {
            "inference_delay": int(inference_delay),
            "prev_chunk_left_over": previous_tail,
            "prev_processed_actions": processed_tail,
            "execution_horizon": 0 if previous_tail is None else len(previous_tail),
            "observation_history": [self._copy_observation(frame) for frame in self._history],
        }
        return request

    def _initial_inference(self, obs: Dict) -> None:
        results = self._policy.infer(self._make_request(obs, None, inference_delay=0))
        self._validate_chunk(results)
        self._current = results
        self._index = 0
        self._steps_since_request = 0

    def _launch_inference(self, obs: Dict) -> None:
        assert self._current is not None
        previous_tail = self._current["model_actions"][self._index :].copy()
        processed_tail = self._current["actions"][self._index :].copy()
        inference_delay = self._predicted_delay()
        execution_steps = self._steps_since_request
        expected_tail = self._horizon - execution_steps
        if len(previous_tail) != expected_tail:
            raise RuntimeError(
                "RTC broker lost its action-time alignment: expected an overlap of "
                f"{expected_tail} steps, got {len(previous_tail)}"
            )
        if not inference_delay <= execution_steps <= self._horizon - inference_delay:
            raise RuntimeError(
                "RTC timing is infeasible: expected d <= s <= H - d, got "
                f"d={inference_delay}, s={execution_steps}, H={self._horizon}"
            )
        request = self._make_request(
            obs,
            previous_tail,
            processed_tail,
            inference_delay=inference_delay,
        )
        self._request_started_at = time.monotonic()
        self._request_start_action_count = self._total_actions
        self._future = self._executor.submit(self._policy.infer, request)
        self._discard_future = False

    def _finish_inference(self, *, block: bool) -> bool:
        if self._future is None or (not block and not self._future.done()):
            return False

        results = self._future.result()
        elapsed = time.monotonic() - self._request_started_at
        consumed_during_inference = max(0, self._total_actions - self._request_start_action_count)
        self._future = None

        if self._discard_future:
            self._discard_future = False
            return False

        self._validate_chunk(results)
        observed_delay = self._record_wall_delay(elapsed, consumed_during_inference)
        if 2 * observed_delay > self._horizon:
            raise RuntimeError(
                "RTC inference overran the feasible horizon: no s satisfies "
                f"d <= s <= H - d for d={observed_delay}, H={self._horizon}"
            )
        if consumed_during_inference >= self._horizon:
            raise RuntimeError("RTC inference completed after its entire predicted action chunk had expired")
        self._current = self._drop_steps(results, consumed_during_inference)
        self._index = 0
        # Algorithm 1 carries forward actions consumed while inference was in
        # flight. This makes request starts remain min_execute_steps apart.
        self._steps_since_request = consumed_during_inference
        return True

    def infer(self, obs: Dict) -> Dict:
        stable_obs = self._copy_observation(obs)
        self._history.append(stable_obs)

        self._finish_inference(block=False)
        if self._current is None:
            if self._future is not None:
                self._finish_inference(block=True)
            if self._current is None:
                self._initial_inference(stable_obs)

        assert self._current is not None
        if self._index >= len(self._current["actions"]):
            if self._future is not None:
                if not self._future.done():
                    raise RuntimeError(
                        "RTC action chunk exhausted before asynchronous inference completed; "
                        "stopping instead of blocking the control loop or repeating an unsafe action"
                    )
                self._finish_inference(block=False)
            if self._current is None or self._index >= len(self._current["actions"]):
                raise RuntimeError("RTC action chunk exhausted without a usable replacement")

        # With this controller API, obs_t and action_t arrive in the same call.
        # Launch before selecting action_t so the request uses obs_t together
        # with a previous tail whose first entry is action_t.
        if self._future is None:
            predicted_delay = self._predicted_delay()
            if 2 * predicted_delay > self._horizon:
                raise RuntimeError(
                    "RTC predicted delay is infeasible for this action horizon: "
                    f"d={predicted_delay}, H={self._horizon}"
                )
            launch_after = max(self._min_execute_steps, predicted_delay)
            if self._steps_since_request >= launch_after:
                self._launch_inference(stable_obs)

        assert self._current is not None and self._index < len(self._current["actions"])
        output = _slice_step(self._current, self._index)
        self._index += 1
        self._total_actions += 1
        self._steps_since_request += 1

        return output

    def reset(self) -> None:
        if self._future is not None:
            # A websocket connection cannot be reset or reused concurrently with
            # its in-flight request. Finish and discard that response first.
            self._discard_future = True
            self._finish_inference(block=True)
        self._current = None
        self._index = 0
        self._history.clear()
        self._delay_estimates.clear()
        self._total_actions = 0
        self._steps_since_request = 0
        self._policy.reset()

    def close(self) -> None:
        # The caller closes the underlying websocket immediately afterwards, so
        # do not leave a worker using it in the background.
        self._executor.shutdown(wait=True, cancel_futures=True)
