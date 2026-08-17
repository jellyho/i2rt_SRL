"""AsyncChunkBroker — inference overlapped with execution, and spliced at the right index.

The load-bearing test here is `test_the_executed_stream_is_exact_whatever_the_latency`. The fake
policy answers the observation of step ``t`` with the chunk ``[t, t+1, t+2, ...]``, so the action
predicted for global step ``M`` has the value ``M`` no matter which chunk it came from. Therefore
the stream the broker hands out must be exactly ``0, 1, 2, ...``: any error in the delay
alignment shows up immediately as a repeat (executing an action already executed -- the judder a
naive async prefetch produces) or a skip.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from yam_policy.async_chunk_broker import AsyncChunkBroker, LatencyTracker


class _RampPolicy:
    """Answers the observation of step ``t`` with the chunk ``[t, t+1, ...]``, after ``delay`` s."""

    def __init__(self, horizon=10, delay=0.0, dim=2, samples=None):
        self.horizon = horizon
        self.delay = delay
        self.dim = dim
        self.samples = samples
        self.calls = 0
        self.resets = 0
        self.concurrent = 0
        self._max_concurrent = 0
        self._lock = threading.Lock()

    def infer(self, obs):
        with self._lock:
            self.concurrent += 1
            self._max_concurrent = max(self._max_concurrent, self.concurrent)
            self.calls += 1
        try:
            if self.delay:
                time.sleep(self.delay)
            t = int(obs["t"])
            actions = np.stack([np.full(self.dim, t + i, dtype=float) for i in range(self.horizon)])
            out = {"actions": actions}
            if self.samples:
                # An extras array shaped like action_samples: (chunk, N, dim), sliced per step.
                out["action_samples"] = np.stack(
                    [np.full((self.samples, self.dim), t + i, dtype=float) for i in range(self.horizon)]
                )
            return out
        finally:
            with self._lock:
                self.concurrent -= 1

    def reset(self):
        self.resets += 1

    @property
    def max_concurrent(self):
        return self._max_concurrent


def _drive(broker, ticks, period=0.0):
    """Run the control loop: one infer per tick, obs carrying the global step."""
    out = []
    for t in range(ticks):
        res = broker.infer({"t": t})
        out.append(float(np.asarray(res["actions"]).ravel()[0]))
        if period:
            time.sleep(period)
    return out


@pytest.mark.parametrize(
    "delay,horizon,period",
    [
        (0.0, 10, 0.0),  # instant policy
        (0.02, 10, 0.005),  # ordinary: latency well inside the chunk
        (0.12, 10, 0.005),  # slow: inference outlives the chunk it predicts -> underruns
    ],
)
def test_the_executed_stream_is_exact_whatever_the_latency(delay, horizon, period):
    """No repeats, no skips: action for step M is the one the policy predicted for step M.

    This is the delay alignment. A broker that consumed each reply from index 0 would repeat the
    steps executed while it was inferring; one that over-advanced would skip.
    """
    policy = _RampPolicy(horizon=horizon, delay=delay)
    broker = AsyncChunkBroker(policy, rate_hz=1.0 / max(period, 0.005))
    got = _drive(broker, 40, period)
    assert got == [float(i) for i in range(40)]
    assert policy.max_concurrent == 1, "the wrapped policy must never be called concurrently"


def test_execution_does_not_stall_once_a_chunk_is_in_hand():
    """The point of the class: only the first inference blocks the control loop."""
    policy = _RampPolicy(horizon=20, delay=0.1)
    broker = AsyncChunkBroker(policy, rate_hz=100.0)
    slow = 0
    for t in range(60):
        t0 = time.monotonic()
        broker.infer({"t": t})
        if time.monotonic() - t0 > 0.05:
            slow += 1
        time.sleep(0.01)  # 100 Hz control loop; chunk lasts 0.2 s, inference takes 0.1 s
    assert slow == 1, f"expected only the first call to block, {slow} calls stalled"
    assert broker.underruns == 0


def test_a_policy_slower_than_its_chunk_underruns_and_says_so():
    """Prefetch cannot hide a server slower than the chunk duration -- but it must be visible."""
    policy = _RampPolicy(horizon=4, delay=0.08)
    broker = AsyncChunkBroker(policy, rate_hz=100.0)
    _drive(broker, 20, period=0.01)  # chunk lasts 0.04 s, inference takes 0.08 s
    assert broker.underruns > 0
    assert broker.stats()["underruns"] == broker.underruns


def test_extras_are_sliced_per_step_like_the_sync_broker():
    policy = _RampPolicy(horizon=6, samples=3, dim=2)
    broker = AsyncChunkBroker(policy)
    for t in range(12):
        res = broker.infer({"t": t})
        samples = np.asarray(res["action_samples"])
        assert samples.shape == (3, 2), "one candidate snapshot per step, not the whole chunk"
        assert np.allclose(samples, t), "the candidates must be this step's, not the chunk's first"


def test_leftover_is_the_unexecuted_tail_the_prefix_guidance_would_use():
    policy = _RampPolicy(horizon=8)
    broker = AsyncChunkBroker(policy, prefetch_ticks=1)
    broker.infer({"t": 0})
    left = broker.leftover()
    assert left is not None
    assert len(left) == 7, "8-step chunk with one step consumed"
    assert float(left[0][0]) == 1.0, "the tail starts at the next action to execute"
    broker.infer({"t": 1})
    assert len(broker.leftover()) == 6


def test_the_prefix_hook_rides_along_with_the_request():
    """RTC's client side: the unexecuted prefix is offered to the server with the next request."""
    seen = {}

    class _Spy(_RampPolicy):
        def infer(self, obs):
            if "prefix" in obs:
                seen["prefix"] = np.asarray(obs["prefix"]).copy()
            return super().infer(obs)

    policy = _Spy(horizon=6)
    broker = AsyncChunkBroker(policy, prefetch_ticks=4)
    broker.set_prefix_fn(lambda leftover: {"prefix": leftover})
    _drive(broker, 10)
    assert "prefix" in seen, "a request issued while a chunk was in hand carried no prefix"
    assert seen["prefix"].ndim == 2


def test_reset_drops_the_chunk_and_the_reply_in_flight():
    policy = _RampPolicy(horizon=10, delay=0.02)
    broker = AsyncChunkBroker(policy)
    _drive(broker, 5)
    broker.reset()
    assert policy.resets == 1
    assert broker.leftover() is None
    assert broker.action_horizon == 0
    # A reset rollout starts its numbering over, and is still exact.
    assert _drive(broker, 12) == [float(i) for i in range(12)]


def test_policy_errors_surface_on_the_control_thread():
    class _Broken(_RampPolicy):
        def infer(self, obs):
            raise ConnectionError("server went away")

    broker = AsyncChunkBroker(_Broken())
    with pytest.raises(ConnectionError):
        broker.infer({"t": 0})


def test_latency_estimate_rises_at_once_and_decays_slowly():
    """Sized for the worst recent inference: being late costs a stall, being early costs little."""
    tracker = LatencyTracker(rate_hz=30.0)
    tracker.observe(0.010)
    tracker.observe(0.200)  # a spike must be adopted immediately
    assert tracker.estimate_s == pytest.approx(0.200)
    for _ in range(5):
        tracker.observe(0.010)
    assert tracker.estimate_s > 0.010, "must not forget the spike after one fast reply"
    assert tracker.estimate_ticks == int(np.ceil(tracker.estimate_s * 30.0))


def test_stats_carry_the_provenance_the_dataset_records():
    policy = _RampPolicy(horizon=5)
    broker = AsyncChunkBroker(policy)
    for t in range(7):
        broker.infer({"t": t})
    st = broker.stats()
    assert st["chunk_len"] == 5
    assert st["chunk_index"] >= 1, "a second chunk must have been spliced in by step 7"
    assert 0 <= st["step_in_chunk"] < 5
    assert st["delay_ticks"] >= 0
