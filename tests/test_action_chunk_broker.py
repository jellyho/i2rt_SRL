"""ActionChunkBroker — the chunk size comes from the policy, never from configuration."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_OURS = pathlib.Path(__file__).resolve().parent.parent / "policy_serving" / "yam_policy"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_ours_{name}", _OURS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base_policy = _load("base_policy")
import sys  # noqa: E402

sys.modules.setdefault("_ours_pkg_base", base_policy)
acb_path = _OURS / "action_chunk_broker.py"
_src = acb_path.read_text().replace("from .base_policy import BasePolicy", "BasePolicy = object")
_ns: dict = {}
exec(compile(_src, str(acb_path), "exec"), _ns)  # noqa: S102
ActionChunkBroker = _ns["ActionChunkBroker"]
chunk_len = _ns["chunk_len"]


class FakePolicy:
    """Returns a chunk of `horizon` actions, counting how often it is queried."""

    def __init__(self, horizon: int, action_dim: int = 14) -> None:
        self.horizon = horizon
        self.action_dim = action_dim
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        base = 100 * self.calls
        return {"actions": np.arange(base, base + self.horizon * self.action_dim, dtype=np.float32).reshape(
            self.horizon, self.action_dim
        )}

    def reset(self):
        pass


@pytest.mark.parametrize("horizon", [1, 2, 5, 16, 30])
def test_infers_once_per_chunk_whatever_the_size(horizon):
    """No configured horizon: 3 chunks' worth of steps must cost exactly 3 inferences."""
    p = FakePolicy(horizon)
    broker = ActionChunkBroker(p)
    for _ in range(horizon * 3):
        broker.infer({})
    assert p.calls == 3
    assert broker.action_horizon == horizon


def test_every_action_in_the_chunk_is_used_in_order():
    """The bug this design removes: a client assuming 16 against a 30-chunk policy
    silently discarded the back half of every chunk."""
    p = FakePolicy(30)
    broker = ActionChunkBroker(p)
    got = [broker.infer({})["actions"] for _ in range(30)]
    assert p.calls == 1  # one chunk covered all 30 steps
    expected = p.infer({})["actions"]  # call 2, same layout shifted by 100
    np.testing.assert_array_equal(np.stack(got), expected - 100)


def test_chunk_size_may_change_between_inferences():
    """Nothing caches the horizon, so a policy is free to answer with a different size."""

    class Varying(FakePolicy):
        sizes = [4, 2, 7]

        def infer(self, obs):
            self.horizon = self.sizes[self.calls % len(self.sizes)]
            return super().infer(obs)

    p = Varying(4)
    broker = ActionChunkBroker(p)
    seen = []
    for _ in range(4 + 2 + 7):
        broker.infer({})
        seen.append(broker.action_horizon)
    assert p.calls == 3
    assert seen[0] == 4 and seen[4] == 2 and seen[6] == 7


def test_unchunked_policy_still_works():
    """A policy returning a single action means re-infer every tick, not a crash."""

    class Single:
        calls = 0

        def infer(self, obs):
            Single.calls += 1
            return {"actions": np.zeros(14, dtype=np.float32)}

        def reset(self):
            pass

    p = Single()
    broker = ActionChunkBroker(p)
    for _ in range(3):
        assert broker.infer({})["actions"].shape == (14,)
    assert p.calls == 3


def test_chunk_len_reads_the_leading_dim():
    assert chunk_len({"actions": np.zeros((30, 14))}) == 30
    assert chunk_len({"actions": np.zeros(14)}) == 1
    assert chunk_len({}) == 1


def test_reset_drops_the_cached_chunk():
    p = FakePolicy(30)
    broker = ActionChunkBroker(p)
    broker.infer({})
    broker.reset()
    broker.infer({})
    assert p.calls == 2


# ------------------------------------------------- back-compat with older call sites
def test_action_horizon_argument_is_accepted_and_ignored(caplog):
    """Older call sites pass action_horizon=. They must keep working — and must get the
    adaptive behaviour anyway, not the number they asked for."""
    p = FakePolicy(30)
    with caplog.at_level("WARNING"):
        broker = ActionChunkBroker(p, action_horizon=16)  # the stale default
    for _ in range(30):
        broker.infer({})
    assert p.calls == 1  # followed the policy's 30, not the requested 16
    assert broker.action_horizon == 30
    assert "ignored" in caplog.text


def test_each_chunk_is_inferred_from_the_observation_of_that_very_tick():
    """The reason the prefetching broker is gone.

    It started the next inference two steps early, on the observation available then, so every
    chunk was computed from a slightly older world than the one it was applied to. Measured,
    the two came out a wash — it started early by the same margin it was stale by — but "the
    chunk came from the observation I handed over" is worth being able to state without an
    argument about timing. The cost is a stall at each chunk boundary, which is what openpi's
    and LeRobot's own brokers do too.
    """
    seen = []

    class _Policy:
        def infer(self, obs):
            seen.append(obs["tick"])
            return {"actions": np.zeros((4, 3))}

        def reset(self):
            pass

    broker = ActionChunkBroker(_Policy())
    for tick in range(9):
        broker.infer({"tick": tick})

    # A 4-step chunk re-infers on ticks 0, 4 and 8 — each with THAT tick's observation.
    assert seen == [0, 4, 8]
