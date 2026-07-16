"""Policy serving (websocket + msgpack) loopback + action-chunk broker tests."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("websockets")

from yam_policy import (
    ActionChunkBroker,
    AsyncActionChunkBroker,
    WebsocketClientPolicy,
    WebsocketPolicyServer,
)
from yam_policy.policies.dummy import DummyPolicy
from yam_policy.yam_contract import DEFAULT_IMAGE_KEYS, validate_action_chunk, validate_server_metadata

from tests._util import free_port, wait_port

HORIZON = 8


def _serve(mode: str = "hold"):
    port = free_port()
    policy = DummyPolicy(action_dim=14, action_horizon=HORIZON, mode=mode)
    srv = WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=port,
        metadata={**policy.obs_spec, "action_horizon": HORIZON},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"
    return port


def test_loopback_metadata_and_chunking():
    port = _serve("hold")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    assert client.get_server_metadata()["action_horizon"] == HORIZON
    spec = validate_server_metadata(
        client.get_server_metadata(), configured_execution_horizon=HORIZON, configured_control_hz=30
    )
    assert spec.contract == "yam_bimanual_v1"

    broker = ActionChunkBroker(client, action_horizon=HORIZON)
    state = np.arange(14, dtype=np.float32)
    obs = {"observation/state": state}

    first = broker.infer(obs)["actions"]
    assert first.shape == (14,)
    assert np.allclose(first, state)  # "hold" repeats the state
    # pull more than one horizon -> re-queries the server without error
    for _ in range(2 * HORIZON + 3):
        assert broker.infer(obs)["actions"].shape == (14,)


def test_async_broker_prefetch():
    port = _serve("zeros")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    broker = AsyncActionChunkBroker(client, action_horizon=HORIZON)
    obs = {"observation/state": np.zeros(14, dtype=np.float32)}
    for _ in range(3 * HORIZON):
        a = broker.infer(obs)["actions"]
        assert a.shape == (14,)
    broker.close()


class _CountingPolicy:
    def __init__(self, model_horizon=50):
        self.calls = 0
        self.model_horizon = model_horizon

    def infer(self, obs):
        value = self.calls
        self.calls += 1
        return {"actions": np.full((self.model_horizon, 14), value, dtype=np.float32)}

    def reset(self):
        pass


def test_long_model_chunk_uses_shorter_execution_horizon():
    policy = _CountingPolicy(model_horizon=50)
    broker = ActionChunkBroker(policy, action_horizon=4)
    obs = {"state_id": 1}

    values = [int(broker.infer(obs)["actions"][0]) for _ in range(9)]

    assert values == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert policy.calls == 3


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros(14, dtype=np.float32),
        np.zeros((8, 13), dtype=np.float32),
        np.zeros((7, 14), dtype=np.float32),
        np.full((8, 14), np.nan, dtype=np.float32),
        np.full((8, 14), np.inf, dtype=np.float32),
    ],
)
def test_invalid_policy_chunks_fail_closed(actions):
    with pytest.raises(ValueError):
        validate_action_chunk({"actions": actions}, execution_horizon=8)


def test_production_server_metadata_validation():
    metadata = {
        "contract": "yam_bimanual_v1",
        "action_dim": 14,
        "action_horizon": 8,
        "model_action_horizon": 50,
        "control_hz": 30,
        "image_size": 224,
        "image_keys": DEFAULT_IMAGE_KEYS,
    }
    spec = validate_server_metadata(
        metadata,
        configured_execution_horizon=8,
        configured_control_hz=30,
    )
    assert spec.execution_horizon == 8
    assert spec.model_action_horizon == 50

    with pytest.raises(ValueError, match="disagrees"):
        validate_server_metadata(metadata, configured_execution_horizon=16, configured_control_hz=30)
    with pytest.raises(ValueError, match="missing required fields"):
        validate_server_metadata({"action_horizon": 8})


class _BlockingPrefetchPolicy(_CountingPolicy):
    def __init__(self):
        super().__init__(model_horizon=4)
        self.prefetch_started = threading.Event()
        self.release_prefetch = threading.Event()

    def infer(self, obs):
        call = self.calls
        self.calls += 1
        if call == 1:
            self.prefetch_started.set()
            assert self.release_prefetch.wait(timeout=2)
        return {"actions": np.full((4, 14), call, dtype=np.float32)}


def test_stale_async_prefetch_cannot_be_installed_after_reset():
    inner = _BlockingPrefetchPolicy()
    broker = AsyncActionChunkBroker(inner, action_horizon=4, prefetch_at=1)
    assert np.all(broker.infer({"state_id": "before"})["actions"] == 0)
    assert inner.prefetch_started.wait(timeout=1)

    broker.reset()
    inner.release_prefetch.set()
    time.sleep(0.02)
    post_reset = broker.infer({"state_id": "after"})["actions"]

    assert np.all(post_reset == 2), "prefetch from the pre-reset generation was reused"
    broker.close()


def test_websocket_inference_timeout_fails_closed():
    class SlowPolicy:
        def infer(self, obs):
            time.sleep(0.1)
            return {"actions": np.zeros((HORIZON, 14), dtype=np.float32)}

    port = free_port()
    server = WebsocketPolicyServer(SlowPolicy(), host="127.0.0.1", port=port, metadata={})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert wait_port(port)
    client = WebsocketClientPolicy(host="127.0.0.1", port=port, timeout=0.01)

    with pytest.raises(TimeoutError):
        client.infer({})
    client.close()
