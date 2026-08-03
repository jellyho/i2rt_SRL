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
    RTCActionChunkBroker,
    WebsocketClientPolicy,
    WebsocketPolicyServer,
)
from yam_policy.policies.dummy import DummyPolicy

from tests._util import free_port, wait_port

HORIZON = 8


def _serve(mode: str = "hold"):
    port = free_port()
    srv = WebsocketPolicyServer(
        DummyPolicy(action_dim=14, action_horizon=HORIZON, mode=mode),
        host="127.0.0.1",
        port=port,
        metadata={"action_horizon": HORIZON},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"
    return port


def test_loopback_metadata_and_chunking():
    port = _serve("hold")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    assert client.get_server_metadata()["action_horizon"] == HORIZON

    broker = ActionChunkBroker(client, action_horizon=HORIZON)
    state = np.arange(14, dtype=np.float32)
    obs = {"observation/state": state}

    first = broker.infer(obs)["actions"]
    assert first.shape == (14,)
    assert np.allclose(first, state)  # "hold" repeats the state
    # pull more than one horizon -> re-queries the server without error
    for _ in range(2 * HORIZON + 3):
        assert broker.infer(obs)["actions"].shape == (14,)


def test_websocket_reset_is_applied_on_policy_server():
    class ResettablePolicy:
        def __init__(self):
            self.reset_calls = 0

        def infer(self, obs):
            return {"actions": np.zeros((HORIZON, 1), dtype=np.float32)}

        def reset(self):
            self.reset_calls += 1

    policy = ResettablePolicy()
    port = free_port()
    server = WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=port,
        metadata={"action_horizon": HORIZON},
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"

    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    assert client.get_server_metadata()["supports_reset"] is True
    client.reset()
    assert policy.reset_calls == 1
    client.close()


def test_async_broker_prefetch():
    port = _serve("zeros")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    broker = AsyncActionChunkBroker(client, action_horizon=HORIZON)
    obs = {"observation/state": np.zeros(14, dtype=np.float32)}
    for _ in range(3 * HORIZON):
        a = broker.infer(obs)["actions"]
        assert a.shape == (14,)
    broker.close()


def test_async_broker_reset_waits_for_and_discards_prefetch():
    policy = _ControlledRTCPolicy()
    broker = AsyncActionChunkBroker(policy, action_horizon=HORIZON, prefetch_at=1)
    obs = {"observation/state": np.zeros(1, dtype=np.float32)}

    assert float(broker.infer(obs)["actions"][0]) == 0.0
    deadline = time.monotonic() + 2.0
    while len(policy.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(policy.calls) == 2

    reset_thread = threading.Thread(target=broker.reset)
    reset_thread.start()
    reset_thread.join(timeout=0.05)
    assert reset_thread.is_alive()

    policy.release.set()
    reset_thread.join(timeout=2.0)
    assert not reset_thread.is_alive()
    assert broker._cur is None
    assert broker._next is None
    assert broker._last_obs is None

    # The prefetched call (100) was discarded; restart performs a fresh call (200).
    assert float(broker.infer(obs)["actions"][0]) == 200.0
    broker.close()


class _ControlledRTCPolicy:
    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.returning = threading.Event()

    def infer(self, obs):
        call = len(self.calls)
        self.calls.append(obs)
        if call > 0:
            assert self.release.wait(timeout=2.0)
            self.returning.set()
        values = np.arange(HORIZON, dtype=np.float32)[:, None] + 100 * call
        return {"actions": values.copy(), "model_actions": values.copy()}

    def reset(self):
        pass


def test_rtc_broker_schedules_requests_and_realigns_completed_chunk():
    policy = _ControlledRTCPolicy()
    broker = RTCActionChunkBroker(
        policy,
        action_horizon=HORIZON,
        min_execute_steps=4,
        rate_hz=30,
        n_obs_steps=2,
    )

    emitted = []
    for tick in range(4):
        emitted.append(float(broker.infer({"observation/state": np.array([tick])})["actions"][0]))

    assert emitted == [0.0, 1.0, 2.0, 3.0]
    assert len(policy.calls) == 1

    # This controller supplies obs_t while requesting action_t. Launch at the
    # beginning of tick 4 so obs_4 is paired with a tail beginning at action_4.
    emitted.append(float(broker.infer({"observation/state": np.array([4])})["actions"][0]))
    deadline = time.monotonic() + 2.0
    while len(policy.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(policy.calls) == 2
    rtc_request = policy.calls[1][RTCActionChunkBroker.REQUEST_KEY]
    assert rtc_request["prev_chunk_left_over"].squeeze(-1).tolist() == [4.0, 5.0, 6.0, 7.0]
    assert int(policy.calls[1]["observation/state"][0]) == 4
    assert [int(frame["observation/state"][0]) for frame in rtc_request["observation_history"]] == [3, 4]
    assert rtc_request["inference_delay"] == 4
    assert rtc_request["execution_horizon"] == 4

    # Two more old actions execute while the server is busy (three total,
    # including action_4 emitted immediately after launching the request).
    for tick in range(5, 7):
        emitted.append(float(broker.infer({"observation/state": np.array([tick])})["actions"][0]))
    assert emitted[-3:] == [4.0, 5.0, 6.0]

    policy.release.set()
    assert policy.returning.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while not broker._future.done() and time.monotonic() < deadline:
        time.sleep(0.001)

    # The response was aligned with tick 4; drop actions 4--6 consumed during
    # inference and immediately execute its action for tick 7.
    action = broker.infer({"observation/state": np.array([7])})["actions"]
    assert float(action[0]) == 103.0
    broker.close()


def test_rtc_broker_rejects_infeasible_execution_horizon():
    with pytest.raises(ValueError, match="half the action horizon"):
        RTCActionChunkBroker(_ControlledRTCPolicy(), action_horizon=HORIZON, min_execute_steps=5)


def test_rtc_broker_fails_loudly_when_async_inference_overruns_chunk():
    policy = _ControlledRTCPolicy()
    broker = RTCActionChunkBroker(
        policy,
        action_horizon=HORIZON,
        min_execute_steps=4,
        rate_hz=30,
    )

    try:
        for tick in range(HORIZON):
            broker.infer({"observation/state": np.array([tick])})
        with pytest.raises(RuntimeError, match="exhausted before asynchronous inference completed"):
            broker.infer({"observation/state": np.array([HORIZON])})
    finally:
        policy.release.set()
        broker.close()


def test_rtc_broker_requires_server_model_space_actions():
    class MissingModelActions:
        def infer(self, obs):
            return {"actions": np.zeros((HORIZON, 2), dtype=np.float32)}

        def reset(self):
            pass

    broker = RTCActionChunkBroker(MissingModelActions(), action_horizon=HORIZON, min_execute_steps=2)
    with pytest.raises(ValueError, match="model_actions"):
        broker.infer({"observation/state": np.zeros(2)})
    broker.close()
