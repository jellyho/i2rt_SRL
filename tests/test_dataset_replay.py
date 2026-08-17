"""Replay as a policy: a recorded episode driving the robot through the deploy stack.

Replay used to be a parallel implementation — its own robot client, its own command path
against `run_robot_server wrapper`, its own ramp. It differs from deployment in one thing,
where the actions come from, so it is now a `BasePolicy` that reads them out of a dataset and
everything else is the deploy stack unchanged.

These tests are about the two things that makes load-bearing: the actions must come out in the
recorded order, and what the client is told at the handshake must be enough to line the
past-demonstration overlay up with what the arm is doing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_serving"))

from yam_policy.policies.dataset_policy import DatasetPolicy

ACTION_DIM = 14
FPS = 30
LENGTHS = {0: 40, 1: 25}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """A v3.0-shaped dataset: two episodes packed into one parquet, deliberately out of order.

    Row order is not frame order in a real dataset either -- episodes share a file and the
    reader is expected to sort. Writing it shuffled is what makes that testable.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("replay_dataset")
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()

    episodes, frames, actions = [], [], []
    for episode, length in LENGTHS.items():
        for frame in range(length):
            episodes.append(episode)
            frames.append(frame)
            # value encodes (episode, frame) so a mis-order is visible, not plausible
            actions.append([float(episode * 1000 + frame)] * ACTION_DIM)

    order = np.random.default_rng(0).permutation(len(episodes))
    table = pa.table(
        {
            "episode_index": pa.array([episodes[i] for i in order], pa.int64()),
            "frame_index": pa.array([frames[i] for i in order], pa.int64()),
            "action": pa.array([actions[i] for i in order], pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({"fps": FPS, "total_episodes": len(LENGTHS)}))
    return root


def _drain(policy: DatasetPolicy) -> np.ndarray:
    """Every action the policy serves for one pass, as the broker would consume them."""
    out = []
    while policy.cursor < policy.frames:
        out.append(policy.infer({})["actions"])
    return np.concatenate(out)[: policy.frames]


# --------------------------------------------------------------------------------------- #
# The actions
# --------------------------------------------------------------------------------------- #
def test_the_episode_comes_back_in_recorded_order(dataset):
    """Sorted by frame_index, not by row order: many episodes share one parquet file."""
    replayed = _drain(DatasetPolicy(root=str(dataset), episode=1, chunk=7))
    np.testing.assert_allclose(replayed[:, 0], np.arange(LENGTHS[1]) + 1000)


def test_only_the_requested_episode_is_served(dataset):
    policy = DatasetPolicy(root=str(dataset), episode=0)
    assert policy.frames == LENGTHS[0]
    assert float(policy.infer({})["actions"][0][0]) == 0.0  # episode 0, frame 0


def test_every_reply_is_the_same_length(dataset):
    """The tail is padded rather than short. A short chunk is not wrong for the broker, but a
    ragged one makes an off-by-one in the caller look like data."""
    policy = DatasetPolicy(root=str(dataset), episode=1, chunk=8)
    while policy.cursor < policy.frames:
        assert policy.infer({})["actions"].shape == (8, ACTION_DIM)


def test_the_end_holds_the_final_pose_rather_than_stopping(dataset):
    """The client is driving a robot: it needs something to hold. An empty chunk would leave
    the followers on a stale command."""
    policy = DatasetPolicy(root=str(dataset), episode=0, chunk=5)
    last = _drain(policy)[-1]

    after = policy.infer({})
    np.testing.assert_allclose(after["actions"][0], last)
    assert float(after["replay_done"][0][0]) == 1.0


def test_a_rollout_starts_from_the_beginning(dataset):
    """`reset` is what the deploy client calls when a rollout starts, so pressing start twice
    replays the episode twice instead of continuing off the end."""
    policy = DatasetPolicy(root=str(dataset), episode=0, chunk=5)
    first = _drain(policy)
    policy.reset()
    assert policy.cursor == 0
    np.testing.assert_allclose(_drain(policy), first)


def test_looping_wraps_instead_of_holding(dataset):
    policy = DatasetPolicy(root=str(dataset), episode=1, chunk=4, loop=True)
    _drain(policy)
    np.testing.assert_allclose(policy.infer({})["actions"][0], [1000.0] * ACTION_DIM)


# --------------------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------------------- #
def test_speed_changes_the_stream_not_the_tick_rate(dataset):
    """The client ticks at a fixed rate, so the only way to replay faster is to send fewer
    frames. 2x drops every other one."""
    policy = DatasetPolicy(root=str(dataset), episode=0, speed=2.0)
    assert policy.frames == LENGTHS[0] // 2
    np.testing.assert_allclose(_drain(policy)[:3, 0], [0.0, 2.0, 4.0])


def test_slow_replay_repeats_frames_rather_than_interpolating(dataset):
    """An interpolated pose is one the robot never recorded; repeating holds it instead."""
    policy = DatasetPolicy(root=str(dataset), episode=0, speed=0.5)
    assert policy.frames == LENGTHS[0] * 2
    np.testing.assert_allclose(_drain(policy)[:4, 0], [0.0, 0.0, 1.0, 1.0])


def test_the_advertised_rate_follows_the_speed(dataset):
    """The overlay is played at this rate, so it has to be the replayed rate, not the file's."""
    assert DatasetPolicy(root=str(dataset), episode=0).policy_info["replay_fps"] == FPS
    assert DatasetPolicy(root=str(dataset), episode=0, speed=2.0).policy_info["replay_fps"] == FPS * 2


@pytest.mark.parametrize("speed", [0, -1])
def test_a_nonsense_speed_is_refused(dataset, speed):
    with pytest.raises(ValueError, match="speed"):
        DatasetPolicy(root=str(dataset), episode=0, speed=speed)


# --------------------------------------------------------------------------------------- #
# The handshake — what lets the overlay follow along
# --------------------------------------------------------------------------------------- #
def test_the_handshake_names_the_episode_being_replayed(dataset):
    from yam_policy.serve import build_metadata

    meta = build_metadata(
        DatasetPolicy(root=str(dataset), episode=1),
        "yam_policy.policies.dataset_policy:DatasetPolicy",
        {},
    )
    assert meta["framework"] == "dataset-replay"
    assert meta["replay_dataset"] == dataset.name
    assert meta["replay_episode"] == 1
    assert meta["replay_fps"] == FPS


def test_the_client_reads_that_back_as_a_replay(dataset):
    """The deploy client shows it and, in the GUI, points the overlay at it — so it has to
    survive the handshake as data, not just as a label."""
    from yam_policy.serve import build_metadata

    from workstation.policy_bridge.deploy_runner import _describe_policy

    meta = build_metadata(
        DatasetPolicy(root=str(dataset), episode=1),
        "yam_policy.policies.dataset_policy:DatasetPolicy",
        {},
    )
    assert _describe_policy(meta) == f"replay · {dataset.name}#1"


def test_replay_declares_its_done_flag_as_a_recorded_column(dataset):
    """An eval run of a replay is still a dataset; `replay_done` is what separates the episode
    from the held pose after it."""
    assert DatasetPolicy(root=str(dataset), episode=0).extra_features() == {"replay_done": [1]}


def test_no_cameras_are_requested(dataset):
    """Replay reads none of the observation. Saying so keeps the client from resizing frames
    for a policy that will not look at them."""
    assert DatasetPolicy(root=str(dataset), episode=0).obs_spec["image_keys"] == {}


# --------------------------------------------------------------------------------------- #
# Failing usefully
# --------------------------------------------------------------------------------------- #
def test_an_episode_that_is_not_there_says_what_is(dataset):
    with pytest.raises(ValueError, match="episodes 0..1"):
        DatasetPolicy(root=str(dataset), episode=99)


def test_a_directory_that_is_not_a_dataset_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/"):
        DatasetPolicy(root=str(tmp_path), episode=0)


# --------------------------------------------------------------------------------------- #
# Over the wire
# --------------------------------------------------------------------------------------- #
def test_a_replay_serves_over_the_deploy_wire(dataset):
    """Server + client + broker: the same three pieces deploy uses, with the dataset behind
    them. The broker hands out one step per tick, which is what makes replay real-time."""
    pytest.importorskip("websockets")
    import threading

    from yam_policy import ActionChunkBroker, WebsocketClientPolicy, WebsocketPolicyServer
    from yam_policy.serve import build_metadata

    from tests._util import free_port, wait_port

    policy = DatasetPolicy(root=str(dataset), episode=1, chunk=6)
    meta = build_metadata(policy, "yam_policy.policies.dataset_policy:DatasetPolicy", {})
    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata=meta)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"

    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    assert client.get_server_metadata()["replay_episode"] == 1

    broker = ActionChunkBroker(client)
    steps = [broker.infer({})["actions"] for _ in range(LENGTHS[1])]

    assert all(step.shape == (ACTION_DIM,) for step in steps)
    np.testing.assert_allclose([float(s[0]) for s in steps], np.arange(LENGTHS[1]) + 1000)
