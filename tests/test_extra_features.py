"""Extra per-step data: declared at handshake, sliced per step, stored as dataset columns.

A policy may return more than actions — a critic's value, the candidates it chose between,
whatever it wants kept. The client hard-codes none of it: the server declares the names and
per-step shapes once, at handshake, and everything downstream follows from that.

The one rule is that an extra array's leading axis is the chunk, the same as `actions`. That
is what makes it per-step data, and therefore something a dataset can hold one row of per
frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_serving"))

from yam_policy.action_chunk_broker import ActionChunkBroker

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
from workstation.policy_bridge.deploy_runner import _describe_policy, _extra_features_from_meta

HORIZON = 30
CANDIDATES = 4


# --------------------------------------------------------------------------------------- #
# The declaration
# --------------------------------------------------------------------------------------- #
def test_the_server_names_the_features_not_the_client():
    declared = _extra_features_from_meta({"extra_features": {"critic_q": [1], "action_samples": [CANDIDATES, 14]}})
    assert declared == {"critic_q": (1,), "action_samples": (CANDIDATES, 14)}


def test_a_bare_int_shape_is_accepted():
    assert _extra_features_from_meta({"extra_features": {"q": 3}}) == {"q": (3,)}


def test_a_policy_that_declares_nothing_gets_nothing():
    assert _extra_features_from_meta({}) == {}
    assert _extra_features_from_meta({"extra_features": None}) == {}


@pytest.mark.parametrize(
    "declared",
    [
        {"extra_features": [1, 2]},  # not a mapping
        {"extra_features": {"a": "wide"}},  # shape is not numbers
        {"extra_features": {"a": [0]}},  # a dimension of zero
        {"extra_features": {"a": []}},  # no dimensions at all
    ],
)
def test_a_malformed_declaration_is_dropped_rather_than_trusted(declared):
    """A wrong shape here becomes a wrong column in every episode of the dataset."""
    assert _extra_features_from_meta(declared) == {}


def test_one_bad_entry_does_not_take_the_good_ones_with_it():
    assert _extra_features_from_meta({"extra_features": {"bad": [0], "good": [2]}}) == {"good": (2,)}


# --------------------------------------------------------------------------------------- #
# Naming what answered on the port
# --------------------------------------------------------------------------------------- #
def test_each_stack_is_named_from_its_own_handshake():
    """openpi, its ACRFT fork and a LeRobot checkpoint all speak this wire, and all three return
    a well-formed chunk from an observation the other two would read differently."""
    assert _describe_policy({"framework": "acrft", "policy_name": "pi05_yam_lego_taxi"}) == (
        "ACRFT · pi05_yam_lego_taxi"
    )
    assert _describe_policy({"framework": "lerobot", "policy_type": "act"}) == "LeRobot · act"
    assert _describe_policy({"framework": "openpi", "policy_name": "pi0_fast_droid"}) == ("openpi · pi0_fast_droid")


def test_a_framework_with_no_policy_name_still_names_the_framework():
    assert _describe_policy({"framework": "lerobot"}) == "LeRobot"


def test_the_launcher_spec_is_trimmed_to_the_class():
    """serve.py puts `module:Class` on the wire; the module path is noise in a status bar."""
    assert (
        _describe_policy({"framework": "yam-policy", "policy": "yam_policy.policies.dummy:DummyPolicy"})
        == "yam-policy · DummyPolicy"
    )


def test_a_server_that_does_not_name_itself_is_a_guess_not_an_assertion():
    """Upstream openpi predates this field. Showing a confident wrong name would be worse than
    showing none, so an inferred one is marked as inferred."""
    assert _describe_policy({"action_horizon": 30}).endswith("?")
    assert _describe_policy({"action_horizon": 30, "supports_multi_sample": True}) == "ACRFT?"


def test_an_empty_handshake_says_so():
    assert _describe_policy({}) == "unidentified server"
    assert _describe_policy(None) == "unidentified server"


# --------------------------------------------------------------------------------------- #
# The slicing
# --------------------------------------------------------------------------------------- #
class _Policy:
    """Returns actions plus two extras, each with the chunk on its leading axis."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {
            "actions": np.zeros((HORIZON, 14), np.float32),
            # value of step i is i, so a mis-slice is visible rather than plausible
            "critic_q": np.arange(HORIZON, dtype=np.float32).reshape(HORIZON, 1),
            "action_samples": np.tile(np.arange(CANDIDATES, dtype=np.float32)[None, :, None], (HORIZON, 1, 14)),
            "run_id": np.array([7.0], np.float32),  # NOT per-step
            "note": "hello",  # not an array at all
        }

    def reset(self):
        pass


def test_every_array_whose_leading_axis_is_the_chunk_is_sliced():
    policy = _Policy()
    broker = ActionChunkBroker(policy)

    for step in range(HORIZON + 2):
        result = broker.infer({})
        assert result["actions"].shape == (14,)
        assert result["action_samples"].shape == (CANDIDATES, 14)
        assert float(result["critic_q"][0]) == step % HORIZON, step

    assert policy.calls == 2, "still one inference per chunk"


def test_an_array_that_is_not_per_step_is_passed_through_whole():
    """Slicing it would return a different thing entirely and still look like data."""
    result = ActionChunkBroker(_Policy()).infer({})
    assert result["run_id"].shape == (1,)
    assert result["note"] == "hello"


# --------------------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------------------- #
def _writer(tmp_path, declared):
    cams = {"cam": (32, 32, 3)}
    cfg = RecorderConfig(repo_id="t/extras", root=str(tmp_path), mock=False, fps=30, task="t")
    return AsyncDatasetWriter(cfg, list(cams), cams, extra_features=declared)


def _frame(i, extras):
    frame = {
        "images": {"cam": np.full((32, 32, 3), i % 255, np.uint8)},
        "observation.state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
    }
    frame.update(extras)
    return frame


def test_declared_extras_become_columns_at_their_declared_shape(tmp_path):
    """Self-describing on purpose. Flattened, [N, action_dim] candidates would be stored as an
    anonymous 56-vector and the reader would have to know the layout out of band."""
    declared = {"critic_q": (1,), "action_samples": (CANDIDATES, 14)}
    writer = _writer(tmp_path, declared)
    blank = {k: np.zeros(int(np.prod(v)), np.float32) for k, v in declared.items()}
    writer.open(_frame(0, blank))

    assert writer._features["critic_q"]["shape"] == (1,)
    assert writer._features["action_samples"]["shape"] == (CANDIDATES, 14)

    for i in range(12):
        extras = {
            "critic_q": np.array([float(i)], np.float32),
            "action_samples": np.tile(np.arange(CANDIDATES, dtype=np.float32)[:, None], (1, 14)).reshape(-1),
        }
        writer.stream_frame(_frame(i, extras), "t")
    writer.end_episode("success", "t")
    writer.finalize()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/verify", root=str(tmp_path / "extras"))
    for i in (0, 5, 11):
        item = dataset[i]
        assert float(np.ravel(item["critic_q"])[0]) == i, "the per-frame value did not vary"
        assert tuple(np.asarray(item["action_samples"]).shape) == (CANDIDATES, 14)

    info = json.loads((tmp_path / "extras" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 12


def test_a_run_that_declares_nothing_records_the_same_columns_as_before(tmp_path):
    """Deploy without a declaring policy must produce exactly the dataset it produced before."""
    import pandas as pd

    writer = _writer(tmp_path, {})
    writer.open(_frame(0, {}))
    for i in range(6):
        writer.stream_frame(_frame(i, {}), "t")
    writer.end_episode("success", "t")
    writer.finalize()

    columns = set(pd.read_parquet(sorted((tmp_path / "extras").rglob("data/**/*.parquet"))[0]).columns)
    assert "critic_q" not in columns and "action_samples" not in columns


# --------------------------------------------------------------------------------------- #
# An adaptive chunk
# --------------------------------------------------------------------------------------- #
class _AdaptivePolicy:
    """A policy whose chunk length changes from one replan to the next."""

    LENGTHS = (4, 9, 2, 7)

    def __init__(self):
        self.calls = -1

    def infer(self, obs):
        self.calls += 1
        length = self.LENGTHS[self.calls % len(self.LENGTHS)]
        return {
            "actions": np.zeros((length, 14), np.float32),
            # A critic's value for every candidate, at every step: (X, N)
            "critic_q": np.tile(np.arange(CANDIDATES, dtype=np.float32), (length, 1)),
            # The candidates themselves: (X, N, action_dim)
            "action_samples": np.zeros((length, CANDIDATES, 14), np.float32),
        }

    def reset(self):
        pass


def test_extras_follow_a_chunk_whose_length_changes():
    """The chunk is adaptive, so nothing may assume a fixed horizon.

    The length is read off `actions` on every reply, which is what lets an (X, N, action_dim)
    candidate set follow the same X its (X, action_dim) actions did — with no renegotiation,
    and with the declaration untouched because it only ever named the per-step shape.
    """
    policy = _AdaptivePolicy()
    broker = ActionChunkBroker(policy)

    seen = set()
    for _ in range(30):
        result = broker.infer({})
        seen.add(broker.action_horizon)
        assert result["actions"].shape == (14,)
        assert result["critic_q"].shape == (CANDIDATES,), "a Q per candidate, this step"
        assert result["action_samples"].shape == (CANDIDATES, 14)

    assert len(seen) > 1, f"the chunk length never changed ({seen}) — nothing was proven"


def test_the_declaration_does_not_mention_the_chunk_length():
    """Because neither side can know it: it is whatever a reply carries."""
    declared = _extra_features_from_meta(
        {"extra_features": {"critic_q": [CANDIDATES], "action_samples": [CANDIDATES, 14]}}
    )
    assert declared["critic_q"] == (CANDIDATES,)
    assert declared["action_samples"] == (CANDIDATES, 14)
