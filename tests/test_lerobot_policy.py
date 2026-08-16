"""Deploying a LeRobot checkpoint through this stack, with nothing converted by hand.

The scenario under test is the whole point of the adapter: train with LeRobot, point the server at
the checkpoint directory, and have the deploy client configure itself from the handshake. So the
fixture builds a real (tiny) ACT checkpoint with LeRobot's own factories rather than mocking one,
and the assertions are about the things that break silently -- normalisation applied on the wrong
axis, a camera that quietly becomes a black frame, the two state conventions colliding.

Skipped unless `lerobot` and `torch` are importable, since the policy-server env is separate from
the robot's (see policy_serving/README.md).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_serving"))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from yam_policy.policies.lerobot_policy import LeRobotPolicy

from workstation.policy_bridge.deploy_runner import _describe_policy

STATE_DIM, ACTION_DIM, IMG = 42, 14, 96
CAMERAS = ["wrist_left", "wrist_right", "agentview"]
CHUNK = 20

# Deliberately not the identity: an unnormalisation applied wrongly still returns finite numbers of
# the right shape, so the stats have to be distinctive enough to detect it.
ACTION_MEAN, ACTION_STD = 0.25, 2.0


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A real LeRobot checkpoint, shaped like the datasets this recorder writes."""
    config = ACTConfig(
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        dim_model=64,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **{
                f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG))
                for cam in CAMERAS
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    config.device = "cpu"
    config.pretrained_backbone_weights = None

    stats = {
        "observation.state": {
            "mean": np.zeros(STATE_DIM, np.float32),
            "std": np.ones(STATE_DIM, np.float32),
        },
        "action": {
            "mean": np.full(ACTION_DIM, ACTION_MEAN, np.float32),
            "std": np.full(ACTION_DIM, ACTION_STD, np.float32),
        },
        **{
            f"observation.images.{cam}": {
                "mean": np.full((3, 1, 1), 0.5, np.float32),
                "std": np.full((3, 1, 1), 0.25, np.float32),
            }
            for cam in CAMERAS
        },
    }

    policy = get_policy_class(config.type)(config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    out = tmp_path_factory.mktemp("act_checkpoint")
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def policy(checkpoint) -> LeRobotPolicy:
    return LeRobotPolicy(pretrained_path=str(checkpoint), device="cpu")


def _observation(seed: int = 0, cameras=CAMERAS, size=(240, 320)) -> dict:
    """What the deploy client puts on the wire: dotted keys, HWC uint8 frames, a prompt."""
    rng = np.random.default_rng(seed)
    obs = {
        "observation.state": rng.normal(size=STATE_DIM).astype(np.float32),
        "prompt": "pick up the lego",
    }
    for cam in cameras:
        obs[f"observation.images.{cam}"] = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    return obs


# --------------------------------------------------------------------------------------- #
# The handshake — what lets a client configure itself for an unseen checkpoint
# --------------------------------------------------------------------------------------- #
def test_the_checkpoint_names_its_own_cameras(policy):
    """A policy trained on this recorder's data already names its image features after the
    cameras (`observation.images.wrist_left`), so the mapping needs no per-checkpoint wiring."""
    assert policy.obs_spec["image_keys"] == {cam: f"observation.images.{cam}" for cam in CAMERAS}


def test_the_client_learns_the_image_size_from_the_policy(policy):
    """The client resizes before sending; getting this from the checkpoint is what stops a
    224-trained policy from being fed 96-pixel crops."""
    assert policy.obs_spec["image_shape"] == [IMG, IMG]


def test_the_horizon_is_the_whole_predicted_chunk(policy):
    """`predict_action_chunk` returns `chunk_size` steps. Advertising `n_action_steps` -- what a
    closed-loop LeRobot rollout consumes before replanning -- would under-report the chunk this
    stack actually receives, since it replans on its own schedule."""
    assert policy.action_horizon == CHUNK


# --------------------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------------------- #
def test_inference_returns_a_chunk_the_broker_can_slice(policy):
    actions = policy.infer(_observation())["actions"]
    assert actions.shape == (CHUNK, ACTION_DIM)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()


def test_actions_come_back_unnormalised_against_the_action_stats(policy, checkpoint):
    """The load-bearing assertion.

    LeRobot's postprocessor unnormalises one `(B, action_dim)` step at a time. Handing it the
    whole `(B, chunk, action_dim)` chunk does not raise -- it broadcasts against the wrong axis
    and returns a plausible chunk of wrong numbers. So this recomputes the expected actions from
    the raw network output and the known stats, instead of just checking the shape.
    """
    obs = _observation(seed=3)

    raw = policy._policy.predict_action_chunk(policy._build_batch(obs))
    if raw.ndim != 3:
        raw = raw.unsqueeze(0)
    expected = (raw.squeeze(0) * ACTION_STD + ACTION_MEAN).detach().cpu().numpy()

    actual = policy.infer(obs)["actions"]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_the_openpi_key_convention_is_accepted_too(policy):
    """One client, two conventions: openpi checkpoints read `observation/state`, LeRobot ones
    read `observation.state`, and the deploy client sends both."""
    obs = _observation(seed=1)
    slashed = {
        "observation/state": obs["observation.state"],
        "prompt": obs["prompt"],
        **{f"observation/images/{cam}": obs[f"observation.images.{cam}"] for cam in CAMERAS},
    }
    np.testing.assert_allclose(policy.infer(slashed)["actions"], policy.infer(obs)["actions"])


def test_the_full_state_wins_over_the_openpi_joint_vector(policy):
    """Both names collapse to `observation.state`, but they are different vectors: openpi's is the
    14 joint targets, this stack's is the 42-value record the dataset holds. Picking by dict order
    would feed the policy the wrong one under the right name."""
    obs = _observation(seed=2)
    with_both = {"observation/state": np.zeros(14, np.float32), **obs}
    np.testing.assert_allclose(policy.infer(with_both)["actions"], policy.infer(obs)["actions"])


def test_a_missing_camera_is_refused_rather_than_blacked_out(policy):
    """Zero-filling would drive the robot from a black frame and report nothing wrong."""
    obs = _observation()
    del obs["observation.images.agentview"]
    with pytest.raises(ValueError, match=re.escape("observation.images.agentview")):
        policy.infer(obs)


def test_a_wrongly_sized_state_is_refused(policy):
    obs = _observation()
    obs["observation.state"] = np.zeros(7, np.float32)
    with pytest.raises(ValueError, match=re.escape("observation.state")):
        policy.infer(obs)


def test_frames_are_resized_to_what_the_policy_expects(policy):
    """The client resizes from the handshake, but a mismatch must not become a crash mid-rollout."""
    actions = policy.infer(_observation(size=(480, 640)))["actions"]
    assert actions.shape == (CHUNK, ACTION_DIM)


def test_a_shorter_chunk_can_be_requested(checkpoint):
    trimmed = LeRobotPolicy(pretrained_path=str(checkpoint), device="cpu", actions_per_chunk=5)
    assert trimmed.action_horizon == 5
    assert trimmed.infer(_observation())["actions"].shape == (5, ACTION_DIM)


def test_a_camera_named_differently_can_be_mapped(checkpoint):
    """A checkpoint trained elsewhere will not name its cameras after ours."""
    mapped = LeRobotPolicy(
        pretrained_path=str(checkpoint),
        device="cpu",
        camera_map={"top": "observation.images.agentview", **{c: f"observation.images.{c}" for c in CAMERAS[:2]}},
    )
    assert mapped.obs_spec["image_keys"]["top"] == "observation.images.agentview"


def test_a_camera_map_naming_an_absent_key_fails_at_startup(checkpoint):
    """At construction, where it is a typo -- not at the first rollout, where it is a robot."""
    with pytest.raises(ValueError, match="does not have"):
        LeRobotPolicy(pretrained_path=str(checkpoint), device="cpu", camera_map={"top": "observation.images.nope"})


# --------------------------------------------------------------------------------------- #
# Over the wire — the path a real deployment takes
# --------------------------------------------------------------------------------------- #
def test_a_lerobot_checkpoint_serves_over_the_deploy_wire(policy):
    """Server + client + broker, the same three pieces deploy uses.

    This is where a LeRobot policy has to behave like any other: the client reads the camera
    names and image size out of the handshake, and the broker runs one inference per chunk while
    handing back one step at a time.
    """
    import threading

    from yam_policy import ActionChunkBroker, WebsocketClientPolicy, WebsocketPolicyServer
    from yam_policy.serve import build_metadata

    from tests._util import free_port, wait_port

    # The metadata the launcher actually builds, not a re-creation of it.
    metadata = build_metadata(policy, "yam_policy.policies.lerobot_policy:LeRobotPolicy", {})
    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata=metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"

    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    meta = client.get_server_metadata()
    assert meta["image_keys"] == {cam: f"observation.images.{cam}" for cam in CAMERAS}
    assert meta["image_shape"] == [IMG, IMG]
    # ...and it can say what it is talking to, rather than assuming openpi.
    assert _describe_policy(meta) == "LeRobot · act"

    broker = ActionChunkBroker(client)
    obs = _observation(seed=5)
    steps = [broker.infer(obs)["actions"] for _ in range(CHUNK + 2)]

    assert all(step.shape == (ACTION_DIM,) for step in steps)
    assert not np.allclose(steps[0], steps[1]), "a chunk of identical steps means a slicing bug"


def test_the_deploy_client_feeds_a_lerobot_policy_with_no_conversion(policy):
    """The whole scenario in one assertion: the client configures itself from the handshake, and
    what it puts on the wire is already what the checkpoint reads.

    `_build_obs` is the client's only observation-shaping step, so if its output needs touching up
    before the policy accepts it, "grab a LeRobot model and deploy" is not actually true.
    """
    from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
    from workstation.policy_bridge.config import BridgeConfig
    from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner

    rng = np.random.default_rng(11)
    images = {cam: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for cam in CAMERAS}
    runner = DeploymentPolicyRunner(BridgeConfig(), RecorderConfig(mock=True), lambda: images)

    # Exactly what _connect_policy does with the server's metadata.
    metadata = {"action_horizon": int(policy.action_horizon), **policy.obs_spec}
    runner.cfg.image_keys = metadata["image_keys"]
    runner._image_shape = runner._image_shape_from_meta(metadata)
    assert runner._image_shape == (IMG, IMG)

    robot_obs = {
        arm: {
            "pos": np.zeros(ARM_DOF, np.float32),
            "vel": np.zeros(ARM_DOF, np.float32),
            "eff": np.zeros(ARM_DOF, np.float32),
            "leader_pos": np.zeros(6, np.float32),
            "eef": np.zeros(7, np.float32),
        }
        for arm in ARMS
    }
    obs = runner._build_obs(robot_obs, images)

    assert set(metadata["image_keys"].values()) <= set(obs), "the client did not send the policy's cameras"
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (CHUNK, ACTION_DIM)

    # Every non-image column this recorder writes, because LeRobot's trainer takes all of them
    # as policy inputs -- a checkpoint asking for one the client never sends cannot be deployed
    # at all, and this was found by training on real data rather than by reading the code.
    for column in ("observation.state", "observation.leader", "observation.eef", "observation.control_mode"):
        assert column in obs or column.replace(".", "/", 1) in obs, column


def test_a_checkpoint_that_reads_the_leader_is_flagged_at_load(caplog):
    """`observation.leader` is the teleop command itself, so a policy given it can copy the
    answer -- and nothing downstream can tell: the loss is excellent for that very reason. Load
    time is the last cheap moment to say so."""
    import logging

    with caplog.at_level(logging.WARNING):
        LeRobotPolicy._warn_about_leaking_inputs({"observation.state": 1, "observation.leader": 1})
    assert any("observation.leader" in r.getMessage() for r in caplog.records)


def test_an_ordinary_checkpoint_is_not_warned_about(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        LeRobotPolicy._warn_about_leaking_inputs({"observation.state": 1, "observation.eef": 1})
    assert not caplog.records


def test_torch_is_not_left_in_training_mode(policy):
    """Dropout at deploy time would make the robot's actions non-deterministic."""
    assert not policy._policy.training
    a = policy.infer(_observation(seed=9))["actions"]
    b = policy.infer(_observation(seed=9))["actions"]
    np.testing.assert_allclose(a, b)
