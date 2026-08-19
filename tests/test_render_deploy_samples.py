"""Offline rendering: the candidate fan drawn onto a real recorded episode.

`yam-data render-samples` is the end of the multi-sample chain — a server returns N candidate
chunks, the recorder stores them as an `action_samples` column, and this turns that column back
into a video. Every link before it is tested; this one was not, which is the link where a wrong
assumption is least visible: it produces an mp4 either way.

The input here is written with the recorder's own `AsyncDatasetWriter`, the same path deploy
uses, so the test also pins the column layout the renderer reads back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("mujoco")  # forward kinematics
pytest.importorskip("mink")
pytest.importorskip("imageio")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "policy_serving"))

from workstation.lerobot_recorder.config import ACTION_DIM, STATE_DIM, RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
from workstation.lerobot_recorder.render_deploy_samples import render

CAMERAS = ("wrist_left", "wrist_right", "agentview")
IMG = (64, 64, 3)
HORIZON = 5
CANDIDATES = 3
FRAMES = HORIZON * 4  # four whole replans


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory) -> Path:
    """An episode with an `action_samples` column, written the way deploy writes one."""
    root = tmp_path_factory.mktemp("samples_run")
    shapes = {cam: IMG for cam in CAMERAS}
    cfg = RecorderConfig(repo_id="t/samples", root=str(root), mock=False, fps=30, task="fan")
    writer = AsyncDatasetWriter(
        cfg, list(CAMERAS), shapes, extra_features={"action_samples": (CANDIDATES, ACTION_DIM)}
    )

    rng = np.random.default_rng(0)

    def frame(i: int) -> dict:
        state = np.zeros(STATE_DIM, np.float32)
        state[:7] = np.linspace(0, 0.4, 7) * (i / FRAMES)  # left arm joint positions
        # Candidates fan out around the executed action; candidate 0 is the executed one.
        executed = np.zeros(ACTION_DIM, np.float32)
        executed[:7] = state[:7]
        samples = np.repeat(executed[None, :], CANDIDATES, axis=0)
        samples[1:, :7] += rng.normal(0, 0.05, (CANDIDATES - 1, 7)).astype(np.float32)
        return {
            "images": {cam: rng.integers(0, 255, IMG, dtype=np.uint8) for cam in CAMERAS},
            "observation.state": state,
            "action": executed,
            "action_samples": samples.reshape(-1),
        }

    writer.open(frame(0))
    for i in range(FRAMES):
        writer.stream_frame(frame(i), "fan")
    writer.end_episode("success", "fan")
    writer.finalize()
    return root / "samples"


def _args(dataset: Path, out: Path, **over) -> argparse.Namespace:
    base = dict(
        repo_id="t/samples",
        root=str(dataset.parent),
        # A path that does not exist, so the loaders fail-soft to no calibrated extrinsics (CAD
        # wrist, raw agentview) -- keeps the test hermetic instead of discovering the repo config.
        config=str(dataset.parent / "no_such_config.yaml"),
        source="samples",
        episode=0,
        wrists=["left", "right"],
        agentview_arms=["left", "right"],
        horizon=HORIZON,
        candidates=CANDIDATES,
        replans=0,
        hold=2,
        height=180,
        fps=10,
        out=str(out),
        fx=430.0,
        fy=430.0,
        cx=320.0,
        cy=240.0,
        agent_fx=390.0,
        agent_fy=390.0,
        agent_cx=320.0,
        agent_cy=240.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------------------- #
# The render
# --------------------------------------------------------------------------------------- #
def test_it_writes_a_video_for_a_real_recorded_episode(recorded_run, tmp_path):
    out = render(_args(recorded_run, tmp_path / "fan.mp4"))
    assert out.is_file() and out.stat().st_size > 0


def test_every_tick_is_rendered_not_just_each_chunk_start(recorded_run, tmp_path):
    """The whole point of per-tick rendering: every frame of the episode is drawn (times --hold),
    so you watch each chunk consumed -- not one held frame per chunk."""
    import imageio.v3 as iio

    out = render(_args(recorded_run, tmp_path / "held.mp4", hold=3))
    written = iio.imread(out, plugin="pyav")
    assert len(written) == FRAMES * 3  # every one of the episode's frames, held 3x


def test_the_chunk_limit_is_honoured(recorded_run, tmp_path):
    import imageio.v3 as iio

    # 2 chunks of HORIZON ticks each, held 2x
    out = render(_args(recorded_run, tmp_path / "two.mp4", replans=2, hold=2))
    assert len(iio.imread(out, plugin="pyav")) == 2 * HORIZON * 2


def test_a_horizon_longer_than_the_episode_renders_one_chunk(recorded_run, tmp_path):
    """A horizon past the episode length just makes ONE (shorter) chunk covering all frames --
    the whole trajectory is still rendered, tick by tick, rather than raising."""
    import imageio.v3 as iio

    out = render(_args(recorded_run, tmp_path / "one.mp4", horizon=FRAMES + 10, hold=1))
    assert len(iio.imread(out, plugin="pyav")) == FRAMES  # every frame, as a single chunk


def test_an_episode_that_is_not_there_is_refused(recorded_run, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        render(_args(recorded_run, tmp_path / "nope.mp4", episode=7))


def test_each_wrist_renders(recorded_run, tmp_path):
    """The right arm reads a different action slice and a different state slice; getting either
    wrong still draws a path, just of the wrong joints."""
    for arm in ("left", "right"):
        assert render(_args(recorded_run, tmp_path / f"{arm}.mp4", wrists=[arm])).is_file()


def test_render_all_three_cameras_hstacks_three_panels(recorded_run, tmp_path):
    """Default is agentview + both wrists -> a 3-panel-wide frame, each panel 4:3 (not squashed)."""
    import imageio.v3 as iio

    out = render(_args(recorded_run, tmp_path / "all.mp4"))
    frame = iio.imread(out, index=0)
    assert frame.shape[1] == 3 * (frame.shape[0] * 640 // 480)  # 3 panels, each 4:3 of the height


def test_source_action_needs_no_action_samples(recorded_run, tmp_path):
    """--source action reads the plain action column, so it renders even with candidates unset."""
    out = render(_args(recorded_run, tmp_path / "act.mp4", source="action", candidates=None))
    assert out.is_file()


def test_the_fan_is_coloured_by_the_critic_s_own_ranking():
    """A value-guided run records what the critic thought of each candidate, so the fan can show
    the value landscape the decision was made on instead of a spread of look-alike options.

    Normalised per replan: the absolute numbers are arbitrary (cost-to-goal runs to -2777), the
    useful question is which candidate the critic preferred here."""
    from workstation.lerobot_recorder.render_deploy_samples import _value_color

    worst, best = _value_color(-20.0, -20.0, -5.0), _value_color(-5.0, -20.0, -5.0)
    assert worst != best
    assert best[0] > worst[0], "the preferred candidate should read warmer"
    assert worst[2] > best[2], "...and the rejected one colder"
    # A replan the critic saw nothing to choose between must not paint a false gradient.
    flat = {_value_color(v, -7.0, -7.0) for v in (-7.0, -7.0)}
    assert len(flat) == 1


def test_the_executed_candidate_is_the_one_the_critic_picked():
    """`critic_choice` is read from the recording: highlighting index 0 would draw the wrong path
    as executed on any run where the critic picked something else."""
    import numpy as np

    from workstation.lerobot_recorder.render_deploy_samples import _load_critic

    class _Reader:
        def get_extra(self, ep, frame, key, shape):
            return np.array([-9.0, -3.0, -7.0]) if key == "critic_scores" else None

        def get_scalar(self, ep, frame, key):
            return 1.0 if key == "critic_choice" else None

    scores, chosen = _load_critic(_Reader(), 0, 0, 3)
    assert chosen == 1 and float(scores[chosen]) == -3.0

    class _Plain(_Reader):
        def get_extra(self, ep, frame, key, shape):
            return None

    assert _load_critic(_Plain(), 0, 0, 3) == (None, 0)


def test_a_constant_chunk_index_is_treated_as_no_information():
    """Rollouts recorded while the provenance was written as a constant 0 carry the column but
    nothing in it. Believing it would draw the entire episode as one chunk."""
    from workstation.lerobot_recorder.render_deploy_samples import _recorded_chunk_starts

    class _Reader:
        def __init__(self, values):
            self.values = values

        def has_feature(self, key):
            return True

        def get_scalar(self, ep, frame, key):
            return self.values[frame]

    assert _recorded_chunk_starts(_Reader([0.0] * 40), 0, 40) is None
    assert _recorded_chunk_starts(_Reader([0.0] * 10 + [1.0] * 10), 0, 20) == [0, 10]
