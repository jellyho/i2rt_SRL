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
from workstation.lerobot_recorder.render_deploy_samples import _replan_starts, render

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
        episode=0,
        arm="left",
        horizon=HORIZON,
        candidates=CANDIDATES,
        replans=0,
        hold=2,
        fps=10,
        out=str(out),
        fx=430.0,
        fy=430.0,
        cx=320.0,
        cy=240.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------------------- #
# Grouping ticks back into replans
# --------------------------------------------------------------------------------------- #
def test_only_whole_replans_are_rendered():
    """A trailing partial replan has fewer than `horizon` snapshots, so its chunk cannot be
    reassembled — rendering it would draw a fan from frames belonging to two different ones."""
    assert _replan_starts(20, 5) == [0, 5, 10, 15]
    assert _replan_starts(23, 5) == [0, 5, 10, 15]
    assert _replan_starts(4, 5) == []


# --------------------------------------------------------------------------------------- #
# The render
# --------------------------------------------------------------------------------------- #
def test_it_writes_a_video_for_a_real_recorded_episode(recorded_run, tmp_path):
    out = render(_args(recorded_run, tmp_path / "fan.mp4"))
    assert out.is_file() and out.stat().st_size > 0


def test_every_replan_is_held_for_the_requested_frames(recorded_run, tmp_path):
    """The frame count is the one thing that says the whole episode was walked rather than a
    file merely being produced."""
    import imageio.v3 as iio

    out = render(_args(recorded_run, tmp_path / "held.mp4", hold=3))
    written = iio.imread(out, plugin="pyav")
    assert len(written) == len(_replan_starts(FRAMES, HORIZON)) * 3


def test_the_replan_limit_is_honoured(recorded_run, tmp_path):
    import imageio.v3 as iio

    out = render(_args(recorded_run, tmp_path / "two.mp4", replans=2, hold=2))
    assert len(iio.imread(out, plugin="pyav")) == 4


def test_a_horizon_longer_than_the_episode_is_refused(recorded_run, tmp_path):
    """Rather than writing an empty video that looks like a rendering bug."""
    with pytest.raises(SystemExit, match="shorter than"):
        render(_args(recorded_run, tmp_path / "nope.mp4", horizon=FRAMES + 10))


def test_an_episode_that_is_not_there_is_refused(recorded_run, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        render(_args(recorded_run, tmp_path / "nope.mp4", episode=7))


def test_both_arms_render(recorded_run, tmp_path):
    """The right arm reads a different action slice and a different state slice; getting either
    wrong still draws a fan, just of the wrong joints."""
    for arm in ("left", "right"):
        assert render(_args(recorded_run, tmp_path / f"{arm}.mp4", arm=arm)).is_file()
