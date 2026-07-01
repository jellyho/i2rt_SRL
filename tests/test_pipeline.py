"""Pure-logic pipeline tests: episode gate, dataset doctor, async writer + disk guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter, dataset_dir, dataset_info
from workstation.lerobot_recorder.doctor import outcomes_by_episode, summarize_outcomes
from workstation.lerobot_recorder.episode_gate import EV_IDLE, EV_RECORD, EV_START, EV_STOP, EpisodeGate


# ---------------------------------------------------------------- episode gate
def test_episode_gate_transitions():
    g = EpisodeGate()
    assert g.update("ENGAGED") == EV_IDLE  # not armed -> nothing
    g.arm()
    assert g.update("IDLE") == EV_IDLE
    assert g.update("ENGAGED") == EV_START
    assert g.update("ENGAGED") == EV_RECORD
    assert g.update("HOMING") == EV_RECORD  # records through the homing return
    assert g.update("IDLE") == EV_STOP
    assert g.update("IDLE") == EV_IDLE  # episode already closed


# ---------------------------------------------------------------- doctor
def test_summarize_outcomes(tmp_path):
    rows = [
        {"outcome": "success", "task": "pick", "frames": 10},
        {"outcome": "fail", "task": "pick", "frames": 8},
        {"outcome": "success", "task": "stack", "frames": 12},
    ]
    (tmp_path / "outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = summarize_outcomes(str(tmp_path))
    assert s["exists"] and s["episodes"] == 3 and s["frames_total"] == 30
    assert s["outcomes"]["success"] == 2 and s["outcomes"]["fail"] == 1
    assert abs(s["success_rate"] - 2 / 3) < 1e-9
    assert s["by_task"]["pick"] == {"success": 1, "fail": 1}


def test_summarize_outcomes_missing(tmp_path):
    assert summarize_outcomes(str(tmp_path))["exists"] is False


def test_outcomes_by_episode(tmp_path):
    rows = [
        {"episode": 0, "outcome": "success"},
        {"episode": 1, "outcome": "fail"},
        {"episode": 2, "outcome": "discard"},
    ]
    (tmp_path / "outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = outcomes_by_episode(str(tmp_path))
    assert m == {0: "success", 1: "fail", 2: "discard"}
    assert outcomes_by_episode(str(tmp_path / "nope")) == {}


# ---------------------------------------------------------------- async writer
def _frame():
    return {
        "images": {"agentview": np.zeros((4, 4, 3), np.uint8)},
        "observation.state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
    }


def test_async_writer_saves_each_queued_episode(tmp_path):
    cfg = RecorderConfig(root=str(tmp_path), mock=True)
    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    w.open(_frame())
    for _ in range(3):
        w.submit([_frame() for _ in range(5)], "success", "pick")
    w.finalize()  # drains the queue
    assert w.num_episodes == 3
    # the dataset (and its outcomes sidecar) lives at <root>/<name>
    sidecar = Path(dataset_dir(str(tmp_path), cfg.repo_id)) / "outcomes.jsonl"
    assert len(sidecar.read_text().splitlines()) == 3


def test_async_writer_resume_preserves_encoding_kwargs(tmp_path, monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.num_episodes = 7
            self.episodes_since_last_encoding = 0

        def finalize(self):
            pass

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        lambda: FakeDataset,
    )

    cfg = RecorderConfig(
        repo_id="test/yam",
        root=str(tmp_path),
        mock=False,
        resume=True,
        vcodec="auto",
        batch_encoding_size=4,
        encoder_threads=2,
    )
    Path(dataset_dir(str(tmp_path), cfg.repo_id)).mkdir(parents=True)

    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    w.open(_frame())
    w.finalize()

    assert calls == [
        (
            ("test/yam",),
            {
                "root": str(tmp_path / "yam"),
                "vcodec": "auto",
                "batch_encoding_size": 4,
                "encoder_threads": 2,
            },
        )
    ]


def test_async_writer_resume_recovers_missing_outcome_rows(tmp_path, monkeypatch):
    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.num_episodes = 3
            self.episodes_since_last_encoding = 0
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [
                        {"episode_index": 0, "tasks": ["pick"], "length": 10},
                        {"episode_index": 1, "tasks": ["pick"], "length": 11},
                        {"episode_index": 2, "tasks": ["place"], "length": 12},
                    ]
                },
            )()

        def finalize(self):
            pass

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        lambda: FakeDataset,
    )

    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=False, resume=True)
    ds_dir = Path(dataset_dir(str(tmp_path), cfg.repo_id))
    ds_dir.mkdir(parents=True)
    sidecar = ds_dir / "outcomes.jsonl"
    sidecar.write_text(json.dumps({"episode": 1, "outcome": "success", "task": "pick", "frames": 11}) + "\n")

    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    w.open(_frame())
    w.finalize()

    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [row["episode"] for row in rows] == [0, 1, 2]
    assert rows[0]["outcome"] == "unknown"
    assert rows[0]["task"] == "pick"
    assert rows[0]["frames"] == 10
    assert rows[0]["recovered"] is True
    assert rows[1]["outcome"] == "success"
    assert rows[2]["outcome"] == "unknown"
    assert rows[2]["task"] == "place"


def test_dataset_info_prefers_lerobot_episode_count(tmp_path):
    ds_dir = tmp_path / "yam"
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": 41}))
    (ds_dir / "outcomes.jsonl").write_text(json.dumps({"episode": 40, "outcome": "success"}) + "\n")

    assert dataset_info(str(ds_dir)) == {"exists": True, "episodes": 41}


def test_disk_guard_refuses_save(tmp_path):
    cfg = RecorderConfig(root=str(tmp_path), mock=True, min_free_gb=1e9)  # impossible threshold
    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    w.open(_frame())
    w.submit([_frame()], "success", "pick")
    w.finalize()
    assert w.low_disk is True
    assert w.num_episodes == 0
