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


def test_async_writer_stops_cleanly_after_first_save_failure(tmp_path, monkeypatch):
    cfg = RecorderConfig(root=str(tmp_path), mock=True)
    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    writer.open(_frame())
    monkeypatch.setattr(writer, "_save_episode", lambda *_args: (_ for _ in ()).throw(RuntimeError("encode")))

    writer.submit([_frame()], "success", "pick")
    writer.submit([_frame()], "success", "pick")
    writer.finalize()

    assert writer.num_episodes == 0
    assert writer.queue_depth == 0
    assert writer.progress["failed"] is True
    assert writer.progress["failed_episodes"] == 1
    assert "encode" in writer.progress["last_error"]

    try:
        writer.submit([_frame()], "success", "pick")
    except RuntimeError as exc:
        assert "save failure" in str(exc)
    else:
        raise AssertionError("a failed writer must reject later episodes")


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
        encoding_backend="pyav",
        batch_encoding_size=4,
        encoder_threads=2,
    )
    ds_dir = Path(dataset_dir(str(tmp_path), cfg.repo_id))
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": 7}))

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
                # Always, under PyAV: the alternative stages every frame as a PNG.
                "streaming_encoding": True,
            },
        )
    ]


def test_async_writer_resume_cleans_interrupted_next_episode(tmp_path, monkeypatch):
    cleaned = []

    class FakeWriter:
        def cleanup_interrupted_episode(self, episode_index):
            cleaned.append(episode_index)

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.num_episodes = 7
            self.episodes_since_last_encoding = 0
            self.writer = FakeWriter()

        def finalize(self):
            pass

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        lambda: FakeDataset,
    )
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=False, resume=True)
    ds_dir = Path(dataset_dir(str(tmp_path), cfg.repo_id))
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": 7}))

    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    writer.open(_frame())
    writer.finalize()

    assert cleaned == [7]


def test_async_writer_resume_cleans_interrupted_episode_with_legacy_lerobot_api(tmp_path, monkeypatch):
    cleaned = []

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.num_episodes = 7
            self.episodes_since_last_encoding = 0
            self.writer = None
            self.episode_buffer = None

        def create_episode_buffer(self):
            return {"size": 0, "episode_index": self.num_episodes}

        def clear_episode_buffer(self, delete_images=True):
            assert self.episode_buffer is not None
            cleaned.append(delete_images)

        def finalize(self):
            pass

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        lambda: FakeDataset,
    )
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=False, resume=True)
    ds_dir = Path(dataset_dir(str(tmp_path), cfg.repo_id))
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": 7}))
    stale_dir = ds_dir / "images" / "observation.images.agentview" / "episode-000007"
    stale_dir.mkdir(parents=True)
    (stale_dir / "frame-000000.png").write_bytes(b"")

    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    writer.open(_frame())
    writer.finalize()

    assert cleaned == [True]
    assert not stale_dir.exists()


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
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3}))
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


def test_async_writer_rejects_incomplete_local_resume_without_hub_lookup(tmp_path, monkeypatch):
    imported = False

    def fake_import():
        nonlocal imported
        imported = True
        return object

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        fake_import,
    )
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=False, resume=True)
    Path(dataset_dir(str(tmp_path), cfg.repo_id)).mkdir(parents=True)
    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})

    try:
        writer.open(_frame())
    except RuntimeError as exc:
        assert "meta/info.json is missing" in str(exc)
    else:
        raise AssertionError("incomplete resume should fail")
    assert imported is False


def test_async_writer_recreates_empty_local_resume_without_hub_lookup(tmp_path, monkeypatch):
    calls = []

    class FakeDataset:
        @classmethod
        def create(cls, **kwargs):
            calls.append(("create", kwargs))
            return cls()

        def finalize(self):
            pass

    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer._import_lerobot_dataset",
        lambda: FakeDataset,
    )
    cfg = RecorderConfig(repo_id="test", root=str(tmp_path), mock=False, resume=True)
    ds_dir = Path(dataset_dir(str(tmp_path), cfg.repo_id))
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0})
    )

    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (4, 4, 3)})
    writer.open(_frame())
    writer.finalize()

    assert [kind for kind, _ in calls] == ["create"]
    assert calls[0][1]["root"] == str(ds_dir)


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


def test_streaming_memory_does_not_grow_with_episode_length(tmp_path):
    """The writer must not hold an episode whole.

    Three 640x480 cameras are ~2.8 MB a frame, so buffering a 90 s episode as a list is
    ~7.5 GB -- which is what pushed the machine into swap and got the recorder OOM-killed
    mid-episode. Streaming caps the resident set at the queue instead. Measured, in separate
    processes so the high-water mark is not carried over:

        frames   buffered   streamed
           300     2.67 GB    2.03 GB
           900     4.34 GB    2.05 GB
          2700     9.21 GB    2.05 GB

    This test asserts the shape of that -- flat, not linear -- on a small enough scale to run
    in CI, by counting how many frames are ever resident at once rather than measuring RSS.
    """
    import numpy as np

    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter

    cams = {"cam": (32, 32, 3)}

    def frame(i):
        return {
            "state": np.zeros(42, np.float32),
            "action": np.zeros(14, np.float32),
            "leader": np.zeros(12, np.float32),
            "eef": np.zeros(14, np.float32),
            "images": {"cam": np.full((32, 32, 3), i % 255, np.uint8)},
            "task": "t",
        }

    cfg = RecorderConfig(repo_id="t/stream", root=str(tmp_path), mock=False, fps=30)
    writer = AsyncDatasetWriter(cfg, list(cams), cams)
    writer.open(frame(0))
    assert writer.supports_streaming()

    high_water = 0
    for i in range(400):
        writer.stream_frame(frame(i))
        high_water = max(high_water, writer._frame_queue.qsize())
    writer.end_episode("success", "t")
    writer.finalize()

    # Bounded by the queue, not by the 400 frames that went through it.
    assert high_water <= writer._frame_queue.maxsize, high_water

    info = json.loads((tmp_path / "stream" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 400, "every streamed frame must reach the dataset"


def test_streaming_abort_drops_the_episode_and_leaves_the_writer_usable(tmp_path):
    """The review 'Delete', once frames are already inside LeRobot's buffer."""
    import numpy as np

    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter

    cams = {"cam": (32, 32, 3)}

    def frame(i):
        return {
            "state": np.zeros(42, np.float32),
            "action": np.zeros(14, np.float32),
            "leader": np.zeros(12, np.float32),
            "eef": np.zeros(14, np.float32),
            "images": {"cam": np.full((32, 32, 3), i % 255, np.uint8)},
            "task": "t",
        }

    cfg = RecorderConfig(repo_id="t/abort", root=str(tmp_path), mock=False, fps=30)
    writer = AsyncDatasetWriter(cfg, list(cams), cams)
    writer.open(frame(0))

    for i in range(20):
        writer.stream_frame(frame(i))
    writer.abort_episode()
    for i in range(25):
        writer.stream_frame(frame(i))
    writer.end_episode("success", "t")
    writer.finalize()

    info = json.loads((tmp_path / "abort" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1, "the aborted episode must not be saved"
    assert info["total_frames"] == 25, "and must not contribute frames to the kept one"
