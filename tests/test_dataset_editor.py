"""Dataset editor: relabel (a two-column rewrite of one episode) + the metadata readers.

The structural edits (delete_episodes / set_task) go through LeRobot's dataset_tools and
carry the verdict columns along like any other feature; here we lock the lightweight
relabel path and the readers, against a real (tiny) LeRobotDataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from workstation.lerobot_recorder import outcomes as _outcomes
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_editor import (
    DatasetEditor,
    detect_homing_start,
    repair_length_consistency,
    video_length_mismatches,
)
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter


def _frame():
    return {
        "images": {"agentview": np.zeros((32, 32, 3), np.uint8)},
        "observation.state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
    }


def _record(tmp_path, episodes, repo_id="me/ds"):
    """``[(outcome, n_frames), ...]`` -> a real dataset at <tmp_path>/<name>."""
    cfg = RecorderConfig(repo_id=repo_id, root=str(tmp_path), mock=False, encoding_backend="pyav")
    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (32, 32, 3)})
    w.open(_frame())
    for outcome, n in episodes:
        w.submit([_frame() for _ in range(n)], outcome, "t")
    w.finalize()
    return DatasetEditor(repo_id, str(tmp_path))


def test_relabel_rewrites_only_that_episode(tmp_path):
    pytest.importorskip("lerobot")
    ed = _record(tmp_path, [("success", 4), ("success", 5), (None, 3)])
    ed.relabel(1, "fail")
    assert ed.outcomes_by_episode() == {0: "success", 1: "fail", 2: "unknown"}
    # the terminal frame carries it, the others stay untouched
    import pandas as pd

    df = pd.read_parquet(sorted((tmp_path / "ds" / "data").rglob("*.parquet"))[0])
    ep1 = df[df["episode_index"] == 1].sort_values("frame_index")
    assert ep1["next.done"].tolist() == [False, False, False, False, True]
    assert ep1["next.success"].tolist() == [False] * 5
    # and the dataset-level stats followed: 1 success frame out of 12
    stats = json.load(open(tmp_path / "ds" / "meta" / "stats.json"))
    assert abs(stats["next.success"]["mean"][0] - 1 / 12) < 1e-9
    assert abs(stats["next.done"]["mean"][0] - 2 / 12) < 1e-9


def test_relabel_can_withdraw_a_verdict(tmp_path):
    pytest.importorskip("lerobot")
    ed = _record(tmp_path, [("fail", 3)])
    ed.relabel(0, "unknown")
    assert ed.outcomes_by_episode() == {0: "unknown"}
    ed.relabel(0, "success")
    assert ed.outcomes_by_episode() == {0: "success"}


def test_relabel_refuses_a_pre_schema_dataset(tmp_path):
    ds_dir = tmp_path / "old"
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "features": {"observation.state": {"dtype": "float32", "shape": [42]}}})
    )
    ed = DatasetEditor("me/old", str(tmp_path))
    with pytest.raises(_outcomes.OutcomeColumnsMissing, match="migrate-outcomes"):
        ed.relabel(0, "fail")
    assert ed.outcomes_by_episode() == {}  # a listing degrades to "no verdicts", it does not crash


def test_frames_by_episode_comes_from_the_metadata(tmp_path):
    pytest.importorskip("lerobot")
    ed = _record(tmp_path, [("success", 4), ("fail", 7)])
    assert ed.frames_by_episode() == {0: 4, 1: 7}


# --------------------------------------------------------------- video-length consistency
def _write_meta(ds_dir, fps, rows, cams=("agentview", "wrist_left")):
    """Write a minimal meta/{info.json,episodes/...} with one episode per row.

    Each row: (episode_index, length, {cam: (from_frame, to_frame)}) — timestamps are
    frame/fps. Mirrors LeRobot v3.0's episode metadata closely enough for the checks.
    """
    import pandas as pd

    (ds_dir / "meta").mkdir(parents=True, exist_ok=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"fps": fps}))
    ep_dir = ds_dir / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ep, length, windows in rows:
        rec = {"episode_index": ep, "length": length}
        for cam in cams:
            ff, tf = windows[cam]
            rec[f"videos/observation.images.{cam}/from_timestamp"] = ff / fps
            rec[f"videos/observation.images.{cam}/to_timestamp"] = tf / fps
        records.append(rec)
    pd.DataFrame(records).to_parquet(ep_dir / "file-000.parquet", index=False)


def test_video_length_mismatch_detected(tmp_path):
    pytest.importorskip("pandas")
    ds = tmp_path / "ds"
    # ep0 consistent (100 frames); ep1 agentview dropped a trailing frame (99 vs 100).
    _write_meta(ds, 30, [
        (0, 100, {"agentview": (0, 100), "wrist_left": (0, 100)}),
        (1, 100, {"agentview": (100, 199), "wrist_left": (100, 200)}),
    ])
    bad = video_length_mismatches(str(ds))
    assert len(bad) == 1
    assert bad[0]["episode"] == 1 and bad[0]["camera"].endswith("agentview")
    assert bad[0]["length"] == 100 and bad[0]["derived"] == 99


def test_repair_snaps_timestamp_to_length(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    ds = tmp_path / "ds"
    _write_meta(ds, 30, [
        (0, 100, {"agentview": (0, 100), "wrist_left": (0, 100)}),
        (1, 100, {"agentview": (100, 199), "wrist_left": (100, 200)}),
    ])
    n = repair_length_consistency(str(ds))
    assert n == 1
    assert video_length_mismatches(str(ds)) == []
    df = pd.read_parquet(ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    to_frame = round(float(df.iloc[1]["videos/observation.images.agentview/to_timestamp"]) * 30)
    assert to_frame == 200  # snapped up so the window spans exactly length (100) frames


def test_repair_is_noop_on_consistent_dataset(tmp_path):
    pytest.importorskip("pandas")
    ds = tmp_path / "ds"
    _write_meta(ds, 30, [(0, 100, {"agentview": (0, 100), "wrist_left": (0, 100)})])
    assert repair_length_consistency(str(ds)) == 0
    assert video_length_mismatches(str(ds)) == []


# ------------------------------------------------------------- homing (control_mode) trim
def _write_data(ds_dir, episodes):
    """Write data/chunk-000/file-000.parquet with control_mode=0 for the given episode lengths."""
    import numpy as np
    import pandas as pd

    (ds_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (ds_dir / "meta").mkdir(parents=True, exist_ok=True)
    (ds_dir / "meta" / "stats.json").write_text(json.dumps({"observation.control_mode": {"min": [0.0]}}))
    rows = []
    for ep, length in episodes:
        for fi in range(length):
            rows.append({"episode_index": ep, "frame_index": fi, "observation.control_mode": np.float32(0.0)})
    pd.DataFrame(rows).to_parquet(ds_dir / "data" / "chunk-000" / "file-000.parquet", index=False)


def test_set_homing_tail_marks_only_the_tail(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    ds = tmp_path / "ds"
    _write_data(ds, [(0, 100), (1, 50)])
    ed = DatasetEditor("me/ds", str(tmp_path))
    n = ed.set_homing_tail(0, 80)  # frames 80..99 of ep0
    assert n == 20
    df = pd.read_parquet(ds / "data" / "chunk-000" / "file-000.parquet")
    ep0 = df[df.episode_index == 0].sort_values("frame_index")["observation.control_mode"].to_numpy()
    assert (ep0[:80] == 0.0).all() and (ep0[80:] == 4.0).all()  # 4.0 == homing
    # other episode untouched
    assert (df[df.episode_index == 1]["observation.control_mode"] == 0.0).all()


def test_clear_homing_resets_to_teleop(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    ds = tmp_path / "ds"
    _write_data(ds, [(0, 100)])
    ed = DatasetEditor("me/ds", str(tmp_path))
    ed.set_homing_tail(0, 90)
    cleared = ed.clear_homing(0)
    assert cleared == 10
    df = pd.read_parquet(ds / "data" / "chunk-000" / "file-000.parquet")
    assert (df["observation.control_mode"] == 0.0).all()


# ------------------------------------------------------------- auto homing detection
def _gripper_close_signal(task_len, close_len, open_val=0.99, floor=0.003):
    """An episode that stays open through the task then ramps to a closed floor and holds."""
    task = np.full(task_len, open_val)
    ramp = np.linspace(open_val, floor, close_len // 2)
    hold = np.full(close_len - ramp.size, floor)
    return np.concatenate([task, ramp, hold])


def test_detect_homing_start_finds_final_close():
    g = _gripper_close_signal(task_len=200, close_len=80)
    start = detect_homing_start(g)
    assert start is not None
    # homing starts where the final descent begins (~end of the open task segment)
    assert 195 <= start <= 205


def test_detect_homing_returns_none_when_not_ending_closed():
    # ends open (never closes) → no homing to guess
    g = np.concatenate([np.full(150, 0.99), np.linspace(0.99, 0.6, 50)])
    assert detect_homing_start(g) is None


def test_detect_homing_ignores_midtask_close_that_reopens():
    # closes mid-task (a grasp) but reopens and ends open → not a homing tail
    g = np.concatenate([np.full(80, 0.9), np.full(40, 0.05), np.full(120, 0.9)])
    assert detect_homing_start(g) is None
