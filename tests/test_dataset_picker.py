"""Setup-page dataset picker helpers: list datasets under the root dir and read
the task strings already used by an existing dataset (LeRobot v3 tasks.parquet,
v2 tasks.jsonl fallback).
"""

from __future__ import annotations

import json

from workstation.lerobot_recorder.dataset_writer import dataset_tasks, list_datasets


def test_list_datasets_returns_subdirs_sorted(tmp_path):
    (tmp_path / "yam_pick" / "meta").mkdir(parents=True)
    (tmp_path / "stack").mkdir()  # no meta yet (fresh dataset) — still listed
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert list_datasets(str(tmp_path)) == ["stack", "yam_pick"]


def test_list_datasets_missing_root(tmp_path):
    assert list_datasets(str(tmp_path / "nope")) == []


def test_dataset_tasks_parquet(tmp_path):
    import pandas as pd

    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    pd.DataFrame({"task_index": [0, 1]}, index=["pick the cube", "stack blocks"]).to_parquet(
        meta / "tasks.parquet"
    )
    assert dataset_tasks(str(tmp_path / "ds")) == ["pick the cube", "stack blocks"]


def test_dataset_tasks_jsonl_fallback(tmp_path):
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    lines = [{"task_index": 0, "task": "open the drawer"}, {"task_index": 1, "task": "close it"}]
    (meta / "tasks.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert dataset_tasks(str(tmp_path / "ds")) == ["open the drawer", "close it"]


def test_dataset_tasks_missing(tmp_path):
    assert dataset_tasks(str(tmp_path / "none")) == []
