"""Dataset editor: outcome-sidecar remapping (pure) + relabel (sidecar I/O).

The structural edits (delete_episodes / set_task) go through LeRobot's dataset_tools
and are covered by the end-to-end check against a real dataset; here we lock the
pure re-indexing logic and the lightweight relabel path, which need no lerobot.
"""

from __future__ import annotations

import json

from workstation.lerobot_recorder.dataset_editor import DatasetEditor, remap_outcome_entries


def _entries(*eps):
    return [{"episode": e, "outcome": "success", "task": "t", "frames": 10} for e in eps]


def test_remap_drops_deleted_and_reindexes_contiguously():
    entries = _entries(0, 1, 2, 3, 4)
    out = remap_outcome_entries(entries, deleted=[1, 3], total_episodes=5)
    # survivors were old 0,2,4 -> new 0,1,2 in order
    assert [e["episode"] for e in out] == [0, 1, 2]


def test_remap_preserves_payload_fields():
    entries = [{"episode": 2, "outcome": "fail", "task": "pick", "frames": 42, "source": "teleop"}]
    out = remap_outcome_entries(entries, deleted=[0], total_episodes=3)
    # old 2 -> new 1 (0 deleted, 1 and 2 survive)
    assert out == [{"episode": 1, "outcome": "fail", "task": "pick", "frames": 42, "source": "teleop"}]


def test_remap_ignores_rows_for_deleted_or_unknown_episodes():
    entries = _entries(0, 1, 2) + [{"episode": 99}, {"nope": 1}]
    out = remap_outcome_entries(entries, deleted=[1], total_episodes=3)
    assert [e["episode"] for e in out] == [0, 1]  # old 0->0, old 2->1


def test_remap_handles_missing_sidecar_rows():
    # only episode 2 has a row; 0 and 1 don't. deleting 0 -> old 2 becomes new 1.
    out = remap_outcome_entries(_entries(2), deleted=[0], total_episodes=3)
    assert out == [{"episode": 1, "outcome": "success", "task": "t", "frames": 10}]


def test_relabel_updates_existing_sidecar_row(tmp_path):
    ds_dir = tmp_path / "close_bottle_cap"
    ds_dir.mkdir()
    sidecar = ds_dir / "outcomes.jsonl"
    sidecar.write_text(
        json.dumps({"episode": 0, "outcome": "success", "task": "t"}) + "\n"
        + json.dumps({"episode": 1, "outcome": "success", "task": "t"}) + "\n"
    )
    ed = DatasetEditor("me/close_bottle_cap", str(tmp_path))
    ed.relabel(1, "fail")

    rows = [json.loads(x) for x in sidecar.read_text().splitlines() if x.strip()]
    assert rows[0]["outcome"] == "success"
    assert rows[1]["outcome"] == "fail"
    assert len(rows) == 2  # no duplicate row added


def test_relabel_appends_when_row_absent(tmp_path):
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    ed = DatasetEditor("me/ds", str(tmp_path))
    ed.relabel(0, "discard")
    rows = [json.loads(x) for x in (ds_dir / "outcomes.jsonl").read_text().splitlines() if x.strip()]
    assert rows == [{"episode": 0, "outcome": "discard"}]


def test_outcomes_by_episode_reads_back(tmp_path):
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    ed = DatasetEditor("me/ds", str(tmp_path))
    ed.relabel(0, "success")
    ed.relabel(1, "fail")
    assert ed.outcomes_by_episode() == {0: "success", 1: "fail"}
