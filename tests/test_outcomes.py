"""Episode verdicts inside the LeRobot schema (``next.success`` / ``next.done``).

What is locked here: the terminal-frame encoding, the episode-level reading of it from the
metadata alone, the migration of a pre-schema dataset (columns, features, per-episode stats,
aggregate stats -- and that nothing else changes), and the refusal to guess on a dataset that
was never migrated. The writer's own use of it is covered in test_recorder / test_pipeline.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from workstation.lerobot_recorder import outcomes as oc
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter, dataset_dir


# ------------------------------------------------------------------ the encoding
def test_terminal_flags_mark_only_the_last_frame():
    f = oc.terminal_flags(4, "success")
    assert f[oc.SUCCESS_KEY].tolist() == [[False], [False], [False], [True]]
    assert f[oc.DONE_KEY].tolist() == [[False], [False], [False], [True]]
    f = oc.terminal_flags(3, "fail")
    assert f[oc.SUCCESS_KEY].tolist() == [[False]] * 3
    assert f[oc.DONE_KEY].tolist() == [[False], [False], [True]]


def test_no_verdict_is_all_false_not_a_failure():
    for outcome in (None, "unknown", "discard", ""):
        f = oc.terminal_flags(3, outcome)
        assert not f[oc.SUCCESS_KEY].any() and not f[oc.DONE_KEY].any(), outcome
    assert oc.terminal_flags(0, "success")[oc.DONE_KEY].shape == (0, 1)


def test_normalize_accepts_every_spelling_the_sidecar_ever_used():
    assert oc.normalize("keep") == oc.SUCCESS  # the pre-rename DAgger label
    assert oc.normalize(" Success ") == oc.SUCCESS
    assert oc.normalize("failure") == oc.FAIL
    assert oc.normalize("discard") == oc.UNKNOWN
    assert oc.normalize(None) == oc.UNKNOWN


def test_features_are_declared_the_way_lerobot_wants_them():
    for spec in oc.OUTCOME_FEATURES.values():
        assert spec == {"dtype": "bool", "shape": (1,), "names": None}
    assert oc.frame_value(True).dtype == bool and oc.frame_value(True).shape == (1,)


# ------------------------------------------------------------------ a real dataset
def _frame():
    return {
        "images": {"agentview": np.zeros((32, 32, 3), np.uint8)},
        "observation.state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
    }


def _record(tmp_path, episodes, name="ds"):
    """``[(outcome, n_frames), ...]`` -> path of a real dataset written by the recorder."""
    cfg = RecorderConfig(repo_id=f"me/{name}", root=str(tmp_path), mock=False, encoding_backend="pyav")
    w = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (32, 32, 3)})
    w.open(_frame())
    for outcome, n in episodes:
        w.submit([_frame() for _ in range(n)], outcome, "t")
    w.finalize()
    return dataset_dir(str(tmp_path), cfg.repo_id)


def _strip_to_pre_schema(ds_dir):
    """Turn a freshly recorded dataset into what the recorder wrote before this module
    existed: no columns, no features, no stats for them -- plus the old sidecar."""
    import glob
    import os

    import pandas as pd

    from workstation.lerobot_recorder.crash_recovery import reaggregate_stats

    verdicts = oc.episode_outcomes(ds_dir)
    for f in glob.glob(os.path.join(ds_dir, "data", "**", "*.parquet"), recursive=True):
        df = pd.read_parquet(f).drop(columns=[oc.SUCCESS_KEY, oc.DONE_KEY])
        df.to_parquet(f, index=False)
    for f in glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True):
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in df.columns if c.startswith("stats/next.")])
        df.to_parquet(f, index=False)
    info_path = os.path.join(ds_dir, "meta", "info.json")
    info = json.load(open(info_path))
    for k in oc.OUTCOME_FEATURES:
        info["features"].pop(k)
    json.dump(info, open(info_path, "w"), indent=4)
    stats_path = os.path.join(ds_dir, "meta", "stats.json")
    stats = json.load(open(stats_path))
    for k in oc.OUTCOME_FEATURES:
        stats.pop(k, None)
    json.dump(stats, open(stats_path, "w"), indent=4)
    reaggregate_stats(ds_dir)
    with open(os.path.join(ds_dir, oc.LEGACY_SIDECAR), "w") as fh:
        for ep, state in verdicts.items():
            row = {"episode": ep, "outcome": {"unknown": "discard"}.get(state, state), "task": "t"}
            fh.write(json.dumps(row) + "\n")
    return verdicts


def test_episode_outcomes_reads_the_metadata_alone(tmp_path):
    pytest.importorskip("lerobot")
    ds_dir = _record(tmp_path, [("success", 3), ("fail", 4), (None, 2)])
    assert oc.episode_outcomes(ds_dir) == {0: "success", 1: "fail", 2: "unknown"}
    assert oc.success_episodes(ds_dir) == [0]
    assert oc.outcome_totals(ds_dir) == {"success": 1, "fail": 1}
    assert oc.episode_lengths(ds_dir) == {0: 3, 1: 4, 2: 2}
    table = oc.episode_table(ds_dir)
    assert table[1] == {"outcome": "fail", "task": "t", "length": 4}
    assert oc.has_outcome_features(ds_dir) and not oc.predates_outcome_schema(ds_dir)


def test_migrate_reproduces_what_the_recorder_writes(tmp_path):
    """Strip a recorded dataset back to the pre-schema layout, migrate it from the sidecar,
    and every file the recorder produced must come back identical -- columns, feature
    declaration, per-episode stats, aggregate stats -- with nothing else disturbed."""
    pytest.importorskip("lerobot")
    import glob
    import os

    import pandas as pd

    ds_dir = _record(tmp_path, [("success", 3), ("fail", 4), (None, 2)])
    reference = tmp_path / "reference"
    shutil.copytree(ds_dir, reference)
    verdicts = _strip_to_pre_schema(ds_dir)
    assert oc.predates_outcome_schema(ds_dir)
    with pytest.raises(oc.OutcomeColumnsMissing):
        oc.episode_outcomes(ds_dir)
    assert oc.episode_outcomes(ds_dir, strict=False) == {}

    tally = oc.migrate(ds_dir)
    assert tally == {"episodes": 3, "success": 1, "fail": 1, "unknown": 1}
    assert oc.episode_outcomes(ds_dir) == verdicts
    assert oc.verify_against_sidecar(ds_dir) == []

    # data columns: same values, same dtype, same order
    for f in sorted(glob.glob(os.path.join(ds_dir, "data", "**", "*.parquet"), recursive=True)):
        got = pd.read_parquet(f)
        want = pd.read_parquet(f.replace(str(ds_dir), str(reference)))
        assert list(got.columns) == list(want.columns)
        for k in oc.OUTCOME_FEATURES:
            assert got[k].dtype == want[k].dtype == bool
            assert got[k].tolist() == want[k].tolist()
    # per-episode stats: identical shapes and values
    for f in sorted(glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True)):
        got = pd.read_parquet(f)
        want = pd.read_parquet(f.replace(str(ds_dir), str(reference)))
        for col in [c for c in want.columns if c.startswith("stats/next.")]:
            assert [np.asarray(v).tolist() for v in got[col]] == [np.asarray(v).tolist() for v in want[col]], col
    # feature declaration and aggregate stats
    got_info = json.load(open(os.path.join(ds_dir, "meta", "info.json")))
    want_info = json.load(open(os.path.join(reference, "meta", "info.json")))
    assert got_info["features"] == want_info["features"]
    got_stats = json.load(open(os.path.join(ds_dir, "meta", "stats.json")))
    want_stats = json.load(open(os.path.join(reference, "meta", "stats.json")))
    assert set(got_stats) == set(want_stats)
    for k in want_stats:
        for st in want_stats[k]:
            np.testing.assert_allclose(
                np.asarray(got_stats[k][st], dtype=float),
                np.asarray(want_stats[k][st], dtype=float),
                err_msg=f"{k}/{st}",
            )
    # ...and the bool stats came back bool, not 0.0 / 1.0
    assert got_stats[oc.SUCCESS_KEY]["max"] == [True] and got_stats[oc.SUCCESS_KEY]["min"] == [False]
    # videos untouched
    assert sorted(p.name for p in (tmp_path / "ds" / "videos").rglob("*.mp4")) == sorted(
        p.name for p in (reference / "videos").rglob("*.mp4")
    )


def test_migrate_is_idempotent_and_loads_in_lerobot(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_dir = _record(tmp_path, [("success", 3), ("fail", 2)])
    _strip_to_pre_schema(ds_dir)
    first = oc.migrate(ds_dir)
    again = oc.migrate(ds_dir)  # already migrated: a no-op that reports the same tally
    assert first == again
    ds = LeRobotDataset("me/ds", root=ds_dir)
    assert ds.features[oc.SUCCESS_KEY]["dtype"] == "bool"
    last0 = ds.meta.episodes[0]["dataset_to_index"] - 1
    assert bool(ds[last0][oc.SUCCESS_KEY]) and bool(ds[last0][oc.DONE_KEY])
    assert not bool(ds[0][oc.DONE_KEY])
    last1 = ds.meta.episodes[1]["dataset_to_index"] - 1
    assert not bool(ds[last1][oc.SUCCESS_KEY]) and bool(ds[last1][oc.DONE_KEY])


def test_migrate_with_explicit_verdicts_ignores_the_sidecar(tmp_path):
    pytest.importorskip("lerobot")
    ds_dir = _record(tmp_path, [("success", 2), ("success", 2)])
    _strip_to_pre_schema(ds_dir)
    oc.migrate(ds_dir, outcomes={0: "fail"})  # episode 1 gets no verdict
    assert oc.episode_outcomes(ds_dir) == {0: "fail", 1: "unknown"}
    problems = oc.verify_against_sidecar(ds_dir)
    assert len(problems) == 2  # and verify says so, loudly


def test_relabel_round_trip(tmp_path):
    pytest.importorskip("lerobot")
    ds_dir = _record(tmp_path, [("success", 3), ("success", 3)])
    assert oc.relabel(ds_dir, 1, "fail") == "fail"
    assert oc.episode_outcomes(ds_dir) == {0: "success", 1: "fail"}
    assert oc.relabel(ds_dir, 1, "success") == "success"
    assert oc.episode_outcomes(ds_dir) == {0: "success", 1: "success"}
    with pytest.raises(KeyError):
        oc.relabel(ds_dir, 7, "fail")


def test_cli_show_and_verify(tmp_path, capsys):
    pytest.importorskip("lerobot")
    ds_dir = _record(tmp_path, [("success", 2), ("fail", 2)])
    assert oc.main(["show", ds_dir]) == 0
    out = capsys.readouterr().out
    assert "0  success" in out and "1  fail" in out
    assert oc.main(["relabel", ds_dir, "0", "unknown"]) == 0
    assert oc.episode_outcomes(ds_dir)[0] == "unknown"


def test_migrate_leaves_every_other_stats_entry_alone(tmp_path):
    """A dataset edited in place (mark-homing, set-task) can have a stats.json that disagrees
    with its per-episode stats -- and which side is current differs per feature. The migration
    owns two keys; it must not re-aggregate the rest (seen on yam_lego_taxi: a full
    re-aggregation rolled observation.control_mode back to its pre-edit value)."""
    pytest.importorskip("lerobot")
    import os

    ds_dir = _record(tmp_path, [("success", 3), ("fail", 2)])
    _strip_to_pre_schema(ds_dir)
    stats_path = os.path.join(ds_dir, "meta", "stats.json")
    stats = json.load(open(stats_path))
    stats["observation.state"]["mean"] = [123.0] * len(stats["observation.state"]["mean"])  # "edited in place"
    stats["some.tool.wrote.this"] = {"note": "not a feature at all"}
    json.dump(stats, open(stats_path, "w"))

    oc.migrate(ds_dir)

    after = json.load(open(stats_path))
    assert after["observation.state"]["mean"] == [123.0] * len(stats["observation.state"]["mean"])
    assert after["some.tool.wrote.this"] == {"note": "not a feature at all"}
    assert after[oc.SUCCESS_KEY]["max"] == [True] and abs(after[oc.SUCCESS_KEY]["mean"][0] - 1 / 5) < 1e-9
    assert after[oc.DONE_KEY]["mean"][0] == pytest.approx(2 / 5)
