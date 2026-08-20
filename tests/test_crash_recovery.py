"""Recovering a dataset from a recorder crash mid-episode.

The failure being guarded against is specific and nasty: a half-written data parquet has no
footer, and LeRobot globs `data/` rather than reading the file list from the metadata, so
one truncated file stops the whole dataset opening -- every good episode included.

Its quieter twin leaves no stray files at all: LeRobot rewrites `meta/info.json` every
episode but flushes the episode parquets in batches, so Ctrl-C leaves the counters ahead of
what is on disk and the next open detours to the Hub for the "missing" episodes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from workstation.lerobot_recorder.crash_recovery import (
    find_crash_leftovers,
    find_uncommitted_metadata,
    recover,
    truncate_to_committed,
)

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _dataset(tmp_path, episodes=3, per_ep=10, files=2):
    """A small committed dataset: `files` data parquets, all referenced by meta."""
    import pandas as pd

    root = tmp_path / "ds"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.cam" / "chunk-000").mkdir(parents=True)

    rows, meta, frame = [], [], 0
    for ep in range(episodes):
        fi = ep % files
        for i in range(per_ep):
            rows.append({"episode_index": ep, "frame_index": i, "index": frame + i,
                         "action": np.zeros(4, np.float32)})
        meta.append({
            "episode_index": ep, "length": per_ep,
            "data/chunk_index": 0, "data/file_index": fi,
            "dataset_from_index": frame, "dataset_to_index": frame + per_ep,
            "videos/observation.images.cam/chunk_index": 0,
            "videos/observation.images.cam/file_index": fi,
            "videos/observation.images.cam/from_timestamp": 0.0,
            "videos/observation.images.cam/to_timestamp": per_ep / 30,
        })
        frame += per_ep
    df = pd.DataFrame(rows)
    for fi in range(files):
        eps = [m["episode_index"] for m in meta if m["data/file_index"] == fi]
        df[df.episode_index.isin(eps)].to_parquet(root / "data" / "chunk-000" / f"file-{fi:03d}.parquet")
        (root / "videos" / "observation.images.cam" / "chunk-000" / f"file-{fi:03d}.mp4").write_bytes(b"\x00" * 64)
    pd.DataFrame(meta).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps(
        {"fps": 30, "total_episodes": episodes, "total_frames": frame,
         "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    return root


def _crash(root, *, truncated=True):
    """Reproduce what a crash mid-episode leaves: an orphan data parquet (truncated, so it
    has no footer), an orphan mp4, staged PNGs and a temp dir."""
    orphan = root / "data" / "chunk-000" / "file-019.parquet"
    orphan.write_bytes(b"PAR1" + b"\x00" * 500 if truncated else b"")
    (root / "videos" / "observation.images.cam" / "chunk-000" / "file-019.mp4").write_bytes(b"\x00" * 32)
    staged = root / "images" / "observation.images.cam" / "episode-000019"
    staged.mkdir(parents=True)
    (staged / "frame-000000.png").write_bytes(b"\x89PNG")
    tmp = root / "tmpABCDEF"
    tmp.mkdir()
    (tmp / "cam_19.mp4").write_bytes(b"\x00" * 16)
    return orphan


def test_finds_exactly_what_the_metadata_does_not_claim(tmp_path):
    root = _dataset(tmp_path)
    _crash(root)
    found = find_crash_leftovers(str(root))

    assert [os.path.basename(p) for p in found["corrupt_data"]] == ["file-019.parquet"]
    assert [os.path.basename(p) for p in found["unreferenced_video"]] == ["file-019.mp4"]
    assert found["staged_images"] and found["temp_dirs"]
    # the committed files must not appear anywhere in the report
    flat = [p for v in found.values() for p in v]
    for fi in (0, 1):
        assert not any(f"file-{fi:03d}" in p for p in flat)


def test_clean_dataset_reports_nothing(tmp_path):
    assert find_crash_leftovers(str(_dataset(tmp_path))) == {}


def test_recover_moves_and_never_deletes(tmp_path):
    root = _dataset(tmp_path)
    _crash(root)
    result = recover(str(root))

    assert result["moved"], "nothing was moved"
    q = Path(result["quarantine"])
    assert q.exists()
    # every moved item is still on disk, just out of the way
    for path in result["moved"]:
        assert os.path.exists(path)
    assert not (root / "data" / "chunk-000" / "file-019.parquet").exists()
    assert not (root / "images").exists()
    assert not list(root.glob("tmp*"))
    # committed data untouched
    assert (root / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (root / "data" / "chunk-000" / "file-001.parquet").exists()


def test_quarantine_is_outside_the_dataset(tmp_path):
    """Parking a footerless parquet anywhere under data/ would leave the dataset just as
    unopenable, since LeRobot globs that directory."""
    root = _dataset(tmp_path)
    _crash(root)
    result = recover(str(root))
    q = Path(result["quarantine"]).resolve()
    assert root.resolve() not in q.parents and q != root.resolve()
    assert not list(root.rglob("*.parquet.bak"))
    assert sorted(p.name for p in (root / "data" / "chunk-000").glob("*.parquet")) == [
        "file-000.parquet", "file-001.parquet"
    ]


def test_dry_run_writes_nothing(tmp_path):
    root = _dataset(tmp_path)
    _crash(root)
    result = recover(str(root), dry_run=True)
    assert result["moved"] == []
    assert (root / "data" / "chunk-000" / "file-019.parquet").exists()
    assert (root / "images").exists()


def test_recover_is_idempotent(tmp_path):
    root = _dataset(tmp_path)
    _crash(root)
    recover(str(root))
    again = recover(str(root))
    assert again["moved"] == []
    assert again["quarantine"] is None


def test_two_cameras_with_the_same_file_number_do_not_collide(tmp_path):
    """Both cameras have a file-019.mp4; flattening into one quarantine must keep both."""
    import pandas as pd

    root = _dataset(tmp_path)
    second = root / "videos" / "observation.images.cam2" / "chunk-000"
    second.mkdir(parents=True)
    (second / "file-019.mp4").write_bytes(b"\x00" * 8)
    _crash(root)

    result = recover(str(root))
    names = [os.path.basename(p) for p in result["moved"] if p.endswith(".mp4")]
    assert len(names) == len(set(names)) == 2


def test_a_referenced_but_unreadable_file_is_reported_not_moved(tmp_path, caplog):
    """That would be real data loss; quietly moving it would hide it."""
    root = _dataset(tmp_path)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"garbage")
    with caplog.at_level("ERROR"):
        found = find_crash_leftovers(str(root))
    assert not any("file-000" in p for v in found.values() for p in v)
    assert "REFERENCED" in caplog.text


def _uncommitted(root, *, episodes=2, frames=20):
    """What Ctrl-C mid-batch leaves: info.json counting episodes meta/episodes never got."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] += episodes
    info["total_frames"] += frames
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    info_path.write_text(json.dumps(info))
    (root / "outcomes.jsonl").write_text("".join(
        json.dumps({"episode": ep, "outcome": "success", "frames": 10}) + "\n"
        for ep in range(info["total_episodes"])
    ))
    return info


def test_counters_ahead_of_the_metadata_are_reported(tmp_path):
    root = _dataset(tmp_path, episodes=3, per_ep=10)
    _uncommitted(root)

    drift = find_uncommitted_metadata(str(root))
    assert drift["info_episodes"] == 5 and drift["committed_episodes"] == 3
    assert drift["lost_episodes"] == 2 and drift["lost_frames"] == 20
    # and it is invisible to the file-based check, which is the whole point
    assert find_crash_leftovers(str(root)) == {}


def test_counters_matching_the_metadata_report_nothing(tmp_path):
    assert find_uncommitted_metadata(str(_dataset(tmp_path))) is None


def test_truncate_rolls_info_and_the_sidecar_back(tmp_path):
    root = _dataset(tmp_path, episodes=3, per_ep=10)
    _uncommitted(root)
    q = tmp_path / "q"

    drift = truncate_to_committed(str(root), quarantine=str(q))

    info = json.loads((root / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (3, 30)
    assert info["splits"] == {"train": "0:3"}
    assert drift["lost_episodes"] == 2
    outcomes = [json.loads(line) for line in (root / "outcomes.jsonl").read_text().splitlines()]
    assert [o["episode"] for o in outcomes] == [0, 1, 2]
    # the pre-repair counters stay readable rather than being overwritten in place
    assert json.loads((q / "meta__info.json").read_text())["total_episodes"] == 5


def test_truncate_is_idempotent(tmp_path):
    root = _dataset(tmp_path, episodes=3, per_ep=10)
    _uncommitted(root)
    truncate_to_committed(str(root), quarantine=str(tmp_path / "q"))
    assert truncate_to_committed(str(root), quarantine=str(tmp_path / "q2")) is None


def test_recover_fixes_files_and_counters_together(tmp_path):
    root = _dataset(tmp_path, episodes=3, per_ep=10)
    _crash(root)
    _uncommitted(root)

    result = recover(str(root))

    assert result["moved"] and result["uncommitted"]["lost_episodes"] == 2
    assert json.loads((root / "meta" / "info.json").read_text())["total_episodes"] == 3
    assert find_uncommitted_metadata(str(root)) is None
