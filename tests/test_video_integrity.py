"""Frame-accurate video/metadata integrity checks + repair.

These build a *real* multi-episode mp4 with ffmpeg and then damage it the way the
stream-copy concat does in practice (drop a frame at an episode join, or truncate the
tail), because the whole point of ``video_integrity`` is that the damage is invisible to
metadata — a mocked video would test nothing.

Skipped when ffmpeg/ffprobe or pandas are unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from workstation.lerobot_recorder.video_integrity import (
    diagnose_shortfall,
    keyframe_frames,
    repair_short_videos,
    video_file_shortfalls,
    video_frame_count,
)

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg/ffprobe"
)

FPS = 10


def _clip(path, n_frames, seed):
    """Encode an ``n_frames`` clip whose first frame is an IDR (like a per-episode temp clip).

    ``-g 1000`` keeps that IDR the *only* keyframe, so the keyframe list of the concatenated
    file is exactly the set of episode boundaries — which is what the diagnosis relies on.
    """
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=s=64x48:r={FPS}:d={n_frames / FPS + 1}:decimals=0",
         "-frames:v", str(n_frames), "-c:v", "libx264", "-g", "1000", "-pix_fmt", "yuv420p",
         "-video_track_timescale", "1200", str(path)],
        check=True, capture_output=True,
    )


def _concat(parts, dst):
    listing = dst.parent / f"{dst.stem}.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{p}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(dst)], check=True, capture_output=True,
    )


def _make_dataset(tmp_path, episode_lengths, *, encoded_lengths=None):
    """A one-camera dataset whose metadata claims ``episode_lengths``.

    ``encoded_lengths`` (defaults to ``episode_lengths``) is what actually lands in the mp4 —
    making one entry smaller simulates a frame dropped at that episode's join, which the
    metadata knows nothing about.
    """
    import pandas as pd

    encoded_lengths = encoded_lengths or episode_lengths
    ds = tmp_path / "ds"
    vid_dir = ds / "videos" / "observation.images.cam" / "chunk-000"
    vid_dir.mkdir(parents=True)
    (ds / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    parts = []
    for i, n in enumerate(encoded_lengths):
        p = vid_dir / f"part{i}.mp4"
        _clip(p, n, i)
        parts.append(str(p))
    _concat(parts, vid_dir / "file-000.mp4")
    for p in parts:
        (vid_dir / p.split("/")[-1]).unlink()

    rows, frame = [], 0
    for ep, n in enumerate(episode_lengths):
        rows.append({
            "episode_index": ep, "length": n,
            "videos/observation.images.cam/chunk_index": 0,
            "videos/observation.images.cam/file_index": 0,
            "videos/observation.images.cam/from_timestamp": frame / FPS,
            "videos/observation.images.cam/to_timestamp": (frame + n) / FPS,
        })
        frame += n
    pd.DataFrame(rows).to_parquet(ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    (ds / "meta" / "info.json").write_text(json.dumps({
        "fps": FPS, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }))
    return ds


def _windows(ds):
    """{episode_index: (from_frame, to_frame)} straight from meta/episodes."""
    import pandas as pd

    df = pd.read_parquet(ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return {
        int(r["episode_index"]): (
            round(float(r["videos/observation.images.cam/from_timestamp"]) * FPS),
            round(float(r["videos/observation.images.cam/to_timestamp"]) * FPS),
        )
        for _, r in df.iterrows()
    }


# --------------------------------------------------------------------------- detection
def test_intact_dataset_reports_no_shortfall(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20])
    assert video_file_shortfalls(str(ds)) == []


def test_frame_count_reads_the_real_length(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20])
    path = str(ds / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4")
    assert video_frame_count(path) == 60
    assert video_frame_count(path, exact=True) == 60


def test_interior_drop_is_invisible_to_metadata_but_caught_by_frame_count(tmp_path):
    """The metadata stays perfectly self-consistent — only the real frame count gives it away."""
    pytest.importorskip("pandas")
    from workstation.lerobot_recorder.dataset_editor import video_length_mismatches

    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    assert video_length_mismatches(str(ds)) == []  # metadata-only check sees nothing

    short = video_file_shortfalls(str(ds))
    assert len(short) == 1
    assert short[0]["claimed"] == 60 and short[0]["actual"] == 59 and short[0]["missing"] == 1
    assert short[0]["overrun_episodes"] == [2]  # only the last episode runs off the end


def test_diagnose_attributes_an_interior_drop_to_the_right_join(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    diag = diagnose_shortfall(str(ds), video_file_shortfalls(str(ds))[0])
    # ep1 lost its last frame, so ep2 sits one frame earlier than metadata says
    assert diag["offsets"] == {0: 0, 1: 0, 2: 1}
    assert diag["eof_missing"] == 0


def test_diagnose_attributes_a_tail_truncation_to_the_end(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 20, 18])
    diag = diagnose_shortfall(str(ds), video_file_shortfalls(str(ds))[0])
    assert not any(diag["offsets"].values())  # no join moved
    assert diag["eof_missing"] == 2


# --------------------------------------------------------------------------- repair
def test_repair_realigns_episodes_after_an_interior_drop(tmp_path):
    """Interior loss: metadata only. The video is not rewritten, and the episodes that were
    silently misaligned are moved back onto their own frames."""
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    path = str(ds / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4")
    before = video_frame_count(path)

    out = repair_short_videos(str(ds))
    assert [r["status"] for r in out] == ["repaired"]
    assert out[0]["shifted_episodes"] == [2] and out[0]["appended_frames"] == 0
    assert video_frame_count(path) == before  # video untouched

    win = _windows(ds)
    assert win[0] == (0, 20)
    assert win[1] == (20, 40)  # keeps full length: its last frame is ep2's first (the borrow)
    assert win[2] == (39, 59)  # realigned onto the frames that are really ep2's
    assert win[2][1] == before  # and no longer runs past the end
    assert video_file_shortfalls(str(ds)) == []


def test_repair_appends_frames_for_a_truncated_tail(tmp_path):
    """End-of-file loss: no later episode to realign against, so pad the video instead."""
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 20, 18])
    path = str(ds / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4")

    out = repair_short_videos(str(ds))
    assert [r["status"] for r in out] == ["repaired"]
    assert out[0]["shifted_episodes"] == [] and out[0]["appended_frames"] == 2
    assert video_frame_count(path, exact=True) == 60  # decodes to the claimed length
    assert _windows(ds) == {0: (0, 20), 1: (20, 40), 2: (40, 60)}  # metadata untouched
    assert video_file_shortfalls(str(ds)) == []


def test_repaired_episode_starts_land_on_real_segment_boundaries(tmp_path):
    """The realigned windows must point at actual episode starts, which are always keyframes."""
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    path = str(ds / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4")
    repair_short_videos(str(ds))
    keys = set(keyframe_frames(path, FPS))
    for ep, (start, _end) in _windows(ds).items():
        assert start in keys, f"episode {ep} does not start on a segment boundary"


def test_dry_run_writes_nothing(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    before_windows = _windows(ds)
    path = str(ds / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4")
    before_frames = video_frame_count(path)

    out = repair_short_videos(str(ds), dry_run=True)
    assert [r["status"] for r in out] == ["would-repair"]
    assert _windows(ds) == before_windows
    assert video_frame_count(path) == before_frames
    assert video_file_shortfalls(str(ds))  # still broken


def test_repair_is_idempotent(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20], encoded_lengths=[20, 19, 20])
    repair_short_videos(str(ds))
    after = _windows(ds)
    assert repair_short_videos(str(ds)) == []
    assert _windows(ds) == after


def test_repair_handles_multiple_drops_in_one_file(tmp_path):
    pytest.importorskip("pandas")
    ds = _make_dataset(tmp_path, [20, 20, 20, 20], encoded_lengths=[20, 19, 20, 19])
    diag = diagnose_shortfall(str(ds), video_file_shortfalls(str(ds))[0])
    assert diag["offsets"] == {0: 0, 1: 0, 2: 1, 3: 1}
    assert diag["eof_missing"] == 1  # ep3's own lost frame is off the end

    out = repair_short_videos(str(ds))
    assert out[0]["shifted_episodes"] == [2, 3] and out[0]["appended_frames"] == 1
    assert video_file_shortfalls(str(ds)) == []


def test_missing_metadata_is_not_an_error(tmp_path):
    assert video_file_shortfalls(str(tmp_path / "nope")) == []
    assert repair_short_videos(str(tmp_path / "nope")) == []
