"""Frame-accurate integrity checks (and repair) for a recorded LeRobot **v3.0** dataset.

LeRobot v3 packs many episodes into one ``.mp4`` per camera and locates each episode by a
``[from_timestamp, to_timestamp)`` window in ``meta/episodes``. Those windows come from the
*duration* of a per-episode temporary clip, which is then stream-copy **concatenated** onto
the growing camera file. Two different things can go wrong:

**A. the encoder returns a short episode clip.** The temp clip holds ``length - 1`` frames,
so that episode's own window is one frame short. LeRobot records this honestly, so the next
episode's ``from_timestamp`` is still right and everything downstream stays aligned. A pure
metadata symptom — caught by
:func:`~workstation.lerobot_recorder.dataset_editor.video_length_mismatches`.

**B. the stream-copy concat drops a frame at the join.** Nothing in the metadata learns
about it, so no metadata-only check can see it: every window still spans exactly ``length``
frames. But the file is now one frame shorter than the metadata believes, so *every episode
after that join is shifted by one frame*, and the last episode's window runs off the end.
That is what breaks a training run at load time::

    IndexError: Invalid frame index=6498 for streamIndex=0; must be less than 6498

This module handles **B**, which needs the real frame count of the file:

* :func:`video_frame_count` — exact count from the mp4 sample table (instant, no decode).
* :func:`video_file_shortfalls` — per video file, frames claimed vs. frames really present.
* :func:`diagnose_shortfall` — *where* the frames went missing. Every episode segment starts
  on a keyframe, so an episode boundary that lands early in the keyframe list marks a drop;
  this recovers the per-episode frame offset and how much was lost off the end.
* :func:`repair_short_videos` — fix it, preferring the repair that touches nothing:

  - frames lost at an **interior** join are repaired **in metadata**: each affected episode's
    window is moved back onto the frames that really are its own. No video is rewritten, and
    the episodes that were silently misaligned become correctly aligned again.
  - frames lost off the **end** of the file (a truncated final episode has no later episode
    to realign against) are repaired by **appending** a duplicate of the last frame. Appending
    is a plain stream copy that leaves every existing frame bit-identical.

Mid-file splicing is deliberately not attempted: inserting frames means re-concatenating
across interior joins, which is the very operation that drops frames here.

Everything shells out to ``ffprobe``/``ffmpeg``; when they are unavailable the checks degrade
to "unknown" rather than raising.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How far back a boundary is searched when attributing a loss to a join.
_MAX_SLIP = 16


# --------------------------------------------------------------------------- ffprobe/ffmpeg
def _run(cmd: List[str], timeout: float = 1800.0) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {out.stderr.strip()[:400]}")
    return out.stdout


def video_frame_count(path: str, *, exact: bool = False) -> Optional[int]:
    """Number of frames in ``path``, or None when it can't be determined.

    Reads ``nb_frames`` from the mp4 sample table — exact for mp4 and effectively free (no
    decoding). ``exact=True``, or a container without the field, falls back to
    ``-count_frames``, which decodes the whole file and is far slower.
    """
    if not exact:
        try:
            raw = _run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path,
            ]).strip()
            if raw and raw != "N/A":
                return int(raw)
        except Exception as e:
            logger.debug("nb_frames probe failed for %s: %s", path, e)
    try:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path,
        ]).strip()
        return int(raw) if raw and raw != "N/A" else None
    except Exception as e:
        logger.warning("could not count frames in %s: %s", path, e)
        return None


def keyframe_frames(path: str, fps: int) -> List[int]:
    """Frame indices of the file's keyframes, ascending.

    ``-skip_frame nokey`` means only keyframes are decoded. Each episode segment begins on an
    IDR frame, so the episode boundaries are always a subset of this list.
    """
    try:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", path,
        ])
    except Exception as e:
        logger.warning("could not list keyframes in %s: %s", path, e)
        return []
    out: List[int] = []
    for line in raw.splitlines():
        # csv=p=0 can emit a trailing empty field ("0.000000,"), so take the first column
        tok = line.split(",")[0].strip()
        if not tok:
            continue
        try:
            out.append(int(round(float(tok) * fps)))
        except ValueError:
            continue
    return sorted(set(out))


def _stream_params(path: str) -> Dict[str, str]:
    """pix_fmt / profile / time_base of the video stream (to encode a matching pad clip)."""
    try:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=pix_fmt,profile,time_base", "-of", "json", path,
        ])
        return json.loads(raw)["streams"][0]
    except Exception as e:
        logger.debug("could not read stream params of %s: %s", path, e)
        return {}


# --------------------------------------------------------------------------- metadata access
def _meta_files(ds_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))


def _episode_meta(ds_dir: str):
    """(fps, DataFrame of every episode row, video keys, video_path template) or None."""
    import pandas as pd

    info_path = os.path.join(ds_dir, "meta", "info.json")
    files = _meta_files(ds_dir)
    if not os.path.exists(info_path) or not files:
        return None
    info = json.load(open(info_path))
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
    vkeys = [c.split("/", 2)[1] for c in df.columns if c.endswith("/from_timestamp")]
    return int(info["fps"]), df, vkeys, info["video_path"]


def video_file_shortfalls(ds_dir: str) -> List[dict]:
    """Video files holding **fewer frames than ``meta/episodes`` claims** (failure mode B).

    This is the condition that raises ``IndexError`` at load time. Metadata-only checks
    cannot see it — every window is internally consistent, the *file* is simply short.

    One dict per offending file::

        {"video_key", "chunk", "file", "path", "claimed", "actual", "missing",
         "episodes": [...],           # every episode stored in this file
         "overrun_episodes": [...]}   # those whose window ends past the real EOF

    Empty when every file is intact (or when ffprobe/metadata are unavailable).
    """
    meta = _episode_meta(ds_dir)
    if meta is None:
        return []
    fps, df, vkeys, vpath = meta
    out: List[dict] = []
    for vk in vkeys:
        for (chunk, fidx), g in df.groupby([f"videos/{vk}/chunk_index", f"videos/{vk}/file_index"]):
            path = os.path.join(ds_dir, vpath.format(video_key=vk, chunk_index=int(chunk), file_index=int(fidx)))
            if not os.path.exists(path):
                continue
            claimed = int(round(float(g[f"videos/{vk}/to_timestamp"].max()) * fps))
            actual = video_frame_count(path)
            if actual is None or actual >= claimed:
                continue
            overrun = sorted(
                int(r["episode_index"]) for _, r in g.iterrows()
                if round(float(r[f"videos/{vk}/to_timestamp"]) * fps) > actual
            )
            out.append({
                "video_key": vk, "chunk": int(chunk), "file": int(fidx), "path": path,
                "claimed": claimed, "actual": actual, "missing": claimed - actual,
                "episodes": sorted(int(e) for e in g["episode_index"]),
                "overrun_episodes": overrun,
            })
    return out


def _file_episodes(df, vk: str, chunk: int, fidx: int) -> List[Tuple[int, int, int]]:
    """[(episode_index, expected_from_frame, length)] for one video file, in play order."""
    g = df[(df[f"videos/{vk}/chunk_index"] == chunk) & (df[f"videos/{vk}/file_index"] == fidx)]
    g = g.sort_values(f"videos/{vk}/from_timestamp")
    return [(int(r["episode_index"]), r[f"videos/{vk}/from_timestamp"], int(r["length"])) for _, r in g.iterrows()]


def diagnose_shortfall(ds_dir: str, shortfall: dict) -> dict:
    """Work out *where* a short file lost its frames.

    Every episode segment starts on a keyframe, so each episode's expected start —
    ``from_timestamp * fps`` — must appear in the file's keyframe list. When it turns up a few
    frames early instead, that many frames were dropped from the preceding episode's tail.

    Returns::

        {"offsets": {episode_index: frames_it_sits_earlier_than_metadata_says},
         "eof_missing": frames lost off the end of the file (no later episode to realign),
         "attributed": frames explained by an interior join}

    ``offsets`` + ``eof_missing`` always account for the whole shortfall, so a repair built
    from this restores the claimed length even when attribution is imperfect.
    """
    meta = _episode_meta(ds_dir)
    missing = int(shortfall["missing"])
    if meta is None or missing <= 0:
        return {"offsets": {}, "eof_missing": 0, "attributed": 0}
    fps, df, _vkeys, _vpath = meta
    vk = shortfall["video_key"]
    episodes = _file_episodes(df, vk, shortfall["chunk"], shortfall["file"])
    keys = set(keyframe_frames(shortfall["path"], fps))

    offsets: Dict[int, int] = {}
    offset = 0
    for ep, from_ts, _length in episodes:
        expected = int(round(float(from_ts) * fps))
        if expected > 0 and keys and offset < missing and (expected - offset) not in keys:
            for slip in range(1, min(_MAX_SLIP, missing - offset) + 1):
                if (expected - offset - slip) in keys:
                    offset += slip
                    break
        offsets[ep] = offset
    return {"offsets": offsets, "eof_missing": missing - offset, "attributed": offset}


# --------------------------------------------------------------------------- repair
def _apply_window_shifts(ds_dir: str, vk: str, new_windows: Dict[int, Tuple[float, float]]) -> int:
    """Write new ``from``/``to`` timestamps for ``vk`` into ``meta/episodes``. Returns rows changed."""
    import pandas as pd

    fcol, tcol = f"videos/{vk}/from_timestamp", f"videos/{vk}/to_timestamp"
    changed_total = 0
    for path in _meta_files(ds_dir):
        df = pd.read_parquet(path)
        if fcol not in df.columns:
            continue
        changed = False
        for i in range(len(df)):
            ep = int(df.iloc[i]["episode_index"])
            if ep not in new_windows:
                continue
            new_from, new_to = new_windows[ep]
            if float(df.iloc[i][fcol]) != new_from or float(df.iloc[i][tcol]) != new_to:
                df.iat[i, df.columns.get_loc(fcol)] = new_from
                df.iat[i, df.columns.get_loc(tcol)] = new_to
                changed = True
                changed_total += 1
        if changed:
            df.to_parquet(path, index=False)
    return changed_total


def _append_frames(src: str, n: int, fps: int) -> None:
    """Append ``n`` duplicates of ``src``'s final frame, in place.

    Append-only stream copy: existing frames are never re-encoded and stay bit-identical.
    The rebuilt file replaces the original only once its frame count verifies.
    """
    params = _stream_params(src)
    pix_fmt = str(params.get("pix_fmt") or "yuv420p")
    profile = str(params.get("profile") or "main").lower().replace(" ", "")
    if profile not in ("baseline", "main", "high"):
        profile = "main"
    timescale = str(params.get("time_base") or "1/15360").split("/")[-1]
    before = video_frame_count(src)

    with tempfile.TemporaryDirectory(dir=os.path.dirname(src)) as td:
        png = os.path.join(td, "last.png")
        pad = os.path.join(td, "pad.mp4")
        rebuilt = os.path.join(td, "rebuilt.mp4")
        listing = os.path.join(td, "parts.ffconcat")
        # -sseof -1 seeks to the last second; -update 1 keeps overwriting, so the file
        # left behind is the very last frame.
        _run(["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", src,
              "-update", "1", "-q:v", "1", png])
        _run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(fps), "-i", png,
              "-frames:v", str(n), "-c:v", "libx264", "-profile:v", profile,
              "-pix_fmt", pix_fmt, "-video_track_timescale", timescale, pad])
        with open(listing, "w") as fh:
            fh.write("ffconcat version 1.0\n")
            for p in (os.path.abspath(src), pad):
                fh.write(f"file '{p}'\n")
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
              "-c", "copy", "-movflags", "faststart", rebuilt])

        # Verify by decoding: the sample table can claim frames the decoder won't hand back.
        got = video_frame_count(rebuilt, exact=True)
        if got != (before or 0) + n:
            raise RuntimeError(f"padded file decodes to {got} frames, expected {(before or 0) + n}")

        backup = f"{src}.pre-pad"
        shutil.move(src, backup)
        shutil.move(rebuilt, src)
        os.remove(backup)


def repair_short_videos(ds_dir: str, *, dry_run: bool = False) -> List[dict]:
    """Repair every video file that holds fewer frames than ``meta/episodes`` claims.

    Per file, :func:`diagnose_shortfall` splits the loss into two kinds, each getting the
    repair that touches least:

    * **interior** losses — metadata only. Each episode from the join onward is moved back
      onto the frames that are actually its own, which both removes the overrun at the end of
      the file *and* un-does the silent misalignment those episodes were carrying. The video
      bytes are untouched. The one episode that truly lost its final frame keeps a full-length
      window, so its last frame is its successor's first (33 ms of the next episode) — the
      same trade the metadata-only repair already makes for failure mode A.
    * **end-of-file** losses — a duplicate of the last frame is appended, since a truncated
      final episode has no later episode to realign against.

    Returns one dict per file: the shortfall plus ``status``
    (``"repaired"`` / ``"would-repair"`` / ``"failed"``), ``shifted_episodes`` and
    ``appended_frames``. Nothing is written when ``dry_run``.
    """
    meta = _episode_meta(ds_dir)
    if meta is None:
        return []
    fps, df, _vkeys, _vpath = meta
    results: List[dict] = []

    for sf in video_file_shortfalls(ds_dir):
        diag = diagnose_shortfall(ds_dir, sf)
        offsets = diag["offsets"]
        episodes = _file_episodes(df, sf["video_key"], sf["chunk"], sf["file"])
        new_windows = {
            ep: ((round(float(from_ts) * fps) - offsets.get(ep, 0)) / fps,
                 (round(float(from_ts) * fps) - offsets.get(ep, 0) + length) / fps)
            for ep, from_ts, length in episodes
            if offsets.get(ep, 0)
        }
        entry = dict(
            sf,
            shifted_episodes=sorted(new_windows),
            appended_frames=diag["eof_missing"],
            status="would-repair" if dry_run else "repaired",
        )
        if dry_run:
            results.append(entry)
            continue
        try:
            if new_windows:
                _apply_window_shifts(ds_dir, sf["video_key"], new_windows)
            if diag["eof_missing"] > 0:
                _append_frames(sf["path"], int(diag["eof_missing"]), fps)
            logger.info(
                "repaired %s: realigned %d episode(s), appended %d frame(s)",
                os.path.basename(sf["path"]), len(new_windows), diag["eof_missing"],
            )
        except Exception as e:
            logger.error("could not repair %s: %s", sf["path"], e)
            entry["status"] = "failed"
            entry["error"] = str(e)
        results.append(entry)
    return results
