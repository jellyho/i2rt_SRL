"""Discover and play synchronized camera videos from prior dataset episodes.

The recorder may pack many demonstrations into each camera MP4.  This module reads
LeRobot's episode metadata to resolve each demonstration's file and timestamp slice
without opening a second ``LeRobotDataset`` while the recorder is appending to it.
A single ffmpeg process decodes the selected slices into one synchronized strip;
only the latest preview frame is kept.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceEpisode:
    """One demonstration and its slice within synchronized camera containers."""

    episode: int
    paths: Dict[str, Path]
    fps: float
    from_timestamps: Dict[str, float] = field(default_factory=dict)
    to_timestamps: Dict[str, float] = field(default_factory=dict)
    tasks: Tuple[str, ...] = ()
    length: int = 0

    @property
    def label(self) -> str:
        task = f" · {self.tasks[0]}" if self.tasks else ""
        return f"demonstration {self.episode:04d}{task}"


def _read_episode_rows(root: Path, columns: list[str]) -> list[Dict[str, Any]]:
    """Read only episode-routing columns, leaving the large stats columns untouched."""

    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read LeRobot demonstration metadata") from exc

    rows: list[Dict[str, Any]] = []
    for path in files:
        try:
            rows.extend(pq.read_table(path, columns=columns).to_pylist())
        except Exception as exc:
            raise RuntimeError(f"could not read demonstration metadata {path}: {exc}") from exc
    return rows


def discover_reference_episodes(
    dataset_root: str | Path,
    camera_keys: Iterable[str] = ("wrist_left", "agentview", "wrist_right"),
) -> list[ReferenceEpisode]:
    """Return demonstrations that have a completed MP4 slice for every camera.

    LeRobot v3 may pack many demonstrations into one ``file-NNN.mp4``.  The
    authoritative ``meta/episodes`` rows provide each demonstration's camera
    file plus ``from_timestamp`` / ``to_timestamp`` slice; MP4 filenames are
    container identifiers, never demonstration identifiers.
    """

    root = Path(dataset_root).expanduser()
    keys = tuple(camera_keys)
    if not keys:
        return []

    fps = 30.0
    video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        fps = float(info.get("fps", fps))
        video_path = str(info.get("video_path", video_path))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    columns = ["episode_index", "tasks", "length"]
    for key in keys:
        prefix = f"videos/observation.images.{key}"
        columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )

    episodes: list[ReferenceEpisode] = []
    for row in _read_episode_rows(root, columns):
        paths: Dict[str, Path] = {}
        starts: Dict[str, float] = {}
        ends: Dict[str, float] = {}
        try:
            for key in keys:
                prefix = f"videos/observation.images.{key}"
                path = root / video_path.format(
                    video_key=f"observation.images.{key}",
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths[key] = path
                starts[key] = float(row[f"{prefix}/from_timestamp"])
                ends[key] = float(row[f"{prefix}/to_timestamp"])
            episodes.append(
                ReferenceEpisode(
                    episode=int(row["episode_index"]),
                    paths=paths,
                    fps=fps,
                    from_timestamps=starts,
                    to_timestamps=ends,
                    tasks=tuple(str(task) for task in (row.get("tasks") or [])),
                    length=int(row.get("length") or 0),
                )
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError):
            # Metadata can appear before an async video encode is complete.
            continue
    return sorted(episodes, key=lambda episode: episode.episode)


def _probe_size(path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        stream = json.loads(result.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for past-demonstration overlays") from exc
    except (subprocess.SubprocessError, OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"could not inspect reference video {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"reference video has an invalid size: {path}")
    return width, height


class ReferenceVideoPlayer:
    """Loop one episode's synchronized views in a low-overhead background thread."""

    def __init__(self, camera_keys: Iterable[str], preview_fps: float = 15.0) -> None:
        self.camera_keys = tuple(camera_keys)
        self.preview_fps = max(float(preview_fps), 1.0)
        self._lock = threading.Lock()
        self._frames: Dict[str, np.ndarray] = {}
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._episode: Optional[ReferenceEpisode] = None
        self._error = ""
        self._finished = False

    @property
    def episode(self) -> Optional[ReferenceEpisode]:
        return self._episode

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Return a shallow snapshot; stored arrays are replaced, never mutated."""
        with self._lock:
            return dict(self._frames)

    def play(self, episode: ReferenceEpisode, *, start_paused: bool = True) -> None:
        """Load an episode, showing its first frame before optional playback.

        References default to paused because scene-reset comparisons should be
        stable.  The decoder still reads exactly one synchronized frame so the
        overlay appears immediately; playback begins only after ``set_paused(False)``.
        """
        self.stop()
        if not self.camera_keys:
            raise ValueError("reference overlay has no configured camera views")
        missing = [key for key in self.camera_keys if key not in episode.paths]
        if missing:
            raise ValueError(f"reference episode is missing cameras: {', '.join(missing)}")

        width, height = _probe_size(episode.paths[self.camera_keys[0]])
        cmd = ["ffmpeg", "-v", "error"]
        for key in self.camera_keys:
            start = max(float(episode.from_timestamps.get(key, 0.0)), 0.0)
            cmd.extend(["-ss", f"{start:.9f}", "-i", str(episode.paths[key])])

        filters = []
        for index, key in enumerate(self.camera_keys):
            start = float(episode.from_timestamps.get(key, 0.0))
            end = episode.to_timestamps.get(key)
            duration = float(end) - start if end is not None else 0.0
            trim = f"trim=duration={duration:.9f}," if duration > 0 else ""
            filters.append(
                f"[{index}:v]{trim}setpts=PTS-STARTPTS,scale={width}:{height}:flags=fast_bilinear[v{index}]"
            )
        inputs = "".join(f"[v{index}]" for index in range(len(self.camera_keys)))
        filters.append(f"{inputs}hstack=inputs={len(self.camera_keys)},fps={self.preview_fps}[out]")
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for past-demonstration overlays") from exc

        self._process = process
        self._episode = episode
        self._stop.clear()
        if start_paused:
            self._paused.set()
        else:
            self._paused.clear()
        with self._lock:
            self._frames = {}
            self._error = ""
            self._finished = False
        self._thread = threading.Thread(
            target=self._read_loop,
            args=(process, width, height, start_paused),
            name="reference-video",
            daemon=True,
        )
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            episode = self._episode
            if self.finished and episode is not None:
                self.play(episode, start_paused=False)
                return
            self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._paused.clear()
        self._episode = None
        with self._lock:
            self._frames = {}
            self._finished = False

    def _read_loop(self, process: subprocess.Popen, width: int, height: int, read_first_while_paused: bool) -> None:
        frame_bytes = width * len(self.camera_keys) * height * 3
        period = 1.0 / self.preview_fps
        next_frame = time.monotonic()
        try:
            while not self._stop.is_set():
                if self._paused.is_set() and not read_first_while_paused:
                    self._stop.wait(0.05)
                    next_frame = time.monotonic()
                    continue
                raw = self._read_exact(process, frame_bytes)
                if len(raw) != frame_bytes:
                    if not self._stop.is_set():
                        returncode = process.wait(timeout=1.0)
                        if returncode:
                            raise RuntimeError(f"reference video decoder exited with status {returncode}")
                        with self._lock:
                            self._finished = True
                        self._paused.set()
                    return
                strip = np.frombuffer(raw, dtype=np.uint8).reshape(height, width * len(self.camera_keys), 3)
                frames = {
                    key: np.ascontiguousarray(strip[:, index * width : (index + 1) * width])
                    for index, key in enumerate(self.camera_keys)
                }
                with self._lock:
                    self._frames = frames
                read_first_while_paused = False
                next_frame += period
                self._stop.wait(max(0.0, next_frame - time.monotonic()))
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            logger.warning("past-demonstration overlay stopped: %s", exc)

    def _read_exact(self, process: subprocess.Popen, size: int) -> bytes:
        stdout = process.stdout
        if stdout is None:
            return b""
        chunks = bytearray()
        while len(chunks) < size and not self._stop.is_set():
            chunk = stdout.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def __enter__(self) -> "ReferenceVideoPlayer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
