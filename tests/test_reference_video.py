"""Past-demonstration discovery for the recorder/deployment overlay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import workstation.lerobot_recorder.reference_video as reference_module
from workstation.lerobot_recorder.reference_video import (
    ReferenceEpisode,
    ReferenceVideoPlayer,
    discover_reference_episodes,
)

CAMERAS = ("wrist_left", "agentview", "wrist_right")


def _video(tmp_path, camera: str, chunk: int, file_index: int):
    path = tmp_path / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _episode_row(episode: int, file_index: int, start: float, end: float):
    row = {"episode_index": episode, "tasks": ["pick the cube"], "length": round((end - start) * 30)}
    for camera in CAMERAS:
        prefix = f"videos/observation.images.{camera}"
        row[f"{prefix}/chunk_index"] = 0
        row[f"{prefix}/file_index"] = file_index
        row[f"{prefix}/from_timestamp"] = start
        row[f"{prefix}/to_timestamp"] = end
    return row


def test_discovers_demonstration_slices_not_mp4_container_files(tmp_path, monkeypatch):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 30}))
    shared_container = {camera: _video(tmp_path, camera, 0, 1) for camera in CAMERAS}
    rows = [_episode_row(7, 1, 2.0, 5.0), _episode_row(8, 1, 5.0, 9.0)]
    monkeypatch.setattr(reference_module, "_read_episode_rows", lambda _root, _columns: rows)

    episodes = discover_reference_episodes(tmp_path, CAMERAS)

    assert [episode.episode for episode in episodes] == [7, 8]
    assert episodes[0].paths == shared_container
    assert episodes[1].paths == shared_container
    assert episodes[0].from_timestamps == dict.fromkeys(CAMERAS, 2.0)
    assert episodes[0].to_timestamps == dict.fromkeys(CAMERAS, 5.0)
    assert episodes[0].length == 90
    assert episodes[0].label == "demonstration 0007 · pick the cube"


def test_missing_or_non_video_dataset_has_no_references(tmp_path):
    assert discover_reference_episodes(tmp_path, CAMERAS) == []
    for camera in CAMERAS:
        path = _video(tmp_path, camera, 0, 0)
        path.rename(path.with_suffix(".part"))
    assert discover_reference_episodes(tmp_path, CAMERAS) == []


def test_player_loads_first_frame_but_defaults_to_paused(monkeypatch):
    class FakeStdout:
        def read(self, size):
            return bytes([17]) * size

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    commands = []
    monkeypatch.setattr(reference_module, "_probe_size", lambda _path: (2, 1))
    monkeypatch.setattr(
        reference_module.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or FakeProcess(),
    )
    episode = ReferenceEpisode(
        episode=4,
        paths={key: Path(f"{key}.mp4") for key in CAMERAS},
        fps=30,
        from_timestamps=dict.fromkeys(CAMERAS, 12.5),
        to_timestamps=dict.fromkeys(CAMERAS, 15.5),
    )
    player = ReferenceVideoPlayer(CAMERAS)

    try:
        player.play(episode)
        deadline = time.monotonic() + 1
        while not player.get_frames() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert player.paused is True
        assert set(player.get_frames()) == set(CAMERAS)
        assert all((frame == 17).all() for frame in player.get_frames().values())
        assert commands[0].count("12.500000000") == 3
        assert "-stream_loop" not in commands[0]
        assert "trim=duration=3.000000000" in commands[0][commands[0].index("-filter_complex") + 1]
    finally:
        player.stop()
