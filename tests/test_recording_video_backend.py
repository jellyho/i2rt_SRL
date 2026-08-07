from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
from workstation.lerobot_recorder.video_encoding import (
    EncodingBackendDecision,
    GpuMemory,
    encode_frames_torchcodec,
    select_encoding_backend,
)


def test_pyav_is_honored_without_gpu_probe():
    def unexpected_probe():
        raise AssertionError("explicit PyAV must not inspect the GPU")

    decision = select_encoding_backend(
        "pyav",
        memory_query=unexpected_probe,
        torchcodec_available=unexpected_probe,
    )
    assert decision.effective == "pyav"


def test_torchcodec_is_never_selected_even_when_the_gpu_is_free():
    """TorchCodec cannot run without staging every frame as a PNG.

    Its encoder reads frames from memory -- which is what made the staging look like waste --
    but lerobot 0.4.4 computes each episode's image stats by loading those same PNGs back off
    disk, with no streaming equivalent. Suppressing the writes without the stats knowing kills
    the episode at save time on a file nothing wrote. So it is gigabytes of PNGs per episode
    or no TorchCodec, and this repo chose no TorchCodec.
    """
    decision = select_encoding_backend(
        "torchcodec",
        max_used_vram_gb=5.0,
        memory_query=lambda: GpuMemory(used_mib=0, free_mib=25119, total_mib=25119),
        torchcodec_available=lambda: True,
    )
    assert decision.effective == "pyav"
    assert "PNG" in decision.reason


@pytest.mark.parametrize(
    ("memory", "available"),
    [
        (GpuMemory(used_mib=5120, free_mib=20000, total_mib=25120), True),
        (None, True),
        (GpuMemory(used_mib=0, free_mib=25120, total_mib=25120), False),
    ],
)
def test_torchcodec_safely_falls_back_to_pyav(memory, available):
    decision = select_encoding_backend(
        "torchcodec",
        max_used_vram_gb=5.0,
        memory_query=lambda: memory,
        torchcodec_available=lambda: available,
    )
    assert decision.effective == "pyav"


def test_invalid_backend_fails_fast():
    with pytest.raises(ValueError, match="encoding_backend"):
        select_encoding_backend("magic")


def test_torchcodec_disables_streaming_and_batching(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer.select_encoding_backend",
        lambda *_args: EncodingBackendDecision("torchcodec", "torchcodec", "test"),
    )
    cfg = RecorderConfig(
        repo_id="t/torchcodec",
        root=str(tmp_path),
        mock=False,
        encoding_backend="torchcodec",
        batch_encoding_size=4,
    )
    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (8, 8, 3)})
    kwargs = writer._dataset_encoding_kwargs()
    assert writer.encoding_backend == "torchcodec"
    assert kwargs["batch_encoding_size"] == 1
    assert "streaming_encoding" not in kwargs


def test_vram_fallback_uses_cpu_h264_for_auto_codec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer.select_encoding_backend",
        lambda *_args: EncodingBackendDecision("torchcodec", "pyav", "VRAM busy"),
    )
    cfg = RecorderConfig(
        repo_id="t/fallback",
        root=str(tmp_path),
        mock=False,
        encoding_backend="torchcodec",
        vcodec="auto",
    )
    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (8, 8, 3)})
    kwargs = writer._dataset_encoding_kwargs()
    assert writer.encoding_backend == "pyav"
    assert kwargs["vcodec"] == "h264"
    assert kwargs["streaming_encoding"] is True


def test_encode_time_vram_recheck_can_fall_back_after_startup(monkeypatch, tmp_path):
    decisions = iter(
        [
            EncodingBackendDecision("torchcodec", "torchcodec", "safe at startup"),
            EncodingBackendDecision("torchcodec", "pyav", "policy now uses the GPU"),
        ]
    )
    monkeypatch.setattr(
        "workstation.lerobot_recorder.dataset_writer.select_encoding_backend",
        lambda *_args: next(decisions),
    )
    cfg = RecorderConfig(
        repo_id="t/recheck",
        root=str(tmp_path),
        mock=False,
        encoding_backend="torchcodec",
        vcodec="auto",
    )
    writer = AsyncDatasetWriter(cfg, ["agentview"], {"agentview": (8, 8, 3)})
    writer._ds = SimpleNamespace(vcodec="h264_nvenc")
    writer._active_episode_frames = [{"images": {"agentview": np.zeros((8, 8, 3), np.uint8)}}]
    observed = []

    def pyav_encode(video_key, episode_index):
        observed.append((video_key, episode_index, writer._ds.vcodec))
        return Path(tmp_path) / "pyav.mp4"

    writer._pyav_encode_temporary = pyav_encode
    result = writer._encode_torchcodec_episode("observation.images.agentview", 3)

    assert result.name == "pyav.mp4"
    assert observed == [("observation.images.agentview", 3, "h264")]
    assert writer._ds.vcodec == "h264_nvenc"


@pytest.mark.skipif(importlib.util.find_spec("torchcodec") is None, reason="TorchCodec is not installed")
def test_torchcodec_encoder_writes_decodable_video(tmp_path):
    av = pytest.importorskip("av")
    frames = [np.full((64, 64, 3), i, dtype=np.uint8) for i in range(12)]
    output = tmp_path / "episode.mp4"

    encode_frames_torchcodec(frames, output, fps=30, vcodec="h264")

    with av.open(str(output)) as container:
        assert sum(1 for _ in container.decode(video=0)) == len(frames)


# ------------------------------------------------------------- PNG staging removal
class _StubDataset:
    """Enough of LeRobotDataset for the staging check."""

    def __init__(self, streaming=False):
        self._streaming_encoder = object() if streaming else None
        self.saved = []

    def _save_image(self, img, path, compress_level=1):
        self.saved.append(path)


def _writer(monkeypatch, backend: str, streaming: bool):
    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
    from workstation.lerobot_recorder import dataset_writer as dw
    from workstation.lerobot_recorder.video_encoding import EncodingBackendDecision

    monkeypatch.setattr(
        dw, "select_encoding_backend",
        lambda *a, **k: EncodingBackendDecision("torchcodec", backend, "test"),
    )
    cfg = RecorderConfig(mock=False)
    w = AsyncDatasetWriter(cfg, ["cam"], {"cam": (480, 640, 3)})
    w._ds = _StubDataset(streaming=streaming)
    return w


def test_torchcodec_writes_no_pngs(monkeypatch):
    """TorchCodec reads frames from memory, so staging them is pure waste -- 3.4 GB an
    episode of files nothing opens."""
    w = _writer(monkeypatch, "torchcodec", streaming=False)
    w._suppress_image_staging()
    w._ds._save_image(object(), "/tmp/x.png")
    assert w._ds.saved == []
    assert w._skipped_images == 1


def test_streaming_pyav_writes_no_pngs(monkeypatch):
    w = _writer(monkeypatch, "pyav", streaming=True)
    w._suppress_image_staging()
    w._ds._save_image(object(), "/tmp/x.png")
    assert w._ds.saved == []


def test_pyav_always_streams_so_png_staging_cannot_be_configured(monkeypatch, tmp_path):
    """The PNG path is not a setting any more.

    Staging every frame of every camera as a ~1.5 MB PNG used to be one config key away
    (recorder.streaming_encoding, defaulting to OFF), and a deploy run that landed there left
    7926 files and 2.7 GB behind. PyAV now always gets streaming_encoding."""
    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
    from workstation.lerobot_recorder import dataset_writer as dw
    from workstation.lerobot_recorder.video_encoding import EncodingBackendDecision

    assert not hasattr(RecorderConfig(), "streaming_encoding"), "the knob is gone on purpose"
    monkeypatch.setattr(
        dw, "select_encoding_backend",
        lambda *a, **k: EncodingBackendDecision("pyav", "pyav", "test"),
    )
    cfg = RecorderConfig(mock=False, repo_id="t/x", root=str(tmp_path))
    w = AsyncDatasetWriter(cfg, ["cam"], {"cam": (480, 640, 3)})
    assert w._dataset_encoding_kwargs()["streaming_encoding"] is True


def test_pyav_without_streaming_refuses_rather_than_writing_pngs(monkeypatch):
    """Unreachable via config now, so reaching it means lerobot dropped the kwarg. Filling
    the disk mid-episode is worse than refusing to start."""
    w = _writer(monkeypatch, "pyav", streaming=False)
    with pytest.raises(RuntimeError, match="PNG"):
        w._suppress_image_staging()
    w._ds._save_image(object(), "/tmp/x.png")
    assert w._ds.saved == ["/tmp/x.png"]  # untouched: it refused instead of half-patching


@pytest.mark.parametrize("configured_backend", ["torchcodec", "pyav"])
def test_a_real_episode_writes_no_png_whatever_the_config_says(tmp_path, configured_backend):
    """End to end, on disk: record an episode and look for PNGs.

    Every earlier attempt at this checked an intermediate -- that a kwarg was passed, that
    `_save_image` was patched -- and each time something downstream still staged frames. The
    last one suppressed the writes but left lerobot recording the paths, so episodes died at
    save time on files nothing had written. The only check worth keeping asks the filesystem.
    """
    import json

    import numpy as np

    from workstation.lerobot_recorder.config import RecorderConfig
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter

    cams = {"agentview": (64, 64, 3), "wrist_left": (64, 64, 3)}

    def frame(i):
        return {
            "state": np.zeros(42, np.float32),
            "action": np.zeros(14, np.float32),
            "leader": np.zeros(12, np.float32),
            "eef": np.zeros(14, np.float32),
            "images": {k: np.full(shape, i % 255, np.uint8) for k, shape in cams.items()},
            "task": "t",
        }

    cfg = RecorderConfig(repo_id="t/nopng", root=str(tmp_path), mock=False, fps=30,
                         use_videos=True, encoding_backend=configured_backend)
    writer = AsyncDatasetWriter(cfg, list(cams), cams)
    writer.open(frame(0))
    writer.submit([frame(i) for i in range(15)], "t", "success")
    writer.finalize()

    dataset = tmp_path / "nopng"
    assert list(dataset.rglob("*.png")) == []
    assert not (dataset / "images").exists()
    # ...and the episode actually saved, rather than "no PNGs" because nothing was written
    assert len(list(dataset.rglob("*.mp4"))) == len(cams)
    info = json.loads((dataset / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] == 15
