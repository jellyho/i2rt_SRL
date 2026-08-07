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


def test_torchcodec_is_preferred_below_five_gib_used():
    decision = select_encoding_backend(
        "torchcodec",
        max_used_vram_gb=5.0,
        memory_query=lambda: GpuMemory(used_mib=5119, free_mib=20000, total_mib=25119),
        torchcodec_available=lambda: True,
    )
    assert decision.effective == "torchcodec"


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
        streaming_encoding=True,
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
        streaming_encoding=True,
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
    cfg = RecorderConfig(mock=False, streaming_encoding=streaming)
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


def test_pyav_without_streaming_keeps_staging_and_says_why(monkeypatch, caplog):
    """That encoder reads the PNGs off disk, so removing them would break encoding --
    leave them and explain, rather than silently produce empty videos."""
    w = _writer(monkeypatch, "pyav", streaming=False)
    with caplog.at_level("ERROR"):
        w._suppress_image_staging()
    w._ds._save_image(object(), "/tmp/x.png")
    assert w._ds.saved == ["/tmp/x.png"]
    assert "streaming_encoding" in caplog.text
