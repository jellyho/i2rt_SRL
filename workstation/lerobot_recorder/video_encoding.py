"""Workstation video-encoding backend selection and TorchCodec adapter.

LeRobot currently owns the dataset container/metadata, but its recorder encodes
episode videos with PyAV.  This module provides a narrow TorchCodec adapter for
the workstation recorder and keeps the safety decision independent of LeRobot's
*decoder* ``video_backend`` setting.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor

VALID_ENCODING_BACKENDS = frozenset({"torchcodec", "pyav"})


@dataclass(frozen=True)
class GpuMemory:
    used_mib: int
    free_mib: int
    total_mib: int


@dataclass(frozen=True)
class EncodingBackendDecision:
    requested: str
    effective: str
    reason: str
    gpu_memory: GpuMemory | None = None


def query_primary_gpu_memory() -> GpuMemory | None:
    """Return primary-GPU memory from ``nvidia-smi``, or ``None`` safely."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        used, free, total = (int(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return GpuMemory(used_mib=used, free_mib=free, total_mib=total)
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def torchcodec_encoder_available() -> bool:
    try:
        return importlib.util.find_spec("torchcodec.encoders") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def select_encoding_backend(
    requested: str,
    max_used_vram_gb: float = 5.0,
    *,
    memory_query: Callable[[], GpuMemory | None] = query_primary_gpu_memory,
    torchcodec_available: Callable[[], bool] = torchcodec_encoder_available,
) -> EncodingBackendDecision:
    """Resolve the requested backend at writer startup or just before encoding.

    ``torchcodec`` is the default/preferred backend, but it safely falls back to
    PyAV when TorchCodec is unavailable, GPU memory cannot be measured, or the
    primary GPU is already using at least ``max_used_vram_gb``.  An explicit
    ``pyav`` request never probes or imports TorchCodec.
    """
    requested = str(requested).strip().lower()
    if requested not in VALID_ENCODING_BACKENDS:
        allowed = ", ".join(sorted(VALID_ENCODING_BACKENDS))
        raise ValueError(f"encoding_backend must be one of {allowed}; got {requested!r}")
    if requested == "pyav":
        return EncodingBackendDecision(requested, "pyav", "explicitly configured")
    if not torchcodec_available():
        return EncodingBackendDecision(requested, "pyav", "TorchCodec encoder is unavailable")

    memory = memory_query()
    if memory is None:
        return EncodingBackendDecision(requested, "pyav", "GPU memory could not be measured")
    limit_mib = int(float(max_used_vram_gb) * 1024)
    if memory.used_mib >= limit_mib:
        return EncodingBackendDecision(
            requested,
            "pyav",
            f"GPU already uses {memory.used_mib} MiB (limit {limit_mib} MiB)",
            memory,
        )
    return EncodingBackendDecision(
        requested,
        "torchcodec",
        f"GPU uses {memory.used_mib} MiB (below {limit_mib} MiB limit)",
        memory,
    )


def _as_chw_uint8(image: np.ndarray) -> "Tensor":
    import torch

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 video frame, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def encode_frames_torchcodec(
    frames: Iterable[np.ndarray],
    video_path: str | Path,
    *,
    fps: int,
    vcodec: str,
) -> None:
    """Encode in-memory HWC uint8 frames with TorchCodec 0.9+'s batch API.

    Frames stay in host memory.  NVENC is still selected by ``h264_nvenc`` but
    avoiding a full-episode CUDA tensor keeps peak VRAM small enough to coexist
    with a policy process.
    """
    import torch
    from torchcodec.encoders import VideoEncoder

    tensors = [_as_chw_uint8(frame) for frame in frames]
    if not tensors:
        raise ValueError("cannot encode an empty video")
    frame_tensor = torch.stack(tensors)
    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "h264_nvenc" if vcodec == "auto" else vcodec
    encoder = VideoEncoder(frame_tensor, frame_rate=float(fps))
    if codec.endswith("_nvenc"):
        encoder.to_file(path, codec=codec, pixel_format="yuv420p", extra_options={"qp": 30})
    else:
        encoder.to_file(path, codec=codec, pixel_format="yuv420p", crf=30)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"TorchCodec did not produce a video at {path}")
