"""Benchmark workstation PyAV and TorchCodec recording encoders.

The benchmark keeps one synthetic camera episode in RAM and measures only the
backend work needed to produce the final H.264 MP4. GPU memory is sampled with
``nvidia-smi`` throughout each run.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from workstation.lerobot_recorder.video_encoding import encode_frames_torchcodec, query_primary_gpu_memory


class PeakVramMonitor:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None

    def __enter__(self) -> "PeakVramMonitor":
        sample = query_primary_gpu_memory()
        self.baseline_mib = None if sample is None else sample.used_mib
        self.peak_mib = self.baseline_mib

        def poll() -> None:
            while not self._stop.wait(self.interval_s):
                current = query_primary_gpu_memory()
                if current is not None:
                    self.peak_mib = max(self.peak_mib or 0, current.used_mib)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def delta_mib(self) -> int | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib)


def synthetic_frames(count: int, height: int, width: int) -> np.ndarray:
    """Generate deterministic, compressible motion without camera hardware."""
    frames = np.empty((count, height, width, 3), dtype=np.uint8)
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    for index in range(count):
        frames[index, :, :, 0] = (x + index * 3) % 256
        frames[index, :, :, 1] = (y + index * 2) % 256
        frames[index, :, :, 2] = ((x // 2 + y // 2 + index * 5) % 256).astype(np.uint8)
    return frames


def encode_pyav(frames: np.ndarray, output: Path, fps: int, codec: str) -> None:
    import av

    options = {"g": "2", "crf": "30"}
    if codec.endswith("_nvenc"):
        options = {"rc": "constqp", "qp": "30"}
    with av.open(str(output), "w") as container:
        stream = container.add_stream(codec, fps, options=options)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def benchmark(name: str, callback: Callable[[], None], output: Path, frame_count: int) -> dict:
    output.unlink(missing_ok=True)
    with PeakVramMonitor() as monitor:
        started = time.perf_counter()
        callback()
        elapsed = time.perf_counter() - started
    return {
        "backend": name,
        "seconds": round(elapsed, 4),
        "frames_per_second": round(frame_count / elapsed, 2),
        "peak_vram_delta_mib": monitor.delta_mib,
        "baseline_vram_mib": monitor.baseline_mib,
        "peak_total_vram_mib": monitor.peak_mib,
        "output_mib": round(output.stat().st_size / 1024**2, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    frames = synthetic_frames(args.frames, args.height, args.width)
    with tempfile.TemporaryDirectory(prefix="yam-encoder-benchmark-") as directory:
        root = Path(directory)
        pyav_output = root / "pyav.mp4"
        torchcodec_output = root / "torchcodec.mp4"
        results = [
            benchmark(
                "pyav",
                lambda: encode_pyav(frames, pyav_output, args.fps, args.codec),
                pyav_output,
                args.frames,
            ),
            benchmark(
                "torchcodec",
                lambda: encode_frames_torchcodec(frames, torchcodec_output, fps=args.fps, vcodec=args.codec),
                torchcodec_output,
                args.frames,
            ),
        ]

    gpu = query_primary_gpu_memory()
    report = {
        "system": {
            "platform": platform.platform(),
            "frames": args.frames,
            "resolution": f"{args.width}x{args.height}",
            "fps": args.fps,
            "codec": args.codec,
            "gpu_total_mib": None if gpu is None else gpu.total_mib,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
