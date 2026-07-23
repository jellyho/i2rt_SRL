"""Write a reproducible, credential-free policy deployment manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
from datetime import datetime, timezone


def _command(*args: str, cwd: pathlib.Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=pathlib.Path, required=True)
    parser.add_argument("--i2rt-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-horizon", type=int, required=True)
    parser.add_argument("--execution-horizon", type=int, required=True)
    parser.add_argument("--control-hz", type=float, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "yam_bimanual_v1",
        "openpi_commit": _command("git", "rev-parse", "HEAD", cwd=args.openpi_root),
        "openpi_dirty": bool(_command("git", "status", "--porcelain", cwd=args.openpi_root)),
        "i2rt_commit": _command("git", "rev-parse", "HEAD", cwd=args.i2rt_root),
        "i2rt_dirty": bool(_command("git", "status", "--porcelain", cwd=args.i2rt_root)),
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "config_name": args.config_name,
        "prompt": args.prompt,
        "model_action_horizon": args.model_horizon,
        "execution_horizon": args.execution_horizon,
        "control_hz": args.control_hz,
        "compute_hostname": platform.node(),
        "gpu": _command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
