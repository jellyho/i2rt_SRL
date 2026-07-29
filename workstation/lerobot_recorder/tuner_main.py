"""Entry point for the interactive exposure tuner.

    workstation/yam-data tune              # all configured cameras
    workstation/yam-data tune --mock       # synthetic frames, no hardware
"""

from __future__ import annotations

import argparse
import logging
import sys

from i2rt.serving.rig_config import apply_camera_serials, find_rig, load_rig
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Interactive RealSense exposure/brightness tuner")
    p.add_argument("--config", default=None, help="config.yaml (cameras); auto-discovered by default")
    p.add_argument("--mock", action="store_true", help="synthetic frames (no RealSense)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, help="preview stream fps")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = find_rig(args.config)
    cams = apply_camera_serials(default_cameras(), load_rig(args.config))
    for cam in cams:
        cam.width, cam.height, cam.fps = args.width, args.height, args.fps
    cfg = RecorderConfig(cameras=cams, mock=args.mock)

    from workstation.lerobot_recorder.tuner_gui import run

    return run(cfg, config_path=path)


if __name__ == "__main__":
    sys.exit(main())
