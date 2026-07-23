"""Entry point for the DAgger deployment UI.

This replaces the normal headless ``yam-data bridge`` operator flow: the UI owns
policy streaming and DAgger recording in one workstation process.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from typing import List, Optional, Tuple

from i2rt.serving.rig_config import Resolver, apply_camera_serials, load_rig
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras
from workstation.policy_bridge.config import BridgeConfig

logger = logging.getLogger(__name__)

_NVENC_MIN_FREE_MIB = 3 * 1024


def _free_vram_mib() -> Optional[int]:
    """Return free VRAM on the primary NVIDIA GPU, or None when unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _select_deploy_codec(explicit_codec: Optional[str]) -> str:
    if explicit_codec is not None:
        logger.warning("deploy video codec explicitly set to %s", explicit_codec)
        return str(explicit_codec)

    free_mib = _free_vram_mib()
    if free_mib is not None and free_mib > _NVENC_MIN_FREE_MIB:
        logger.warning("deploy selected GPU h264_nvenc encoding (%d MiB VRAM free)", free_mib)
        return "h264_nvenc"

    if free_mib is None:
        logger.warning("deploy selected CPU h264 encoding (free VRAM unavailable)")
    else:
        logger.warning(
            "deploy selected CPU h264 encoding (%d MiB VRAM free; GPU requires >%d MiB)",
            free_mib,
            _NVENC_MIN_FREE_MIB,
        )
    return "h264"


def build_configs(argv: Optional[List[str]] = None) -> Tuple[RecorderConfig, BridgeConfig]:
    p = argparse.ArgumentParser(description="YAM DAgger deployment UI")
    p.add_argument("--config", default=None, help="config.yaml (robot/policy/cameras/recorder)")
    p.add_argument("--repo-id", default="user/yam_bimanual")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--task", default="do the task", help="active language instruction")
    p.add_argument("--prompt", default=None, help="policy prompt; defaults to --task")
    p.add_argument("--tasks", default="", help="';'-separated task templates for quick switching")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--robot-host", default="127.0.0.1")
    p.add_argument("--robot-port", type=int, default=11331)
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--contract", default="yam_bimanual_v1")
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--execution-horizon", "--action-horizon", dest="execution_horizon", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--camera-max-age", type=float, default=0.25)
    p.add_argument("--inference-timeout", type=float, default=2.0)
    p.add_argument("--no-async", action="store_true", help="disable action-chunk prefetch")
    p.add_argument(
        "--codec",
        "--vcodec",
        dest="codec",
        default=None,
        help="force video codec; otherwise NVENC is selected when >3 GiB VRAM is free",
    )
    p.add_argument("--min-free-gb", type=float, default=1.0)
    p.add_argument("--no-review", action="store_true", help="auto-save each DAgger segment")
    p.add_argument("--auto-arm", action="store_true", help="arm collection automatically on Start")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset at --root")
    p.add_argument("--mock", action="store_true", help="synthetic cameras/robot state")
    p.add_argument("--serials", default="", help="comma-separated RealSense serials")
    args = p.parse_args(argv)

    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    rob = Resolver(args, p, rig.get("robot", {}))
    pol = Resolver(args, p, rig.get("policy", {}))

    cams = apply_camera_serials(default_cameras(), rig)
    if args.serials:
        for cam, serial in zip(cams, [s.strip() for s in args.serials.split(",")], strict=False):
            cam.serial = serial

    rec_section = rig.get("recorder", {}) or {}
    tasks = [t.strip() for t in args.tasks.split(";") if t.strip()] or list(rig.get("tasks", []) or [])
    task = args.prompt or rec.get("task")
    if task == p.get_default("task") and "task" not in rec_section and tasks:
        task = tasks[0]
    review_before_save = bool(rec_section.get("review_before_save", True)) and not args.no_review
    auto_arm = bool(rec_section.get("auto_arm", False)) or args.auto_arm

    recorder_cfg = RecorderConfig(
        repo_id=rec.get("repo_id"),
        root=rec.get("root"),
        task=task,
        tasks=tasks,
        fps=int(rec.get("fps")),
        cameras=cams,
        robot_host=rob.get("robot_host", key="host"),
        robot_port=int(rob.get("robot_port", key="port")),
        record_source="dagger",
        resume=args.resume,
        min_free_gb=float(rec.get("min_free_gb")),
        mock=args.mock,
        review_before_save=review_before_save,
        auto_arm=auto_arm,
    )
    recorder_cfg.use_videos = bool(rec_section.get("use_videos", recorder_cfg.use_videos))
    recorder_cfg.vcodec = _select_deploy_codec(args.codec)
    recorder_cfg.encoder_threads = int(rec_section.get("encoder_threads", recorder_cfg.encoder_threads))
    recorder_cfg.batch_encoding_size = int(rec_section.get("batch_encoding_size", recorder_cfg.batch_encoding_size))
    recorder_cfg.image_writer_threads = int(rec_section.get("image_writer_threads", recorder_cfg.image_writer_threads))
    recorder_cfg.image_writer_processes = int(
        rec_section.get("image_writer_processes", recorder_cfg.image_writer_processes)
    )

    bridge_cfg = BridgeConfig(
        robot_host=recorder_cfg.robot_host,
        robot_port=recorder_cfg.robot_port,
        policy_host=pol.get("policy_host", key="host"),
        policy_port=int(pol.get("policy_port", key="port")),
        contract=str(pol.get("contract")),
        execution_horizon=int(pol.get("execution_horizon")),
        rate_hz=float(pol.get("rate", key="rate_hz")),
        image_size=int(pol.get("image_size")),
        prompt=task,
        use_async=not args.no_async,
        camera_max_age_s=float(pol.get("camera_max_age", key="camera_max_age_s")),
        inference_timeout_s=float(pol.get("inference_timeout", key="inference_timeout_s")),
    )
    return recorder_cfg, bridge_cfg


def main(argv: Optional[List[str]] = None) -> None:
    recorder_cfg, bridge_cfg = build_configs(argv)
    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder import theme
    from workstation.lerobot_recorder.deploy_gui import DeployGUI

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    gui = DeployGUI(recorder_cfg, bridge_cfg)
    gui.resize(900, 980)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
