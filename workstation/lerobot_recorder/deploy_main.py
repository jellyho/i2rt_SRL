"""Entry point for the policy deployment UI.

This replaces the normal headless ``yam-data bridge`` operator flow: the UI owns
policy streaming and (optionally) DAgger recording in one workstation process.

    workstation/yam-data deploy      # everything else is chosen in the UI

A run has two independent axes, both on the setup page: **mode** (deploy = watch the
policy, leader free; dagger = correct it, leader mirrors) and **record** (whether the run
lands in a dataset). The flags below only set their initial values.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from i2rt.serving.rig_config import Resolver, apply_camera_serials, load_rig
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras
from workstation.lerobot_recorder.deploy_gui import DeployGUI
from workstation.policy_bridge.config import BridgeConfig


def build_configs(
    argv: Optional[List[str]] = None,
) -> Tuple[RecorderConfig, BridgeConfig, str, bool, bool]:
    p = argparse.ArgumentParser(description="YAM policy deployment UI (DAgger or watch-only)")
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
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--no-async", action="store_true", help="disable action-chunk prefetch")
    p.add_argument("--min-free-gb", type=float, default=1.0)
    p.add_argument("--no-review", action="store_true", help="auto-save each DAgger segment")
    p.add_argument("--auto-arm", action="store_true", help="arm collection automatically on Start")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset at --root")
    p.add_argument(
        "--mode",
        choices=["deploy", "dagger"],
        default="dagger",
        help="deploy = watch the policy (leader free); dagger = correct it (leader mirrors). "
        "Also selectable on the setup page.",
    )
    p.add_argument(
        "--no-record",
        action="store_true",
        help="start with recording OFF (no dataset is created or opened). Independent of "
        "--mode; also togglable on the setup page.",
    )
    mirror = p.add_mutually_exclusive_group()
    mirror.add_argument(
        "--leader-mirror",
        dest="leader_mirror",
        action="store_true",
        default=None,
        help="force leader mirroring ON regardless of the record mode",
    )
    mirror.add_argument(
        "--no-leader-mirror",
        dest="leader_mirror",
        action="store_false",
        help="force leader mirroring OFF: the handles hang free instead of tracking the "
        "follower while the policy drives",
    )
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
        record_source=DeployGUI.SOURCES[(args.mode, not args.no_record)],
        resume=args.resume,
        min_free_gb=float(rec.get("min_free_gb")),
        mock=args.mock,
        review_before_save=review_before_save,
        auto_arm=auto_arm,
    )
    recorder_cfg.use_videos = bool(rec_section.get("use_videos", recorder_cfg.use_videos))
    recorder_cfg.vcodec = str(rec_section.get("vcodec", recorder_cfg.vcodec))
    recorder_cfg.encoding_backend = str(rec_section.get("encoding_backend", recorder_cfg.encoding_backend))
    recorder_cfg.torchcodec_max_used_vram_gb = float(
        rec_section.get("torchcodec_max_used_vram_gb", recorder_cfg.torchcodec_max_used_vram_gb)
    )
    recorder_cfg.encoder_threads = int(rec_section.get("encoder_threads", recorder_cfg.encoder_threads))
    recorder_cfg.batch_encoding_size = int(rec_section.get("batch_encoding_size", recorder_cfg.batch_encoding_size))
    recorder_cfg.image_writer_threads = int(rec_section.get("image_writer_threads", recorder_cfg.image_writer_threads))
    recorder_cfg.image_writer_processes = int(
        rec_section.get("image_writer_processes", recorder_cfg.image_writer_processes)
    )
    recorder_cfg.streaming_encoding = bool(
        rec_section.get("streaming_encoding", recorder_cfg.streaming_encoding)
    )
    recorder_cfg.expected_robot_mode = "dagger"  # only that controller accepts policy actions

    bridge_cfg = BridgeConfig(
        robot_host=recorder_cfg.robot_host,
        robot_port=recorder_cfg.robot_port,
        policy_host=pol.get("policy_host", key="host"),
        policy_port=int(pol.get("policy_port", key="port")),
        action_horizon=args.action_horizon,
        rate_hz=args.rate,
        image_size=args.image_size,
        prompt=task,
        use_async=not args.no_async,
    )
    return recorder_cfg, bridge_cfg, args.mode, not args.no_record, args.leader_mirror


def main(argv: Optional[List[str]] = None) -> None:
    recorder_cfg, bridge_cfg, mode, record, leader_mirror = build_configs(argv)
    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder import theme
    from workstation.lerobot_recorder.deploy_gui import DeployGUI

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)
    gui = DeployGUI(
        recorder_cfg, bridge_cfg, mode=mode, record=record, leader_mirror=leader_mirror
    )
    gui.resize(900, 980)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
