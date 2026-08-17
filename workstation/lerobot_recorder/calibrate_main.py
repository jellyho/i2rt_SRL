"""Entry point for the agentview camera calibration GUI (board-on-gripper, eye-to-hand).

    workstation/yam-data calibrate                     # both arms; grasp the board, one arm at a time
    workstation/yam-data calibrate --arms left         # only calibrate agentview from the left gripper
    workstation/yam-data calibrate --mock              # GUI shell only, no hardware
    workstation/yam-data calibrate --capture-button left.1   # enable a handle button (see below)

The agentview camera is mounted too high to co-see a desk board with a wrist camera, so it is
calibrated the other way round: **grasp the board with the gripper** and lift it into agentview's
view. agentview + forward kinematics alone recover ``base_T_agentview`` per arm -- see
:mod:`workstation.lerobot_recorder.calibrate_gui` for the flow and
:mod:`workstation.lerobot_recorder.charuco` for the geometry. The wrist cameras are NOT calibrated
here; their mount comes from the CAD ``T_GRIPPER_CAMERA`` constant.

**Capture is hands-free by default** (hold the arm still while engaged). **Space** is the manual
fallback, NOT a handle button: in teleop the robot server consumes the handles while engaged, so a
press there would move the arm. ``--capture-button <side>.<index>`` opts into a handle trigger only
against a robot mode that leaves the handles free.

The robot host/port come from config.yaml's ``robot.host``/``robot.port`` when the flags are not
passed. The result is written back into ``config.yaml`` itself (``cameras.agentview.extrinsic`` +
``calibration.board``) -- THE single source of truth for the rig.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Optional, Sequence

from i2rt.serving.rig_config import Resolver, apply_camera_serials, find_rig, load_rig
from i2rt.serving.robot_client import RobotClient
from workstation.lerobot_recorder.charuco import BoardSpec
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Calibrate the agentview camera from a ChArUco board grasped on the gripper (eye-to-hand)"
    )
    p.add_argument("--config", default=None, help="config.yaml (cameras); auto-discovered by default")
    p.add_argument("--mock", action="store_true", help="GUI shell with synthetic frames, no hardware")
    # Defaults left at the built-ins; config.yaml's robot.host/port fills in when the flag is not
    # passed (see Resolver below) -- so a configured rig needs no --robot-host, same as deploy.
    p.add_argument("--robot-host", default="127.0.0.1", help="overrides config.yaml robot.host")
    p.add_argument("--robot-port", type=int, default=11331, help="overrides config.yaml robot.port")
    p.add_argument(
        "--arms",
        nargs="+",
        choices=["left", "right"],
        default=["left", "right"],
        help="which arm(s) can grasp the board to calibrate agentview from (default: both)",
    )
    p.add_argument(
        "--capture-button",
        dest="capture_buttons",
        nargs="*",
        default=[],
        metavar="SIDE.INDEX",
        help=(
            "OPT-IN leader-handle button(s) that also trigger a capture (any ONE firing), "
            "'<side>.<index>' e.g. left.1 -- upper=0, lower=1. Default OFF: in teleop the robot "
            "consumes the handles, so enable this only against a robot mode that leaves them free. "
            "Space is always the capture key regardless."
        ),
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--no-auto-capture",
        dest="auto_capture",
        action="store_false",
        help="disable hands-free auto-capture (hold the arm still while engaged -> captures); "
        "on by default since both hands are on the leaders",
    )
    p.add_argument("--auto-dwell", type=float, default=1.0, help="seconds to hold still before an auto-capture")
    # Board geometry -- MUST match the printed board (see charuco.BoardSpec). Default None so an
    # unpassed flag falls through to config.yaml's calibration.board, then to the BoardSpec
    # default; a passed flag overrides both. **Measure the printed squares -- don't trust the
    # page-fit scale.**
    p.add_argument("--squares-x", type=int, default=None)
    p.add_argument("--squares-y", type=int, default=None)
    p.add_argument("--square-length-m", type=float, default=None)
    p.add_argument("--marker-length-m", type=float, default=None)
    p.add_argument("--dictionary", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    arms = list(dict.fromkeys(args.arms))  # de-dup, keep order, in case of a repeated flag

    path = find_rig(args.config)
    rig = load_rig(args.config)
    cams = apply_camera_serials(default_cameras(), rig)
    cams = [c for c in cams if c.key == "agentview"]  # eye-to-hand uses agentview only
    for cam in cams:
        cam.width, cam.height, cam.fps = args.width, args.height, args.fps
    cfg = RecorderConfig(cameras=cams, mock=args.mock)

    robot = None
    if not args.mock:
        # CLI flag wins; otherwise config.yaml's robot.host/port; otherwise the built-in default.
        rob = Resolver(args, p, rig.get("robot", {}))
        robot = RobotClient(
            host=rob.get("robot_host", key="host"), port=int(rob.get("robot_port", key="port")), timeout=2.0
        )

    # The combined arm+gripper model the robot runs, one instance per arm so each arm's FK is
    # driven by its own joints. FK only -- the wrist extrinsic is not used by eye-to-hand.
    from yam_policy.viz import WristCameraGeometry

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    geometries = {arm: WristCameraGeometry(xml) for arm in arms}

    # Board geometry: config.yaml's calibration.board is the baseline; any explicitly passed CLI
    # flag overrides that field.
    board = BoardSpec.from_config((rig.get("calibration") or {}).get("board"))
    overrides = {
        k: v
        for k, v in {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length_m,
            "marker_length_m": args.marker_length_m,
            "dictionary": args.dictionary,
        }.items()
        if v is not None
    }
    board = dataclasses.replace(board, **overrides) if overrides else board

    from workstation.lerobot_recorder.calibrate_gui import run

    logging.info(
        "calibrate agentview: config=%s arms=%s auto_capture=%s capture_buttons=%s",
        path,
        arms,
        args.auto_capture,
        args.capture_buttons,
    )
    return run(
        cfg,
        robot,
        geometries,
        board=board,
        config_path=path,
        mock=args.mock,
        capture_buttons=args.capture_buttons,
        auto_capture=args.auto_capture,
        auto_dwell_s=args.auto_dwell,
    )


if __name__ == "__main__":
    sys.exit(main())
