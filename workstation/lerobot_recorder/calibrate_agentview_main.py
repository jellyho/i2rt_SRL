"""Entry point for the agentview extrinsic calibration GUI.

    workstation/yam-data calibrate-agentview
    workstation/yam-data calibrate-agentview --arms left
    workstation/yam-data calibrate-agentview --mock          # GUI shell only, no hardware
    workstation/yam-data calibrate-agentview --capture-button left.0   # a different handle button
    workstation/yam-data calibrate-agentview --capture-button          # Space only, no handle trigger

YAM is bimanual, so both wrist cameras are used as bridges by default -- ``--arms`` narrows that
to one if the other wrist camera is not mounted/working. Capture fires on Space OR either
configured leader-handle button (default: the "lower" button on each handle) -- both hands are
usually busy holding the robot in position by the time a pose is worth capturing.

The result is written into ``config.yaml`` itself (the same file ``--config`` points at, or the
auto-discovered one -- see :func:`i2rt.serving.rig_config.find_rig`), not a separate file: THE
single source of truth for the rig already lives there (camera serials, robot host, button map,
...), so this is the one place any tool that already calls ``load_rig()`` would look. See
:mod:`workstation.lerobot_recorder.calibrate_agentview_gui` for what it does and
:mod:`workstation.lerobot_recorder.charuco` for the geometry it solves.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from i2rt.serving.rig_config import apply_camera_serials, find_rig, load_rig
from i2rt.serving.robot_client import RobotClient
from workstation.lerobot_recorder.charuco import BoardSpec
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate agentview's extrinsic against a desk-mounted ChArUco board")
    p.add_argument("--config", default=None, help="config.yaml (cameras); auto-discovered by default")
    p.add_argument("--mock", action="store_true", help="GUI shell with synthetic frames, no hardware")
    p.add_argument("--robot-host", default="127.0.0.1")
    p.add_argument("--robot-port", type=int, default=11331)
    p.add_argument(
        "--arms",
        nargs="+",
        choices=["left", "right"],
        default=["left", "right"],
        help="which wrist camera(s) can bridge to the base frame (default: both)",
    )
    p.add_argument(
        "--capture-button",
        dest="capture_buttons",
        nargs="*",
        default=["left.1", "right.1"],
        metavar="SIDE.INDEX",
        help=(
            "leader-handle button(s) that trigger a capture (any ONE firing is enough), "
            "'<side>.<index>' e.g. left.1 -- upper=0, lower=1, same convention as config.py's "
            "button_map. Pass with no values to disable the handle trigger (Space still works)."
        ),
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    # Board geometry -- MUST match the printed board (see charuco.BoardSpec).
    p.add_argument("--squares-x", type=int, default=8)
    p.add_argument("--squares-y", type=int, default=6)
    p.add_argument("--square-length-m", type=float, default=0.030)
    p.add_argument("--marker-length-m", type=float, default=0.022)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    arms = list(dict.fromkeys(args.arms))  # de-dup, keep order, in case of a repeated flag

    path = find_rig(args.config)
    cams = apply_camera_serials(default_cameras(), load_rig(args.config))
    wanted = {"agentview", *(f"wrist_{arm}" for arm in arms)}
    cams = [c for c in cams if c.key in wanted]
    for cam in cams:
        cam.width, cam.height, cam.fps = args.width, args.height, args.fps
    cfg = RecorderConfig(cameras=cams, mock=args.mock)

    robot = None
    if not args.mock:
        robot = RobotClient(host=args.robot_host, port=args.robot_port, timeout=2.0)

    # Same combined arm+gripper model, and the same published wrist extrinsic, that
    # render_deploy_samples.py's candidate fan uses -- one source of truth for "where is the
    # wrist camera", not a second copy that could drift from it. Both arms share the model (it
    # is the kinematic chain, not a per-side mirror); each gets its own instance so its FK is
    # driven by that arm's own joints only.
    from yam_policy.viz import WristCameraGeometry

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    geometries = {arm: WristCameraGeometry(xml) for arm in arms}

    board = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dictionary=args.dictionary,
    )

    from workstation.lerobot_recorder.calibrate_agentview_gui import run

    logging.info("calibrate-agentview: config=%s arms=%s capture_buttons=%s", path, arms, args.capture_buttons)
    return run(
        cfg, robot, geometries, board=board, config_path=path, mock=args.mock, capture_buttons=args.capture_buttons
    )


if __name__ == "__main__":
    sys.exit(main())
