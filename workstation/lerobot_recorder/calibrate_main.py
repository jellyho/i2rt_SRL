"""Entry point for the full-rig camera calibration GUI (wrist extrinsics + agentview + offset).

    workstation/yam-data calibrate
    workstation/yam-data calibrate --arms left
    workstation/yam-data calibrate --mock                     # GUI shell only, no hardware
    workstation/yam-data calibrate --capture-button left.0    # enable a handle button (see below)

One ChArUco board on the desk recovers each wrist camera's own extrinsic (hand-eye), each
agentview extrinsic, and the left<->right arm offset -- see
:mod:`workstation.lerobot_recorder.calibrate_gui` for the flow and
:mod:`workstation.lerobot_recorder.charuco` for the geometry.

YAM is bimanual, so both wrist cameras are used as bridges by default -- ``--arms`` narrows that
to one if the other wrist camera is not mounted/working. **Capture is Space by default**, NOT a
handle button: in teleop the robot server consumes the handles while engaged (outcome buttons
force homing, the fine button starts recentering), so a press there would move the arm rather than
capture. ``--capture-button <side>.<index>`` opts into a handle trigger only if you are running a
robot mode that leaves the handles free.

The result is written into ``config.yaml`` itself (the same file ``--config`` points at, or the
auto-discovered one -- see :func:`i2rt.serving.rig_config.find_rig`), not a separate file: THE
single source of truth for the rig already lives there (camera serials, robot host, button map,
...), so this is the one place any tool that already calls ``load_rig()`` would look.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Optional, Sequence

from i2rt.serving.rig_config import apply_camera_serials, find_rig, load_rig
from i2rt.serving.robot_client import RobotClient
from workstation.lerobot_recorder.charuco import BoardSpec
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Calibrate the camera rig (wrist extrinsics + agentview + arm offset) from one desk ChArUco board"
    )
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
        default=[],
        metavar="SIDE.INDEX",
        help=(
            "OPT-IN leader-handle button(s) that also trigger a capture (any ONE firing), "
            "'<side>.<index>' e.g. left.1 -- upper=0, lower=1. Default OFF: in teleop the robot "
            "consumes the handles (homing/recentering), so enable this only against a robot mode "
            "that leaves them free. Space is always the capture key regardless."
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

    # Board geometry: config.yaml's calibration.board is the baseline; any explicitly passed CLI
    # flag overrides that field. So a rig configured once needs no board flags thereafter, and a
    # one-off different board can still be given on the command line.
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
        "calibrate: config=%s arms=%s auto_capture=%s capture_buttons=%s",
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
