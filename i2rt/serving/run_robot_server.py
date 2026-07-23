"""Run the YAM robot server on the robot machine.

    python -m i2rt.serving.run_robot_server teleop  [--sim] [--bilateral-kp 0.15]
    python -m i2rt.serving.run_robot_server dagger  [--sim] [--mirror-kp 0.2]
    python -m i2rt.serving.run_robot_server wrapper [--sim]            # replay target

The workstation connects with :class:`i2rt.serving.robot_client.RobotClient`.
"""

from __future__ import annotations

import argparse
import logging

from i2rt.robots.utils import ArmType, GripperType
from i2rt.serving import control_config as cc
from i2rt.serving.controllers import (
    DaggerConfig,
    DaggerController,
    TeleopConfig,
    TeleopController,
    WrapperConfig,
    WrapperController,
)
from i2rt.serving.rig_config import apply_control_overrides, load_rig, teleop_button_outcomes
from i2rt.serving.robot_server import DEFAULT_PORT, RobotServer


class _DropChatter(logging.Filter):
    """Drop the periodic per-motor-chain status lines the driver prints on a timer.

    The motor/CAN driver logs routine throughput stats for each of the four
    chains every ~10-30 s, which floods the console. We hide those lines but
    keep real warnings (e.g. "control loop is slow"), which use other messages.
    """

    _NOISE = (
        "Grav Comp Control Frequency",  # motor_chain_robot grav-comp loop
        "Total rate:",  # DMChainCanInterface throughput report
        "s Report] step_time",  # DMChainCanInterface step-time report
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._NOISE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    for _h in logging.getLogger().handlers:
        _h.addFilter(_DropChatter())

    # Phase 1: find --config anywhere and apply control overrides, so the argparse
    # defaults below (and the live control_config used by the controllers) reflect it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    rig = load_rig(known.config)
    applied = apply_control_overrides(rig)
    if applied:
        logging.info("rig control overrides: %s", applied)
    robot_sec = rig.get("robot", {}) or {}
    default_port = int(robot_sec.get("port", DEFAULT_PORT))
    # Gripper / arm come from config.yaml `robot:` (CLI flags below override per run).
    default_arm = str(robot_sec.get("arm_type", "yam"))
    default_gripper = str(robot_sec.get("gripper", "linear_4310"))
    default_leader_gripper = str(robot_sec.get("leader_gripper", "yam_teaching_handle"))

    p = argparse.ArgumentParser(description="YAM robot server (portal)")
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("teleop", help="auto home/engage bimanual teleop")
    pt.add_argument("--config", default=None, help="config.yaml (control overrides + port)")
    pt.add_argument("--port", type=int, default=default_port)
    pt.add_argument("--sim", action="store_true")
    pt.add_argument("--home", default="")
    pt.add_argument("--engage-thr", type=float, default=cc.ENGAGE_THR)
    pt.add_argument("--release-thr", type=float, default=cc.RELEASE_THR)
    pt.add_argument("--dwell", type=float, default=cc.DWELL_S)
    pt.add_argument("--home-kp", type=float, default=cc.HOME_KP)
    pt.add_argument("--bilateral-kp", type=float, default=cc.BILATERAL_KP)
    pt.add_argument("--fine-grained-scale", type=float, default=cc.FINE_GRAINED_SCALE)
    pt.add_argument("--fine-grained-button", default=cc.FINE_GRAINED_BUTTON)
    pt.add_argument("--fine-recenter-speed", type=float, default=cc.FINE_RECENTER_SPEED)
    pt.add_argument("--fine-recenter-kp", type=float, default=cc.FINE_RECENTER_KP)
    pt.add_argument(
        "--fine-recenter-max-following-error",
        type=float,
        default=cc.FINE_RECENTER_MAX_FOLLOWING_ERROR,
    )
    pt.add_argument("--fine-recenter-tolerance", type=float, default=cc.FINE_RECENTER_TOLERANCE)
    pt.add_argument("--fine-recenter-dwell", type=float, default=cc.FINE_RECENTER_DWELL)
    pt.add_argument("--fine-recenter-timeout", type=float, default=cc.FINE_RECENTER_TIMEOUT)
    pt.add_argument("--rate", type=float, default=120.0)
    pt.add_argument("--ramp-speed", type=float, default=cc.RAMP_SPEED)
    pt.add_argument("--home-speed", type=float, default=cc.HOME_SPEED, help="rad/s for the (gentle) homing return")
    pt.add_argument("--gate-joints", default=",".join(str(j) for j in cc.GATE_JOINTS))
    pt.add_argument("--arm-type", default=default_arm)
    pt.add_argument("--gripper", default=default_gripper, help="follower (end-effector) gripper type")
    pt.add_argument("--leader-gripper", default=default_leader_gripper)

    pd = sub.add_parser("dagger", help="HG-DAgger policy + button takeover")
    pd.add_argument("--config", default=None, help="config.yaml (control overrides + port)")
    pd.add_argument("--port", type=int, default=default_port)
    pd.add_argument("--sim", action="store_true")
    pd.add_argument("--home", default="")
    pd.add_argument("--mirror-kp", type=float, default=cc.DAGGER_MIRROR_KP)
    pd.add_argument("--feedback-kp", type=float, default=cc.DAGGER_FEEDBACK_KP)
    pd.add_argument("--fine-grained-scale", type=float, default=cc.FINE_GRAINED_SCALE)
    pd.add_argument("--fine-grained-button", default=cc.FINE_GRAINED_BUTTON)
    pd.add_argument("--fine-recenter-speed", type=float, default=cc.FINE_RECENTER_SPEED)
    pd.add_argument("--fine-recenter-kp", type=float, default=cc.FINE_RECENTER_KP)
    pd.add_argument(
        "--fine-recenter-max-following-error",
        type=float,
        default=cc.FINE_RECENTER_MAX_FOLLOWING_ERROR,
    )
    pd.add_argument("--fine-recenter-tolerance", type=float, default=cc.FINE_RECENTER_TOLERANCE)
    pd.add_argument("--fine-recenter-dwell", type=float, default=cc.FINE_RECENTER_DWELL)
    pd.add_argument("--fine-recenter-timeout", type=float, default=cc.FINE_RECENTER_TIMEOUT)
    pd.add_argument("--home-kp", type=float, default=cc.HOME_KP)
    pd.add_argument("--home-speed", type=float, default=cc.HOME_SPEED, help="rad/s for DAgger keep/discard homing")
    pd.add_argument("--rate", type=float, default=120.0)
    pd.add_argument("--max-joint-speed", type=float, default=1.5)
    pd.add_argument("--arm-type", default=default_arm)
    pd.add_argument("--gripper", default=default_gripper, help="follower (end-effector) gripper type")
    pd.add_argument("--leader-gripper", default=default_leader_gripper)

    pw = sub.add_parser("wrapper", help="followers track an external command (replay)")
    pw.add_argument("--config", default=None, help="config.yaml (control overrides + port/channels)")
    pw.add_argument("--port", type=int, default=default_port)
    pw.add_argument("--sim", action="store_true")
    pw.add_argument("--arm-type", default=default_arm)
    pw.add_argument("--gripper", default=default_gripper)
    pw.add_argument("--rate", type=float, default=100.0)
    pw.add_argument("--max-joint-speed", type=float, default=1.5)
    pw.add_argument("--control", choices=["joint", "eef"], default="joint", help="command space (eef is experimental)")

    args = p.parse_args()

    # Fail fast (with the valid list) if a config/CLI gripper or arm is misspelled,
    # instead of deep in robot construction.
    if getattr(args, "gripper", None) is not None:
        GripperType.from_string_name(args.gripper)
    if getattr(args, "leader_gripper", None) is not None:
        GripperType.from_string_name(args.leader_gripper)
    if getattr(args, "arm_type", None) is not None:
        ArmType(args.arm_type)

    if args.mode == "teleop":
        ctrl = TeleopController(
            TeleopConfig(
                sim=args.sim,
                home=args.home,
                engage_thr=args.engage_thr,
                release_thr=args.release_thr,
                dwell=args.dwell,
                home_kp=args.home_kp,
                bilateral_kp=args.bilateral_kp,
                fine_grained_scale=args.fine_grained_scale,
                fine_grained_button=args.fine_grained_button,
                fine_recenter_speed=args.fine_recenter_speed,
                fine_recenter_kp=args.fine_recenter_kp,
                fine_recenter_max_following_error=args.fine_recenter_max_following_error,
                fine_recenter_tolerance=args.fine_recenter_tolerance,
                fine_recenter_dwell=args.fine_recenter_dwell,
                fine_recenter_timeout=args.fine_recenter_timeout,
                button_outcomes=teleop_button_outcomes(rig),
                rate=args.rate,
                ramp_speed=args.ramp_speed,
                home_speed=args.home_speed,
                gate_joints=args.gate_joints,
                arm_type=args.arm_type,
                leader_gripper=args.leader_gripper,
                follower_gripper=args.gripper,
            )
        )
    elif args.mode == "dagger":
        dagger_sec = rig.get("dagger", {}) or {}
        ctrl = DaggerController(
            DaggerConfig(
                sim=args.sim,
                home=args.home,
                mirror_kp=args.mirror_kp,
                feedback_kp=args.feedback_kp,
                fine_grained_scale=args.fine_grained_scale,
                fine_grained_button=args.fine_grained_button,
                fine_recenter_speed=args.fine_recenter_speed,
                fine_recenter_kp=args.fine_recenter_kp,
                fine_recenter_max_following_error=args.fine_recenter_max_following_error,
                fine_recenter_tolerance=args.fine_recenter_tolerance,
                fine_recenter_dwell=args.fine_recenter_dwell,
                fine_recenter_timeout=args.fine_recenter_timeout,
                home_kp=args.home_kp,
                home_speed=args.home_speed,
                rate=args.rate,
                max_joint_speed=args.max_joint_speed,
                button_map=dict(dagger_sec.get("buttons", {}))
                if dagger_sec.get("buttons")
                else DaggerConfig().button_map,
                arm_type=args.arm_type,
                leader_gripper=args.leader_gripper,
                follower_gripper=args.gripper,
            )
        )
    else:  # wrapper
        wcfg = WrapperConfig(
            sim=args.sim,
            arm_type=args.arm_type,
            gripper=args.gripper,
            rate=args.rate,
            max_joint_speed=args.max_joint_speed,
            control=args.control,
        )
        if robot_sec.get("channels"):  # {left: can_follower_l, right: can_follower_r}
            wcfg.channels = dict(robot_sec["channels"])
        ctrl = WrapperController(wcfg)

    RobotServer(ctrl, port=args.port, rate_hz=args.rate).serve()


if __name__ == "__main__":
    main()
