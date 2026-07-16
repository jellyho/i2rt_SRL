"""Run the policy bridge (workstation).

    python -m workstation.policy_bridge \
        --robot-host 192.168.1.10 --policy-host 192.168.1.20 \
        --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn> \
        --prompt "pick up the cube"

The robot side must run ``i2rt.serving.run_robot_server dagger`` and the policy
side ``python -m yam_policy.serve ...`` (or a real openpi server).
"""

from __future__ import annotations

import argparse
import logging

from i2rt.serving.rig_config import Resolver, apply_camera_serials, load_rig
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras
from workstation.policy_bridge.bridge import BridgeConfig, PolicyBridge


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="YAM policy bridge (portal robot <-> websocket policy)")
    p.add_argument("--config", default=None, help="config.yaml (robot/policy/cameras)")
    p.add_argument("--robot-host", default="127.0.0.1")
    p.add_argument("--robot-port", type=int, default=11331)
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--contract", default="yam_bimanual_v1")
    p.add_argument("--execution-horizon", "--action-horizon", dest="execution_horizon", type=int, default=16)
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--camera-max-age", type=float, default=0.25)
    p.add_argument("--inference-timeout", type=float, default=2.0)
    p.add_argument("--prompt", default="do the task")
    p.add_argument("--no-async", action="store_true", help="disable action-chunk prefetch (query synchronously)")
    p.add_argument("--arm", action="store_true", help="operator confirmation: permit real policy motion")
    p.add_argument(
        "--allow-legacy-metadata", action="store_true", help="allow dummy/legacy servers without full metadata"
    )
    p.add_argument("--serials", default="", help="comma-separated RealSense serials: wrist_left,wrist_right,agentview")
    p.add_argument("--mock", action="store_true", help="synthetic cameras (no RealSense)")
    args = p.parse_args()

    rig = load_rig(args.config)
    rob = Resolver(args, p, rig.get("robot", {}))
    pol = Resolver(args, p, rig.get("policy", {}))

    cams = apply_camera_serials(default_cameras(), rig)
    if args.serials:
        for cam, serial in zip(cams, [s.strip() for s in args.serials.split(",")], strict=False):
            cam.serial = serial
    recorder_cfg = RecorderConfig(cameras=cams, mock=args.mock)

    cfg = BridgeConfig(
        robot_host=rob.get("robot_host", key="host"),
        robot_port=int(rob.get("robot_port", key="port")),
        policy_host=pol.get("policy_host", key="host"),
        policy_port=int(pol.get("policy_port", key="port")),
        contract=str(pol.get("contract")),
        execution_horizon=int(pol.get("execution_horizon")),
        rate_hz=float(pol.get("rate", key="rate_hz")),
        image_size=int(pol.get("image_size")),
        prompt=str(pol.get("prompt")),
        use_async=not args.no_async,
        require_operator_arm=bool(pol.section.get("require_operator_arm", True)),
        arm_on_start=args.arm,
        allow_legacy_metadata=args.allow_legacy_metadata,
        camera_max_age_s=float(pol.get("camera_max_age", key="camera_max_age_s")),
        inference_timeout_s=float(pol.get("inference_timeout", key="inference_timeout_s")),
    )
    PolicyBridge(cfg, recorder_cfg).run()


if __name__ == "__main__":
    main()
