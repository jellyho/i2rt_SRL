"""Serve any LeRobot checkpoint through the YAM websocket policy protocol.

The policy type is read from the checkpoint's ``config.json`` and resolved by
LeRobot's policy factory.  Operators therefore never need to name a Python
policy class::

    yam-lerobot-serve --checkpoint /path/to/pretrained_model --device cuda

RTC is opt-in.  ``--num-inference-steps`` is intentionally solver-agnostic: it
selects converted-flow steps for RTC diffusion inference (or DDIM steps for
ordinary reduced-step diffusion inference) and flow integration steps for
policies such as flow-matching MultiTaskDiT, pi0.5, and SmolVLA.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .policies.lerobot_policy import LeRobotPolicy
from .serve import policy_metadata
from .websocket_server import WebsocketPolicyServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yam-lerobot-serve",
        description="Serve a LeRobot checkpoint over the YAM websocket policy protocol.",
    )
    parser.add_argument(
        "--checkpoint",
        "--pretrained-path",
        dest="checkpoint",
        type=Path,
        required=True,
        help="local pretrained_model directory containing config.json",
    )
    parser.add_argument("--device", default="cuda", help="torch device used by the policy")
    parser.add_argument("--rtc", action="store_true", help="enable real-time chunking inference")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "override iterative solver steps (converted-flow steps for RTC diffusion; "
            "DDIM steps for ordinary diffusion; flow integration steps for flow policies)"
        ),
    )
    parser.add_argument(
        "--rtc-guidance-weight",
        type=float,
        default=5.0,
        help="maximum RTC guidance weight (default: 5, as used in the RTC paper)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def make_policy(args: argparse.Namespace) -> LeRobotPolicy:
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.rtc_guidance_weight <= 0:
        raise ValueError("--rtc-guidance-weight must be positive")

    return LeRobotPolicy(
        pretrained_path=str(args.checkpoint),
        device=args.device,
        rtc=args.rtc,
        num_inference_steps=args.num_inference_steps,
        rtc_guidance_weight=args.rtc_guidance_weight,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    logging.info(
        "Loading LeRobot checkpoint=%s device=%s rtc=%s inference_steps=%s",
        args.checkpoint,
        args.device,
        args.rtc,
        args.num_inference_steps if args.num_inference_steps is not None else "checkpoint default",
    )
    policy = make_policy(args)
    metadata = policy_metadata(
        policy,
        policy_name="lerobot",
        config={
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "rtc": args.rtc,
            "num_inference_steps": args.num_inference_steps,
        },
    )
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
