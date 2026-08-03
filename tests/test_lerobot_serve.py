"""Checkpoint-driven yam-lerobot-serve CLI tests."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from yam_policy import lerobot_serve
from yam_policy.policies.lerobot_policy import LeRobotPolicy


def test_lerobot_serve_parser_does_not_require_policy_class():
    args = lerobot_serve.build_parser().parse_args(
        ["--checkpoint", "/tmp/checkpoint", "--rtc", "--num-inference-steps", "20"]
    )
    assert str(args.checkpoint) == "/tmp/checkpoint"
    assert args.rtc is True
    assert args.num_inference_steps == 20


def test_lerobot_serve_passes_normalized_options(monkeypatch):
    captured = {}

    class FakePolicy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(lerobot_serve, "LeRobotPolicy", FakePolicy)
    args = SimpleNamespace(
        checkpoint="/tmp/mtd",
        device="cuda",
        rtc=True,
        num_inference_steps=20,
        rtc_guidance_weight=5.0,
    )
    lerobot_serve.make_policy(args)

    assert captured == {
        "pretrained_path": "/tmp/mtd",
        "device": "cuda",
        "rtc": True,
        "num_inference_steps": 20,
        "rtc_guidance_weight": 5.0,
    }


def test_multitask_dit_rtc_preserves_checkpoint_diffusion_scheduler(tmp_path):
    @dataclass
    class FakeConfig:
        type: str = "multi_task_dit"
        device: str = "cpu"
        objective: str = "diffusion"
        noise_scheduler_type: str = "DDPM"
        num_inference_steps: int | None = None
        rtc_config: dict | None = None

    class FakeDraccus:
        config_type = staticmethod(lambda _format: contextlib.nullcontext())

        @staticmethod
        def parse(_config_type, path, args):
            assert args == []
            return json.loads(Path(path).read_text())

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "multi_task_dit",
                "objective": "diffusion",
                "noise_scheduler_type": "DDPM",
            }
        )
    )

    loaded = LeRobotPolicy._load_config(
        FakeDraccus,
        lambda _policy_type: FakeConfig(),
        tmp_path,
        "cpu",
        rtc=True,
        num_inference_steps=20,
    )

    assert loaded["noise_scheduler_type"] == "DDPM"
    assert loaded["num_inference_steps"] == 20
    assert loaded["rtc_config"]["enabled"] is True
    assert LeRobotPolicy._solver_name(FakeConfig(), rtc=True) == "diffusion_to_flow"
