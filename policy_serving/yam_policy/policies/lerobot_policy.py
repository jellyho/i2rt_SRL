"""Serve a LeRobot-trained policy (torch) through this server.

Install your policy's deps (torch, lerobot, ...) into THIS env only — it never
needs the robot's Python.

    yam-lerobot-serve \
        --checkpoint outputs/train/my_act/checkpoints/last/pretrained_model \
        --device cuda
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..base_policy import BasePolicy


class LeRobotPolicy(BasePolicy):
    _RTC_REQUEST_KEY = "_yam_rtc"

    def __init__(
        self,
        pretrained_path: str,
        device: str = "cuda",
        rtc: bool = False,
        num_inference_steps: int | None = None,
        rtc_guidance_weight: float = 5.0,
    ) -> None:
        import draccus
        from lerobot.policies.factory import (
            get_policy_class,
            make_policy_config,
            make_pre_post_processors,
        )

        self._device = device
        self._pretrained_path = Path(pretrained_path)
        self._rtc_enabled = bool(rtc)

        config = self._load_config(
            draccus,
            make_policy_config,
            self._pretrained_path,
            device,
            rtc=self._rtc_enabled,
            num_inference_steps=num_inference_steps,
            rtc_guidance_weight=rtc_guidance_weight,
        )
        policy_cls = get_policy_class(config.type)
        self._policy = policy_cls.from_pretrained(self._pretrained_path, config=config)
        self._policy = self._policy.to(device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            config, pretrained_path=str(self._pretrained_path)
        )
        self._relative_step = None
        self._normalizer_step = None
        if self._rtc_enabled:
            from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep

            self._relative_step = next(
                (
                    step
                    for step in self._preprocessor.steps
                    if isinstance(step, RelativeActionsProcessorStep) and step.enabled
                ),
                None,
            )
            self._normalizer_step = next(
                (step for step in self._preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
                None,
            )
        self._policy.reset()

        rtc_supported = self._supports_rtc(self._policy)
        if self._rtc_enabled and not rtc_supported:
            raise ValueError(
                f"LeRobot policy type {config.type!r} does not support RTC; start the server without --rtc"
            )

        self.action_horizon = int(getattr(config, "n_action_steps", 1))
        self.obs_spec = self._obs_spec(config)
        self.server_metadata = {
            "policy_type": config.type,
            "rtc_supported": rtc_supported,
            "rtc_enabled": self._rtc_enabled,
            "n_obs_steps": int(getattr(config, "n_obs_steps", 1)),
            "solver": self._solver_name(config, rtc=self._rtc_enabled),
            "num_inference_steps": self._solver_steps(config),
        }

    @staticmethod
    def _load_config(
        draccus: Any,
        make_policy_config: Any,
        pretrained_path: Path,
        device: str,
        *,
        rtc: bool = False,
        num_inference_steps: int | None = None,
        rtc_guidance_weight: float = 5.0,
    ) -> Any:
        """Load older LeRobot checkpoints under the current draccus-based config API."""
        raw = json.loads((pretrained_path / "config.json").read_text())
        policy_type = raw.pop("type")
        raw["device"] = device

        base_config = make_policy_config(policy_type)
        valid_fields = {field.name for field in dataclasses.fields(base_config)}
        filtered = {key: value for key, value in raw.items() if key in valid_fields}

        if num_inference_steps is not None:
            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be positive")
            if policy_type == "multi_task_dit":
                objective = filtered.get("objective", getattr(base_config, "objective", None))
                if objective == "diffusion":
                    # Non-RTC reduced-step inference keeps the existing deterministic
                    # DDIM behavior. RTC instead uses the checkpoint's original
                    # scheduler coefficients through its diffusion-to-flow adapter.
                    if not rtc:
                        filtered["noise_scheduler_type"] = "DDIM"
                    filtered["num_inference_steps"] = num_inference_steps
                elif objective == "flow_matching":
                    filtered["num_integration_steps"] = num_inference_steps
                else:
                    raise ValueError(f"Unsupported MultiTaskDiT objective {objective!r}")
            elif "num_inference_steps" in valid_fields:
                # pi0/pi0.5 use this for flow integration steps.
                filtered["num_inference_steps"] = num_inference_steps
            elif "num_steps" in valid_fields:
                # SmolVLA names the same solver control ``num_steps``.
                filtered["num_steps"] = num_inference_steps
            else:
                raise ValueError(f"LeRobot policy type {policy_type!r} has no iterative inference-step setting")

        if rtc:
            if "rtc_config" not in valid_fields:
                raise ValueError(f"LeRobot policy type {policy_type!r} has no RTC configuration")
            filtered["rtc_config"] = {
                "enabled": True,
                "max_guidance_weight": float(rtc_guidance_weight),
            }

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump(filtered, f)
            f.flush()
            with draccus.config_type("json"):
                return draccus.parse(base_config.__class__, f.name, args=[])

    @staticmethod
    def _supports_rtc(policy) -> bool:  # noqa: ANN001
        supports_rtc = getattr(policy, "supports_rtc", None)
        return bool(callable(supports_rtc) and supports_rtc())

    @staticmethod
    def _solver_name(config, *, rtc: bool = False) -> str:  # noqa: ANN001
        if getattr(config, "objective", None) == "diffusion":
            if rtc and getattr(config, "type", None) == "multi_task_dit":
                return "diffusion_to_flow"
            return str(getattr(config, "noise_scheduler_type", "diffusion")).lower()
        if getattr(config, "objective", None) == "flow_matching" or config.type in {
            "pi0",
            "pi05",
            "smolvla",
        }:
            return "flow_matching"
        return "direct"

    @staticmethod
    def _solver_steps(config) -> int | None:  # noqa: ANN001
        for key in ("num_inference_steps", "num_steps", "num_integration_steps"):
            value = getattr(config, key, None)
            if value is not None:
                return int(value)
        return None

    @staticmethod
    def _obs_spec(config) -> Dict:  # noqa: ANN001
        image_keys = {}
        image_shape = None
        for key, feature in config.image_features.items():
            role = key.rsplit(".", 1)[-1]
            image_keys[role] = key
            if image_shape is None and len(feature.shape) == 3:
                image_shape = [int(feature.shape[1]), int(feature.shape[2])]

        spec: Dict = {"image_keys": image_keys}
        if image_shape is not None:
            spec["image_shape"] = image_shape
        return spec

    @staticmethod
    def _canonical_key(key: str) -> str:
        if key.startswith("observation/images/"):
            return "observation.images." + key.rsplit("/", 1)[-1]
        if key == "observation/state":
            return "observation.state"
        return key

    @staticmethod
    def _image_to_chw_float(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        from yam_policy import image_tools

        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 3:
            if arr.shape[1:] == shape:
                chw = arr.astype(np.float32)
                return chw / 255.0 if float(np.max(chw)) > 1.0 else chw
            arr = arr.transpose(1, 2, 0)

        arr = image_tools.resize_with_pad(arr, shape[0], shape[1])
        chw = arr.transpose(2, 0, 1).astype(np.float32)
        return chw / 255.0

    @staticmethod
    def _fit_feature(value, shape: tuple[int, ...]) -> np.ndarray:  # noqa: ANN001
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        size = int(np.prod(shape))
        out = np.zeros(size, dtype=np.float32)
        n = min(size, arr.size)
        if n:
            out[:n] = arr[:n]
        return out.reshape(shape)

    def _build_single_batch(self, obs: Dict) -> Dict:
        import torch

        batch = {}
        canonical = {self._canonical_key(k): v for k, v in obs.items()}
        for key, feature in self._policy.config.input_features.items():
            shape = tuple(int(v) for v in feature.shape)
            if key in self._policy.config.image_features:
                value = canonical.get(key)
                if value is None:
                    batch[key] = torch.zeros(shape, dtype=torch.float32)
                else:
                    batch[key] = torch.as_tensor(
                        self._image_to_chw_float(value, (shape[1], shape[2])), dtype=torch.float32
                    )
            else:
                batch[key] = torch.as_tensor(
                    self._fit_feature(canonical.get(key, np.zeros(shape, dtype=np.float32)), shape),
                    dtype=torch.float32,
                )

        batch["task"] = obs.get("task", obs.get("prompt", ""))

        return self._preprocessor(batch)

    def _build_batch(self, obs: Dict, rtc_request: Dict[str, Any] | None = None) -> Dict:
        """Build a policy batch, preserving consecutive history for temporal policies."""
        import torch

        n_obs_steps = int(getattr(self._policy.config, "n_obs_steps", 1))
        raw_history = list((rtc_request or {}).get("observation_history") or [obs])
        raw_history = [frame for frame in raw_history if isinstance(frame, dict)] or [obs]
        raw_history = raw_history[-n_obs_steps:]
        while len(raw_history) < n_obs_steps:
            raw_history.insert(0, raw_history[0])

        processed = [self._build_single_batch(frame) for frame in raw_history]
        if n_obs_steps == 1:
            return processed[-1]

        # MultiTaskDiT consumes temporal dimensions on its declared observation
        # features, while language tokens remain one per batch item.
        latest = dict(processed[-1])
        for key in self._policy.config.input_features:
            values = [frame.get(key) for frame in processed]
            if all(isinstance(value, torch.Tensor) for value in values):
                latest[key] = torch.stack(values, dim=1)
        return latest

    def infer(self, obs: Dict) -> Dict:
        import torch

        rtc_request = obs.get(self._RTC_REQUEST_KEY)
        batch = self._build_batch(obs, rtc_request if isinstance(rtc_request, dict) else None)
        kwargs: Dict[str, Any] = {}
        if self._rtc_enabled and isinstance(rtc_request, dict):
            prev_actions = rtc_request.get("prev_chunk_left_over")
            if prev_actions is not None:
                param = next(self._policy.parameters())
                # msgpack-numpy may decode a read-only view. Copy before
                # constructing a tensor so autograd never aliases that buffer.
                prev_actions = torch.tensor(np.asarray(prev_actions).copy(), dtype=param.dtype, device=param.device)

                # Relative-action policies generated the old tail around the old
                # state. Re-anchor the executable absolute tail around the newest
                # state, then normalize it back into model space before guidance.
                processed_actions = rtc_request.get("prev_processed_actions")
                current_state = self._relative_step.get_cached_state() if self._relative_step is not None else None
                if processed_actions is not None and current_state is not None:
                    from lerobot.policies.rtc import reanchor_relative_rtc_prefix

                    prev_actions = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=torch.tensor(np.asarray(processed_actions).copy()),
                        current_state=current_state,
                        relative_step=self._relative_step,
                        normalizer_step=self._normalizer_step,
                        policy_device=param.device,
                    ).to(dtype=param.dtype)
            kwargs = {
                "inference_delay": int(rtc_request.get("inference_delay", 0)),
                "prev_chunk_left_over": prev_actions,
                "execution_horizon": rtc_request.get("execution_horizon"),
            }

        # RTC policies selectively re-enable gradients inside their denoising
        # loop for the guidance VJP.  The observation encoder remains no-grad.
        with torch.no_grad():
            model_actions = self._policy.predict_action_chunk(batch, **kwargs)
            actions = self._postprocessor(model_actions)

        return {
            "actions": actions.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "model_actions": model_actions.squeeze(0).detach().cpu().numpy().astype(np.float32),
        }

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
