"""Serve a LeRobot-trained policy (torch) through this server.

Install your policy's deps (torch, lerobot, ...) into THIS env only — it never
needs the robot's Python.

    python -m yam_policy.serve \
        --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
        --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
        --config device=cuda
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from ..base_policy import BasePolicy


class LeRobotPolicy(BasePolicy):
    def __init__(self, pretrained_path: str, device: str = "cuda") -> None:
        import draccus
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors, make_policy_config

        self._device = device
        self._pretrained_path = Path(pretrained_path)

        config = self._load_config(draccus, make_policy_config, self._pretrained_path, device)
        policy_cls = get_policy_class(config.type)
        self._policy = policy_cls.from_pretrained(self._pretrained_path, config=config)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            config, pretrained_path=str(self._pretrained_path)
        )
        self._policy.reset()

        self.action_horizon = int(getattr(config, "n_action_steps", 1))
        self.obs_spec = self._obs_spec(config)

    @staticmethod
    def _load_config(draccus, make_policy_config, pretrained_path: Path, device: str):  # noqa: ANN001
        """Load older LeRobot checkpoints under the current draccus-based config API."""
        raw = json.loads((pretrained_path / "config.json").read_text())
        policy_type = raw.pop("type")
        raw["device"] = device

        base_config = make_policy_config(policy_type)
        valid_fields = {field.name for field in dataclasses.fields(base_config)}
        filtered = {key: value for key, value in raw.items() if key in valid_fields}

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump(filtered, f)
            f.flush()
            with draccus.config_type("json"):
                return draccus.parse(base_config.__class__, f.name, args=[])

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

    def _build_batch(self, obs: Dict) -> Dict:
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

    def infer(self, obs: Dict) -> Dict:
        import torch

        with torch.no_grad():
            actions = self._policy.predict_action_chunk(self._build_batch(obs))
            actions = self._postprocessor(actions)

        return {"actions": actions.squeeze(0).detach().cpu().numpy().astype(np.float32)}

    def reset(self) -> None:
        self._policy.reset()
