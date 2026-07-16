"""Serve an OpenPI YAM checkpoint through the I2RT policy server.

Usually you'd just run openpi's own ``scripts/serve_policy.py`` — but this shows
how to host an openpi policy behind *our* server so every policy you deploy uses
one launcher (:mod:`yam_policy.serve`). Install ``openpi`` into THIS env.

    uv pip install -e /path/to/openpi   # heavy: JAX/torch, GPU

Then:
    python -m yam_policy.serve \
        --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
        --config config_name=pi0_fast_droid \
        --config checkpoint_dir=/abs/path/to/checkpoint
"""

from __future__ import annotations

import pathlib
from typing import Dict

from ..base_policy import BasePolicy
from ..yam_contract import ACTION_DIM, CONTRACT, DEFAULT_IMAGE_KEYS, to_openpi_input, validate_action_chunk


class OpenPiPolicy(BasePolicy):
    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        execution_horizon: int = 16,
        default_prompt: str | None = None,
        contract_override: str | None = None,
    ) -> None:
        # Imported lazily so this module imports fine in envs without openpi.
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        cfg = openpi_config.get_config(config_name)
        metadata = dict(cfg.policy_metadata or {})
        expected_contract = contract_override or CONTRACT
        if metadata.get("contract") != expected_contract:
            raise ValueError(
                f"OpenPI config {config_name!r} declares contract {metadata.get('contract')!r}; "
                f"expected {expected_contract!r}"
            )
        if int(metadata.get("action_dim", -1)) != ACTION_DIM:
            raise ValueError(f"OpenPI config {config_name!r} must declare action_dim={ACTION_DIM}")
        model_horizon = int(cfg.model.action_horizon)
        if int(metadata.get("model_action_horizon", -1)) != model_horizon:
            raise ValueError("OpenPI policy metadata model horizon does not match the model config")
        execution_horizon = int(execution_horizon)
        if not 0 < execution_horizon <= model_horizon:
            raise ValueError(f"execution_horizon must be in [1, {model_horizon}], got {execution_horizon}")

        checkpoint = pathlib.Path(checkpoint_dir).expanduser().resolve()
        self._validate_checkpoint(checkpoint, cfg)
        self._policy = policy_config.create_trained_policy(
            cfg,
            checkpoint,
            default_prompt=default_prompt,
        )
        self.config_name = config_name
        self.checkpoint_dir = str(checkpoint)
        self.action_horizon = execution_horizon
        self.obs_spec = {
            **metadata,
            "contract": expected_contract,
            "action_dim": ACTION_DIM,
            "action_horizon": execution_horizon,
            "model_action_horizon": model_horizon,
            "image_keys": dict(metadata.get("image_keys", DEFAULT_IMAGE_KEYS)),
            "checkpoint_dir": self.checkpoint_dir,
            "config_name": config_name,
        }

    @staticmethod
    def _validate_checkpoint(checkpoint: pathlib.Path, cfg) -> None:  # noqa: ANN001
        params = checkpoint / "params"
        if not params.is_dir() or not any(params.iterdir()):
            raise FileNotFoundError(f"Missing or empty OpenPI params directory: {params}")
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        if not data_cfg.asset_id:
            raise ValueError("OpenPI data config must declare an asset_id")
        stats = checkpoint / "assets" / data_cfg.asset_id / "norm_stats.json"
        if not stats.is_file():
            raise FileNotFoundError(f"Missing OpenPI normalization statistics: {stats}")

    def infer(self, obs: Dict) -> Dict:
        image_keys = self.obs_spec["image_keys"]
        image_size = int(self.obs_spec["image_size"])
        openpi_obs = to_openpi_input(obs, image_keys=image_keys, image_shape=(image_size, image_size))
        result = self._policy.infer(openpi_obs)
        return validate_action_chunk(result, execution_horizon=self.action_horizon)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
