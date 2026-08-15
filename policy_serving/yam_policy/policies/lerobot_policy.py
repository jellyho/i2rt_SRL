"""Serve a LeRobot-trained policy (torch) through this server.

Take a checkpoint LeRobot produced -- a local training output or a Hub repo id -- and deploy it
against the YAM stack with no conversion step:

    python -m yam_policy.serve \
        --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
        --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
        --config device=cuda

Install the policy's deps (torch, lerobot, ...) into THIS env only -- it never needs the robot's
Python.

**What makes it plug in.** The camera roles this stack records (`wrist_left`, `wrist_right`,
`agentview`) are exactly the suffixes of the dataset columns it writes
(`observation.images.<role>`), so a policy trained on data from this recorder already names its
image features after our cameras. :meth:`obs_spec` hands those names back to the client at the
handshake, and the client configures itself from them -- no per-checkpoint wiring. Point
``camera_map`` at the odd one out when a policy was trained elsewhere and its cameras are named
differently.

**Why the inference path is copied rather than invented.** LeRobot's own
``async_inference.policy_server`` is the reference for driving one of its policies, and two of
its steps are easy to get subtly wrong: the pre/post processors need an explicit device override
or the batch stays on the CPU while the model sits on the GPU, and the postprocessor unnormalises
one ``(B, action_dim)`` step at a time -- handing it a whole ``(B, chunk, action_dim)`` chunk does
not raise, it just unnormalises against the wrong axis. Both are mirrored here deliberately.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from ..base_policy import BasePolicy

if TYPE_CHECKING:  # torch and lerobot are imported lazily, so a robot-side import stays cheap
    from lerobot.configs.policies import PreTrainedConfig
    from torch import Tensor

logger = logging.getLogger(__name__)


class LeRobotPolicy(BasePolicy):
    """A LeRobot ``PreTrainedPolicy`` behind this repo's :class:`BasePolicy` contract.

    Args:
        pretrained_path: Local checkpoint dir or a Hub repo id.
        device: Torch device for both the model and the processors.
        camera_map: Optional ``{our_camera_role: policy_image_key}``, as a dict or a
            ``"role=key,role=key"`` string. Only needed when the policy's image features are not
            named after this stack's cameras.
        actions_per_chunk: Truncate each returned chunk to this many steps. Defaults to the whole
            chunk the policy predicts; the client adapts to whatever length arrives.
    """

    def __init__(
        self,
        pretrained_path: str,
        device: str = "cuda",
        camera_map: Optional[Dict[str, str] | str] = None,
        actions_per_chunk: Optional[int] = None,
    ) -> None:
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self._device = self._resolve_device(device)
        self._pretrained_path = str(pretrained_path)
        local = Path(self._pretrained_path).expanduser()
        if local.is_dir():
            self._pretrained_path = str(local)

        config = self._load_config(self._pretrained_path, self._device)
        policy_cls = get_policy_class(config.type)
        self._policy = policy_cls.from_pretrained(self._pretrained_path, config=config)
        self._policy.to(self._device)
        self._policy.eval()

        # The device override is what keeps the batch and the model on the same device; without
        # it the processors default to whatever the checkpoint was trained on.
        device_override = {"device": self._device}
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=self._pretrained_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        self._policy.reset()

        self._actions_per_chunk = int(actions_per_chunk) if actions_per_chunk else None
        self._image_features = dict(self._policy.config.image_features)
        self._input_features = dict(self._policy.config.input_features)
        self._camera_map = self._resolve_camera_map(camera_map, self._image_features)
        self.action_horizon = self._declared_horizon(self._policy.config, self._actions_per_chunk)
        self.obs_spec = self._build_obs_spec(self._camera_map, self._image_features)
        #: Merged into the handshake by serve.py, so deploy names what it connected to.
        self.policy_info = {
            "framework": "lerobot",
            "policy_type": str(config.type),
            "checkpoint": self._pretrained_path,
        }

        logger.info(
            "LeRobot %s loaded on %s | cameras %s | state features %s | horizon %d",
            config.type,
            self._device,
            self._camera_map,
            [k for k in self._input_features if k not in self._image_features],
            self.action_horizon,
        )
        self._warn_about_leaking_inputs(self._input_features)

    #: Columns this recorder writes that must not be policy inputs, and why.
    LEAKING_INPUTS = {
        "observation.leader": (
            "the teleop leader pose, which in a teleop dataset IS the action "
            "(measured at 2e-4 rad apart) -- a policy given this input can copy the answer, "
            "and at deploy the leader is either hanging free or mirroring the follower"
        ),
        "observation.control_mode": (
            "a provenance label, constant within a teleop dataset -- it teaches nothing and "
            "takes a different value during deployment"
        ),
    }

    # ------------------------------------------------------------------ setup
    @classmethod
    def _warn_about_leaking_inputs(cls, input_features: Dict) -> None:
        """Say so when the checkpoint reads something it should not have been trained on.

        LeRobot's trainer takes EVERY column in the dataset as an input feature, and this
        recorder writes more than the policy-relevant ones. Nothing downstream can detect the
        result: training loss is excellent precisely because the answer was an input, and the
        deployed policy then behaves badly for no visible reason. The one moment it is cheap to
        notice is when the checkpoint is loaded, so it is said here rather than left to be
        inferred from a bad rollout.
        """
        for key, why in cls.LEAKING_INPUTS.items():
            if key in input_features:
                logger.warning("this checkpoint reads %s -- %s. See policy_serving/README.md.", key, why)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("device=%s requested but CUDA is unavailable; falling back to cpu", device)
            return "cpu"
        return str(device)

    @classmethod
    def _load_config(cls, pretrained_path: str, device: str) -> "PreTrainedConfig":
        """The checkpoint's own config, with the device pinned.

        The plain ``PreTrainedConfig.from_pretrained`` path is what current LeRobot writes.
        Checkpoints from older versions carry fields the current dataclasses no longer accept, so
        those fall back to a filtered reparse rather than failing to deploy at all.
        """
        from lerobot.configs.policies import PreTrainedConfig

        try:
            config = PreTrainedConfig.from_pretrained(pretrained_path)
            config.device = device
            return config
        except Exception as exc:
            logger.warning("standard config load failed (%s); retrying as a legacy checkpoint", exc)
            return cls._load_legacy_config(pretrained_path, device)

    @staticmethod
    def _load_legacy_config(pretrained_path: str, device: str) -> "PreTrainedConfig":
        """Load an older LeRobot checkpoint by dropping fields the current config cannot take."""
        import draccus
        from lerobot.policies import make_policy_config

        config_path = Path(pretrained_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"no config.json at {pretrained_path}; a Hub repo id needs to resolve through "
                "PreTrainedConfig.from_pretrained, which failed above"
            )
        raw = json.loads(config_path.read_text())
        policy_type = raw.pop("type")
        raw["device"] = device

        base_config = make_policy_config(policy_type)
        valid = {field.name for field in dataclasses.fields(base_config)}
        dropped = sorted(set(raw) - valid)
        if dropped:
            logger.warning("ignoring config fields this LeRobot no longer defines: %s", dropped)
        filtered = {k: v for k, v in raw.items() if k in valid}

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as fh:
            json.dump(filtered, fh)
            fh.flush()
            with draccus.config_type("json"):
                return draccus.parse(base_config.__class__, fh.name, args=[])

    @staticmethod
    def _resolve_camera_map(camera_map, image_features: Dict) -> Dict[str, str]:  # noqa: ANN001
        """``{our camera role: policy image key}``.

        Defaults to the policy key's last segment, which is the role itself for any policy trained
        on data this recorder wrote (`observation.images.wrist_left` -> `wrist_left`).
        """
        if camera_map:
            if isinstance(camera_map, str):
                pairs = [p for p in camera_map.replace(";", ",").split(",") if p.strip()]
                explicit = {}
                for pair in pairs:
                    sep = "=" if "=" in pair else ":"
                    role, _, key = pair.partition(sep)
                    explicit[role.strip()] = key.strip()
            else:
                explicit = {str(k): str(v) for k, v in dict(camera_map).items()}
            unknown = sorted(set(explicit.values()) - set(image_features))
            if unknown:
                raise ValueError(
                    f"camera_map points at image keys the policy does not have: {unknown}. "
                    f"It reads {sorted(image_features)}"
                )
            return explicit
        return {key.rsplit(".", 1)[-1]: key for key in image_features}

    @staticmethod
    def _declared_horizon(config, actions_per_chunk: Optional[int]) -> int:  # noqa: ANN001
        """How many steps a chunk will carry.

        ``predict_action_chunk`` returns ``chunk_size`` steps, not the ``n_action_steps`` a
        closed-loop LeRobot rollout would consume before replanning -- this stack replans on its
        own schedule, so the whole chunk is what gets served.
        """
        horizon = int(getattr(config, "chunk_size", 0) or getattr(config, "n_action_steps", 1) or 1)
        return min(horizon, actions_per_chunk) if actions_per_chunk else horizon

    @staticmethod
    def _build_obs_spec(camera_map: Dict[str, str], image_features: Dict) -> Dict:
        """Metadata the client self-configures from: which key each camera goes to, and at what size."""
        spec: Dict = {"image_keys": dict(camera_map)}
        for key in camera_map.values():
            shape = tuple(int(v) for v in image_features[key].shape)
            if len(shape) == 3:
                spec["image_shape"] = [shape[1], shape[2]]  # (C, H, W) -> [H, W]
                break
        return spec

    # ------------------------------------------------------------------ inference
    def _prepare_image(self, value, shape: Tuple[int, int, int]) -> "Tensor":  # noqa: ANN001
        """HWC uint8 from the wire -> ``(1, C, H, W)`` float32 in [0, 1], at the policy's size."""
        import torch

        from yam_policy import image_tools

        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = arr.transpose(1, 2, 0)  # already CHW
        if arr.shape[:2] != (shape[1], shape[2]):
            arr = image_tools.resize_with_pad(arr, shape[1], shape[2])

        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        tensor = torch.from_numpy(chw)
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        else:
            tensor = tensor.to(torch.float32)
            if float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
        return tensor.contiguous().unsqueeze(0)

    @staticmethod
    def _prepare_vector(value, shape: Tuple[int, ...]) -> "Tensor":  # noqa: ANN001
        """A state-like feature -> ``(1, dim)`` float32, as LeRobot's own helper produces."""
        import torch

        # copy=True: arrays decoded off the wire are read-only, and torch.from_numpy on one is
        # undefined behaviour rather than an error.
        arr = np.array(value, dtype=np.float32, copy=True).reshape(-1)
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"expected {expected} values, got {arr.size}")
        return torch.from_numpy(arr).reshape(1, *shape)

    def _build_batch(self, obs: Dict) -> Dict:
        """Every input the policy declares, or a clear failure naming what is missing.

        Zero-filling a missing camera would drive the robot from a black frame and report nothing
        wrong, so this raises instead; the deploy runner surfaces it and holds the rollout.
        """
        canonical = self._canonicalize(obs)
        batch: Dict = {}
        missing = []

        for key, feature in self._input_features.items():
            shape = tuple(int(v) for v in feature.shape)
            value = canonical.get(key)
            if value is None:
                missing.append(key)
                continue
            try:
                if key in self._image_features:
                    batch[key] = self._prepare_image(value, shape)
                else:
                    batch[key] = self._prepare_vector(value, shape)
            except ValueError as exc:
                raise ValueError(f"observation {key!r} does not fit the policy: {exc}") from exc

        if missing:
            raise ValueError(
                f"the policy needs {missing}, which the client did not send. It sent "
                f"{sorted(canonical)}. Map a differently-named camera with camera_map."
            )

        batch["task"] = obs.get("task", obs.get("prompt", ""))
        return self._preprocessor(batch)

    @staticmethod
    def _canonical_key(key: str) -> str:
        """Accept openpi's slash-separated names alongside LeRobot's dotted ones."""
        if key.startswith("observation/images/"):
            return "observation.images." + key.rsplit("/", 1)[-1]
        if key.startswith("observation/"):
            return "observation." + key.split("/", 1)[1]
        return key

    @classmethod
    def _canonicalize(cls, obs: Dict) -> Dict:
        """Dotted keys, with a real dotted key always beating one translated from a slash name.

        The deploy client sends both conventions: `observation/state` is openpi's 14-dim joint
        vector, while `observation.state` is this stack's full 42-dim record. They collapse onto
        the same name, and picking by dict order would silently feed a LeRobot policy the wrong
        one -- same key, right dtype, wrong content.
        """
        translated = {}
        for key, value in obs.items():
            canonical = cls._canonical_key(key)
            if canonical == key or canonical not in obs:
                translated[canonical] = value
        return translated

    def infer(self, obs: Dict) -> Dict:
        import torch

        batch = self._build_batch(obs)
        with torch.no_grad():
            chunk = self._policy.predict_action_chunk(batch)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)  # -> (B, chunk, action_dim)
            if self._actions_per_chunk:
                chunk = chunk[:, : self._actions_per_chunk, :]

            # The postprocessor unnormalises one (B, action_dim) step at a time; see the module
            # docstring for why the whole chunk cannot be handed over at once.
            steps = [self._postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])]
            actions = torch.stack(steps, dim=1).squeeze(0)

        return {"actions": actions.detach().cpu().numpy().astype(np.float32)}

    def reset(self) -> None:
        self._policy.reset()
