"""Controllable policy streamer used by the DAgger deployment UI.

Unlike the headless bridge, this runner does not own cameras and does not decide
whether policy rollout is active. The robot-side DAgger controller is the source
of truth; this runner only sends policy actions while the robot snapshot reports
``policy_running`` and not intervention/homing/e-stop.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
from workstation.policy_bridge.config import BridgeConfig

logger = logging.getLogger(__name__)


class DeploymentPolicyRunner:
    def __init__(self, cfg: BridgeConfig, recorder_cfg: RecorderConfig, images_fn: Callable[[], Dict[str, np.ndarray]]):
        self.cfg = cfg
        self.recorder_cfg = recorder_cfg
        self.images_fn = images_fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._robot = None
        self._policy = None
        self._policy_client = None
        self._image_shape = (cfg.image_size, cfg.image_size)
        self._was_streaming = False
        self._lock = threading.Lock()
        self._status = {
            "robot_connected": False,
            "policy_connected": False,
            "streaming": False,
            "last_error": "",
            "action_horizon": 0,  # filled in from the first chunk the policy returns
            "image_size": cfg.image_size,
            "image_shape": self._image_shape,
        }

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._policy_client is not None:
            try:
                self._policy_client.close()
            except Exception:
                pass

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)

    def _connect_robot(self) -> None:
        if self.recorder_cfg.mock or self._robot is not None:
            self._set(robot_connected=True)
            return
        from i2rt.serving.robot_client import RobotClient

        self._robot = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0)
        self._set(robot_connected=True)

    def _policy_port_open(self) -> bool:
        try:
            with socket.create_connection((self.cfg.policy_host, self.cfg.policy_port), timeout=0.5):
                return True
        except OSError:
            return False

    def _connect_policy(self) -> None:
        if self.recorder_cfg.mock or self._policy is not None:
            self._set(policy_connected=True)
            return
        if not self._policy_port_open():
            raise ConnectionError(f"policy server offline at {self.cfg.policy_host}:{self.cfg.policy_port}")
        from yam_policy import ActionChunkBroker, AsyncActionChunkBroker, WebsocketClientPolicy

        client = WebsocketClientPolicy(host=self.cfg.policy_host, port=self.cfg.policy_port)
        meta = client.get_server_metadata() or {}
        self._image_shape = self._image_shape_from_meta(meta)
        image_keys = meta.get("image_keys", self.cfg.image_keys)
        # No action_horizon here on purpose: the broker reads the chunk size off each
        # response, so a checkpoint's horizon can never disagree with a client setting.
        broker_cls = AsyncActionChunkBroker if self.cfg.use_async else ActionChunkBroker
        self._policy_client = client
        self._policy = broker_cls(client)
        self.cfg.image_keys = image_keys
        self._set(
            policy_connected=True,
            image_size=self._image_shape[0],
            image_shape=self._image_shape,
        )
        logger.info("deploy policy metadata: %s", meta)

    def _image_shape_from_meta(self, meta: Dict) -> tuple[int, int]:
        image_shape = meta.get("image_shape")
        if isinstance(image_shape, (list, tuple)) and len(image_shape) == 2:
            return (int(image_shape[0]), int(image_shape[1]))
        image_size = int(meta.get("image_size", self.cfg.image_size))
        return (image_size, image_size)

    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.rate_hz, 1.0)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self.recorder_cfg.mock:
                    self._set(robot_connected=True, policy_connected=True, streaming=False, last_error="")
                else:
                    self._connect_robot()
                    obs = self._robot.get_observation()
                    should_stream = bool(obs.get("policy_running")) and not (
                        obs.get("intervention") or obs.get("homing") or obs.get("estop")
                    )
                    if should_stream:
                        self._connect_policy()
                        if not self._was_streaming:
                            self._reset_policy_chunk()
                        policy_obs = self._build_obs(obs, self.images_fn())
                        if policy_obs:
                            action = self._policy.infer(policy_obs)["actions"]
                            self._robot.set_policy_action(self._split(np.asarray(action, dtype=float)))
                            self._set(
                                streaming=True,
                                last_error="",
                                action_horizon=getattr(self._policy, "action_horizon", 0),
                            )
                            self._was_streaming = True
                        else:
                            self._set(streaming=False)
                            self._was_streaming = False
                    else:
                        if self._was_streaming:
                            self._reset_policy_chunk()
                        self._set(streaming=False, last_error="")
                        self._was_streaming = False
            except Exception as e:
                self._set(streaming=False, last_error=f"{type(e).__name__}: {e}")
                self._was_streaming = False
                msg = str(e).lower()
                if "policy" in msg:
                    self._policy = None
                    self._policy_client = None
                    self._set(policy_connected=False)
                else:
                    self._robot = None
                    self._set(robot_connected=False)
                logger.warning("deploy policy tick failed: %s", e)
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _reset_policy_chunk(self) -> None:
        if self._policy is None:
            return
        try:
            self._policy.reset()
        except Exception as e:
            logger.warning("policy reset failed: %s", e)

    @staticmethod
    def _fuse(sides: list[Dict], fields: tuple, per_arm: int | None = None) -> np.ndarray | None:
        parts = []
        for side in sides:
            if not side or any(side.get(field) is None for field in fields):
                return None
            vec = np.concatenate([np.asarray(side[field], dtype=np.float32).reshape(-1) for field in fields])
            if per_arm is not None and vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)


    def _build_obs(self, robot_obs: Dict, images: Dict[str, np.ndarray]) -> Dict:
        from yam_policy import image_tools

        sides = [robot_obs.get(arm) for arm in ARMS]
        # `observation/state` must be the SAME vector the policy was trained on. Training
        # repacks the dataset's `observation.state` (pos+vel+eff per arm, 42) into this
        # key, so sending only the 14 joint positions would keep the key valid, pass every
        # check, and quietly normalize against the wrong statistics. Send all 42 and let
        # the policy's input transform take the slice it wants.
        state = self._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        if state is None:
            return {}

        obs = {"observation/state": state, "prompt": self.cfg.prompt}

        height, width = self._image_shape
        for role, key in self.cfg.image_keys.items():
            if role in images:
                img = image_tools.resize_with_pad(images[role], height, width)
                obs[key] = image_tools.convert_to_uint8(img)
        return obs

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}
