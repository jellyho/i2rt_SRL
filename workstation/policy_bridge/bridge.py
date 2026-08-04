"""Policy bridge: robot (portal) <-> policy (websocket), on the workstation.

Topology 2 (deployment):

    robot server  ──portal──▶  THIS bridge  ──websocket+msgpack──▶  policy server
       (i2rt.serving)              │ build openpi obs, infer            (yam_policy)
                       ◀──portal───┘ set_policy_action(chunk step)

Each tick the bridge:
  1. reads robot state via ``RobotClient.get_observation()`` and cameras locally,
  2. builds an openpi-style obs dict (``observation/state``, ``observation/images/*``,
     ``prompt``),
  3. queries the policy via ``ActionChunkBroker(WebsocketClientPolicy)`` — only
     hitting the (possibly remote) server every ``action_horizon`` steps,
  4. splits the action into per-arm targets and sends them with
     ``RobotClient.set_policy_action({"left": ..., "right": ...})``.

The robot must run in **dagger** mode (policy drives the followers; a human can take
over with a handle button), or **wrapper** mode. The policy env is unconstrained —
it never imports i2rt.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np
from yam_policy import (
    ActionChunkBroker,
    AsyncActionChunkBroker,
    RTCActionChunkBroker,
    WebsocketClientPolicy,
    image_tools,
)

from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import ARM_DOF, ARMS, CONTROL_MODE, EEF_DIM, LEADER_DIM, RecorderConfig
from workstation.policy_bridge.config import BridgeConfig

logger = logging.getLogger(__name__)


class PolicyBridge:
    def __init__(self, cfg: BridgeConfig, recorder_cfg: RecorderConfig):
        from i2rt.serving.robot_client import RobotClient

        self.cfg = cfg
        self.cameras = CameraManager(recorder_cfg)
        self.robot = RobotClient(host=cfg.robot_host, port=cfg.robot_port)

        client = WebsocketClientPolicy(host=cfg.policy_host, port=cfg.policy_port)
        self._policy_client = client
        # Let the server's metadata override the obs/action spec so we don't have to
        # hand-match action_horizon / image keys / image size to the policy.
        meta = client.get_server_metadata() or {}
        self.action_horizon = int(meta.get("action_horizon", cfg.action_horizon))
        self.image_keys = meta.get("image_keys", cfg.image_keys)
        image_shape = meta.get("image_shape")
        if isinstance(image_shape, (list, tuple)) and len(image_shape) == 2:
            self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        else:
            image_size = int(meta.get("image_size", cfg.image_size))
            self.image_shape = (image_size, image_size)
        if meta:
            logger.info("policy server metadata: %s", meta)

        if cfg.rtc_enabled:
            if not meta.get("rtc_enabled"):
                client.close()
                raise ValueError(
                    "yam-data RTC was requested, but the connected policy server did not advertise "
                    "RTC. Start it with policy_serving/yam-serve --rtc."
                )
            self.policy = RTCActionChunkBroker(
                client,
                action_horizon=self.action_horizon,
                min_execute_steps=cfg.rtc_min_execute_steps,
                rate_hz=cfg.rate_hz,
                n_obs_steps=int(meta.get("n_obs_steps", 1)),
            )
        elif cfg.use_async:
            self.policy = AsyncActionChunkBroker(
                client,
                action_horizon=self.action_horizon,
                prefetch_at=cfg.prefetch_at(self.action_horizon),
            )
        else:
            self.policy = ActionChunkBroker(client, action_horizon=self.action_horizon)
        self._stop = False
        self._was_streaming = False

    # ---- obs assembly -------------------------------------------------------
    @staticmethod
    def _fuse(sides: List[Dict], fields: tuple, per_arm: int | None = None) -> np.ndarray | None:
        parts = []
        for side in sides:
            if not side or any(side.get(field) is None for field in fields):
                return None
            vec = np.concatenate([np.asarray(side[field], dtype=np.float32).reshape(-1) for field in fields])
            if per_arm is not None and vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _fit(value: np.ndarray | None, dim: int) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        if value is None:
            return out
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        n = min(dim, arr.size)
        if n:
            out[:n] = arr[:n]
        return out

    def _build_obs(self, robot_obs: Dict, images: Dict[str, np.ndarray]) -> Dict:
        sides = [robot_obs.get(a) for a in ARMS]
        pos = self._fuse(sides, ("pos",), ARM_DOF)
        if pos is None:
            return {}

        obs = {"observation/state": pos, "prompt": self.cfg.prompt}
        full_state = self._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        if full_state is not None:
            obs["observation.state"] = full_state
        obs["observation.leader"] = self._fit(self._fuse(sides, ("leader_pos",)), LEADER_DIM)
        obs["observation.eef"] = self._fit(self._fuse(sides, ("eef",)), EEF_DIM)
        obs["observation.control_mode"] = np.array([CONTROL_MODE["teleop"]], dtype=np.float32)

        height, width = self.image_shape
        for role, key in self.image_keys.items():
            if role in images:
                img = image_tools.resize_with_pad(images[role], height, width)
                obs[key] = image_tools.convert_to_uint8(img)
        return obs

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}

    # ---- run loop -----------------------------------------------------------
    def run(self) -> None:
        self.cameras.start()
        logger.info(
            "PolicyBridge up: robot=%s:%d policy=%s:%d horizon=%d image=%dx%d rate=%.0f Hz rtc=%s",
            self.cfg.robot_host,
            self.cfg.robot_port,
            self.cfg.policy_host,
            self.cfg.policy_port,
            self.action_horizon,
            self.image_shape[1],
            self.image_shape[0],
            self.cfg.rate_hz,
            self.cfg.rtc_enabled,
        )
        period = 1.0 / max(self.cfg.rate_hz, 1.0)
        try:
            try:
                self.robot.set_policy_running(True)
            except Exception:
                pass
            while not self._stop:
                t0 = time.monotonic()
                try:
                    robot_obs = self.robot.get_observation()
                    should_stream = bool(robot_obs.get("policy_running")) and not (
                        robot_obs.get("intervention")
                        or robot_obs.get("homing")
                        or robot_obs.get("estop")
                    )
                    if should_stream:
                        if not self._was_streaming:
                            self.policy.reset()
                        images = self.cameras.read()
                        obs = self._build_obs(robot_obs, images)
                        if obs:
                            action = self.policy.infer(obs)["actions"]
                            self.robot.set_policy_action(self._split(np.asarray(action, dtype=float)))
                            self._was_streaming = True
                        else:
                            self._was_streaming = False
                    else:
                        if self._was_streaming:
                            self.policy.reset()
                        self._was_streaming = False
                except Exception as e:
                    self._was_streaming = False
                    logger.warning("bridge tick failed: %s", e)
                remaining = period - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.robot.set_policy_running(False)
            except Exception:
                pass
            self.cameras.stop()
            close = getattr(self.policy, "close", None)
            if callable(close):
                close()
            self._policy_client.close()

    def stop(self) -> None:
        self._stop = True
