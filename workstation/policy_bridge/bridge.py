"""Fail-closed robot-to-policy bridge for the ``yam_bimanual_v1`` contract."""

from __future__ import annotations

import logging
import time
from enum import Enum

import numpy as np
from yam_policy import (
    ActionChunkBroker,
    AsyncActionChunkBroker,
    WebsocketClientPolicy,
    validate_action_step,
    validate_server_metadata,
)

from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.observation import build_observation

logger = logging.getLogger(__name__)


class RolloutState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    READY = "READY"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    INTERVENING = "INTERVENING"
    STOPPED = "STOPPED"
    ESTOP = "ESTOP"


class PolicyBridge:
    def __init__(self, cfg: BridgeConfig, recorder_cfg: RecorderConfig):
        from i2rt.serving.robot_client import RobotClient

        self.cfg = cfg
        self.simulated = recorder_cfg.mock
        self.cameras = CameraManager(recorder_cfg)
        self.robot = RobotClient(host=cfg.robot_host, port=cfg.robot_port, timeout=2.0)
        client = WebsocketClientPolicy(
            host=cfg.policy_host,
            port=cfg.policy_port,
            timeout=cfg.inference_timeout_s,
        )
        metadata = client.get_server_metadata() or {}
        self.spec = validate_server_metadata(
            metadata,
            configured_contract=cfg.contract,
            configured_execution_horizon=cfg.execution_horizon,
            configured_control_hz=cfg.rate_hz,
            allow_legacy=cfg.allow_legacy_metadata,
        )
        camera_roles = set(self.cameras.image_keys)
        missing = sorted(set(self.spec.image_keys) - camera_roles)
        if missing:
            raise ValueError(f"Configured cameras are missing policy roles: {missing}")
        self.execution_horizon = self.spec.execution_horizon
        self.model_action_horizon = self.spec.model_action_horizon
        self.image_keys = self.spec.image_keys
        self.image_shape = self.spec.image_shape
        broker_cls = AsyncActionChunkBroker if cfg.use_async else ActionChunkBroker
        self.policy = broker_cls(client, action_horizon=self.execution_horizon)
        self._stop = False
        self._armed = not cfg.require_operator_arm or cfg.arm_on_start
        self.state = RolloutState.DISCONNECTED
        self._intervening = False
        logger.info("validated policy server metadata: %s", metadata)

    def arm(self) -> None:
        if self.state in {RolloutState.ESTOP, RolloutState.STOPPED}:
            raise RuntimeError(f"Cannot arm bridge in state {self.state.value}; restart after clearing the fault")
        self._armed = True
        self.state = RolloutState.ARMED
        logger.warning("Rollout ARMED by operator; policy motion will begin when all health gates pass")

    def _reset_policy_chunk(self) -> None:
        try:
            self.policy.reset()
        except Exception as exc:
            logger.warning("policy chunk reset failed: %s", exc)

    def _set_policy_running(self, flag: bool) -> None:
        try:
            self.robot.set_policy_running(flag)
        except Exception as exc:
            logger.warning("failed to set robot policy_running=%s: %s", flag, exc)

    def _fail_closed(self, reason: str, *, estop: bool = False) -> None:
        self._reset_policy_chunk()
        self._armed = False
        self.state = RolloutState.ESTOP if estop else RolloutState.STOPPED
        self._set_policy_running(False)
        logger.error("rollout fail-closed (%s): robot receives no further policy commands", reason)

    def _build_obs(self, robot_obs: dict, images: dict[str, np.ndarray]) -> dict:
        return build_observation(
            robot_obs,
            images,
            prompt=self.cfg.prompt,
            image_keys=self.image_keys,
            image_shape=self.image_shape,
        )

    def _split(self, action: object) -> dict[str, np.ndarray]:
        action = validate_action_step(action, action_dim=self.spec.action_dim)
        if self.cfg.action_limits:
            action = action.copy()
            for index, limits in enumerate(self.cfg.action_limits[: action.size]):
                action[index] = np.clip(action[index], float(limits[0]), float(limits[1]))
        return {arm: action[index * ARM_DOF : (index + 1) * ARM_DOF].astype(float) for index, arm in enumerate(ARMS)}

    def _handle_intervention(self, active: bool) -> bool:
        if active and not self._intervening:
            self._reset_policy_chunk()
            self.state = RolloutState.INTERVENING
            logger.info("human intervention started; discarded current and prefetched policy chunks")
        elif not active and self._intervening:
            self._reset_policy_chunk()
            self.state = RolloutState.ARMED
            logger.info("human intervention ended; next action will use the post-intervention state")
        self._intervening = active
        return active

    def run(self) -> None:
        self.cameras.start()
        logger.info(
            "PolicyBridge connected: contract=%s robot=%s:%d policy=%s:%d model_horizon=%d "
            "execution_horizon=%d image=%dx%d rate=%.0fHz prompt=%r mode=%s armed=%s",
            self.spec.contract,
            self.cfg.robot_host,
            self.cfg.robot_port,
            self.cfg.policy_host,
            self.cfg.policy_port,
            self.model_action_horizon,
            self.execution_horizon,
            self.image_shape[1],
            self.image_shape[0],
            self.spec.control_hz,
            self.cfg.prompt,
            "simulated" if self.simulated else "real",
            self._armed,
        )
        if self._armed:
            self.arm()
        period = 1.0 / self.spec.control_hz
        try:
            while not self._stop:
                started = time.monotonic()
                try:
                    robot_obs = self.robot.get_observation()
                    if bool(robot_obs.get("estop")):
                        self._fail_closed("robot e-stop is active", estop=True)
                        break
                    images = self.cameras.read()
                    if self._handle_intervention(bool(robot_obs.get("intervention"))):
                        continue
                    camera_healthy = self.cameras.healthy
                    stale = {
                        role: age for role, age in self.cameras.frame_ages.items() if age > self.cfg.camera_max_age_s
                    }
                    if not self._armed:
                        self.state = RolloutState.READY if camera_healthy and not stale else RolloutState.DISCONNECTED
                        continue
                    if not camera_healthy:
                        self._fail_closed("one or more required cameras are missing or stale")
                        break
                    if stale:
                        self._fail_closed(
                            f"required camera frames exceeded max age {self.cfg.camera_max_age_s}s: {stale}"
                        )
                        break
                    if self.state == RolloutState.ARMED:
                        self._set_policy_running(True)
                        self.state = RolloutState.RUNNING
                    response = self.policy.infer(self._build_obs(robot_obs, images))
                    action = validate_action_step(response.get("actions"), action_dim=self.spec.action_dim)
                    self.robot.set_policy_action(self._split(action))
                    timing = response.get("broker_timing", {})
                    chunk_step = int(timing.get("chunk_step", 0))
                    log = logger.info if chunk_step == 0 else logger.debug
                    log(
                        "policy action server_infer_ms=%s model_infer_ms=%s chunk_age_ms=%.1f chunk_step=%d",
                        response.get("server_timing", {}).get("infer_ms", "cached"),
                        response.get("policy_timing", {}).get("infer_ms", "cached"),
                        float(timing.get("chunk_age_ms", 0.0)),
                        chunk_step,
                    )
                except Exception as exc:
                    self._fail_closed(f"{type(exc).__name__}: {exc}")
                    break
                finally:
                    remaining = period - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            logger.info("operator stopped bridge")
        finally:
            self.state = RolloutState.STOPPED
            self._set_policy_running(False)
            self._reset_policy_chunk()
            if hasattr(self.policy, "close"):
                self.policy.close()
            self.cameras.stop()

    def stop(self) -> None:
        self._stop = True
