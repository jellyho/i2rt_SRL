"""Controllable policy streamer used by the DAgger deployment UI.

Unlike the headless bridge, this runner does not own cameras and does not decide
whether policy rollout is active. The robot-side DAgger controller is the source
of truth; this runner only sends policy actions while the robot snapshot reports
``policy_running`` and not intervention/rewind/homing/e-stop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np
from yam_policy import validate_action_chunk, validate_action_step, validate_server_metadata

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.observation import build_observation

logger = logging.getLogger(__name__)

_WARMUP_TIMEOUT_S = 30.0


class DeploymentPolicyRunner:
    def __init__(
        self,
        cfg: BridgeConfig,
        recorder_cfg: RecorderConfig,
        images_fn: Callable[[], Dict[str, np.ndarray]],
        camera_health_fn: Callable[[], bool] | None = None,
        robot_io: object | None = None,
    ):
        self.cfg = cfg
        self.recorder_cfg = recorder_cfg
        self.images_fn = images_fn
        self.camera_health_fn = camera_health_fn or (lambda: True)
        self._shared_robot = robot_io
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._robot = None
        self._policy = None
        self._policy_client = None
        self._policy_warmed = False
        self._image_shape = (cfg.image_size, cfg.image_size)
        self._image_keys = dict(cfg.image_keys)
        self._was_streaming = False
        self._lock = threading.Lock()
        self._status = {
            "robot_connected": False,
            "policy_connected": False,
            "policy_ready": False,
            "streaming": False,
            "last_error": "",
            "execution_horizon": cfg.execution_horizon,
            "model_action_horizon": 0,
            "image_size": cfg.image_size,
            "image_shape": self._image_shape,
            "rollout_state": "DISCONNECTED",
        }

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._drop_policy()

    def _drop_policy(self) -> None:
        if self._policy is not None and hasattr(self._policy, "close"):
            try:
                self._policy.close()
            except Exception:
                pass
        if self._policy_client is not None:
            try:
                self._policy_client.close()
            except Exception:
                pass
        self._policy = None
        self._policy_client = None
        self._policy_warmed = False
        self._set(policy_connected=False, policy_ready=False)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)

    def _connect_robot(self) -> None:
        if self._shared_robot is not None:
            self._robot = self._shared_robot
            if not bool(getattr(self._shared_robot, "connected", False)):
                raise ConnectionError("recorder robot link is not connected")
            self._set(robot_connected=True)
            return
        if self.recorder_cfg.mock or self._robot is not None:
            self._set(robot_connected=True)
            return
        from i2rt.serving.robot_client import RobotClient

        self._robot = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0)
        self._set(robot_connected=True)

    def _connect_policy(self) -> None:
        if self.recorder_cfg.mock or self._policy is not None:
            self._set(policy_connected=True)
            return
        from yam_policy import ActionChunkBroker, AsyncActionChunkBroker, WebsocketClientPolicy

        client = WebsocketClientPolicy(
            host=self.cfg.policy_host,
            port=self.cfg.policy_port,
            timeout=self.cfg.inference_timeout_s,
        )
        meta = client.get_server_metadata() or {}
        spec = validate_server_metadata(
            meta,
            configured_contract=self.cfg.contract,
            configured_execution_horizon=self.cfg.execution_horizon,
            configured_control_hz=self.cfg.rate_hz,
            allow_legacy=self.cfg.allow_legacy_metadata,
        )
        action_horizon = spec.execution_horizon
        self._image_shape = spec.image_shape
        self._image_keys = spec.image_keys
        broker_cls = AsyncActionChunkBroker if self.cfg.use_async else ActionChunkBroker
        self._policy_client = client
        self._policy = broker_cls(client, action_horizon=action_horizon)
        self._set(
            policy_connected=True,
            policy_ready=False,
            execution_horizon=action_horizon,
            model_action_horizon=spec.model_action_horizon,
            image_size=self._image_shape[0],
            image_shape=self._image_shape,
            rollout_state="WARMING",
        )
        logger.info("deploy policy metadata: %s", meta)

    def _warm_policy(self, policy_obs: Dict) -> None:
        """Compile/cache the first model request before any policy action is sent."""
        if self.recorder_cfg.mock or self._policy_warmed:
            return
        timeout = max(_WARMUP_TIMEOUT_S, self.cfg.inference_timeout_s)
        logger.info("warming deploy policy (startup timeout %.1fs)", timeout)
        response = self._policy_client.infer(policy_obs, timeout=timeout)
        validate_action_chunk(response, execution_horizon=self._status["execution_horizon"])
        self._policy_warmed = True
        self._set(policy_ready=True, rollout_state="READY", last_error="")
        logger.info(
            "deploy policy warmup complete server_infer_ms=%s model_infer_ms=%s",
            response.get("server_timing", {}).get("infer_ms", "unknown"),
            response.get("policy_timing", {}).get("infer_ms", "unknown"),
        )

    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.rate_hz, 1.0)
        while not self._stop.is_set():
            t0 = time.monotonic()
            stage = "robot"
            try:
                if self.recorder_cfg.mock:
                    self._set(
                        robot_connected=True,
                        policy_connected=True,
                        policy_ready=True,
                        streaming=False,
                        rollout_state="READY",
                        last_error="",
                    )
                else:
                    self._connect_robot()
                    obs = self._robot.get_observation()
                    blocked = bool(obs.get("rewinding") or obs.get("homing") or obs.get("estop"))
                    images = None
                    if not self._policy_warmed and not blocked and self.camera_health_fn():
                        images = self.images_fn()
                        stage = "policy warmup"
                        self._connect_policy()
                        self._warm_policy(self._build_obs(obs, images))
                        stage = "robot"
                    should_stream = bool(obs.get("policy_running")) and not (
                        obs.get("intervention")
                        or obs.get("rewinding")
                        or obs.get("homing")
                        or obs.get("estop")
                    )
                    if should_stream:
                        if not self.camera_health_fn():
                            raise RuntimeError("required camera is missing, unhealthy, or stale")
                        stage = "policy inference"
                        self._connect_policy()
                        if images is None:
                            images = self.images_fn()
                        policy_obs = self._build_obs(obs, images)
                        if not self._policy_warmed:
                            self._warm_policy(policy_obs)
                        if not self._was_streaming:
                            self._reset_policy_chunk()
                        response = self._policy.infer(policy_obs)
                        action = validate_action_step(response.get("actions"))
                        stage = "robot command"
                        self._robot.set_policy_action(self._split(action))
                        timing = response.get("broker_timing", {})
                        chunk_step = int(timing.get("chunk_step", 0))
                        if chunk_step == 0:
                            logger.info(
                                "deploy policy chunk server_infer_ms=%s model_infer_ms=%s chunk_age_ms=%.1f",
                                response.get("server_timing", {}).get("infer_ms", "cached"),
                                response.get("policy_timing", {}).get("infer_ms", "cached"),
                                float(timing.get("chunk_age_ms", 0.0)),
                            )
                        self._set(
                            streaming=True,
                            rollout_state="RUNNING",
                            last_error="",
                            server_infer_ms=response.get("server_timing", {}).get("infer_ms"),
                            model_infer_ms=response.get("policy_timing", {}).get("infer_ms"),
                            chunk_age_ms=float(timing.get("chunk_age_ms", 0.0)),
                        )
                        self._was_streaming = True
                    else:
                        self._pause_streaming(obs)
            except Exception as e:
                self._reset_policy_chunk()
                self._set(
                    streaming=False,
                    rollout_state="STOPPED",
                    last_error=f"{stage}: {type(e).__name__}: {e}",
                )
                self._was_streaming = False
                if self._robot is not None:
                    try:
                        self._robot.set_policy_running(False)
                    except Exception:
                        pass
                if stage.startswith("policy"):
                    self._drop_policy()
                else:
                    self._robot = None
                    self._set(robot_connected=False)
                logger.warning("deploy %s failed: %s", stage, e)
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

    def _pause_streaming(self, obs: Dict) -> None:
        """Stop action delivery and invalidate every pre-pause chunk."""
        if self._was_streaming:
            self._reset_policy_chunk()
        state = (
            "REWINDING"
            if obs.get("rewinding")
            else "INTERVENING"
            if obs.get("intervention")
            else "ESTOP"
            if obs.get("estop")
            else "READY"
        )
        self._set(streaming=False, rollout_state=state, last_error="")
        self._was_streaming = False

    def _build_obs(self, robot_obs: Dict, images: Dict[str, np.ndarray]) -> Dict:
        return build_observation(
            robot_obs,
            images,
            prompt=self.cfg.prompt,
            image_keys=self._image_keys,
            image_shape=self._image_shape,
        )

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}
