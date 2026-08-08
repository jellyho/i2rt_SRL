"""Controllable policy streamer used by the DAgger deployment UI.

Unlike the headless bridge, this runner does not own cameras and does not decide
whether policy rollout is active. The robot-side DAgger controller is the source
of truth; this runner only sends policy actions while the robot snapshot reports
``policy_running`` and not intervention/homing/e-stop.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
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
        # Seconds between idle reachability probes -- see _probe_policy.
        self._PROBE_PERIOD_S = 2.0
        self._last_probe = -1e9
        self._last_probe_error = ""
        self._status = {
            "robot_connected": False,
            "policy_connected": False,
            "streaming": False,
            # Not "" -- an empty error next to a red dot reads as "no problem", when what it
            # really means is that nothing has been tried yet.
            "last_error": "connecting…",
            "action_horizon": 0,  # filled in from the first chunk the policy returns
            "image_size": cfg.image_size,
            "image_shape": self._image_shape,
            "num_samples": 0,   # how many chunks the last reply actually carried
        }
        # The most recent chunk set, for the overlay. Held outside _status because it is an
        # array rather than a status scalar, and a viewer reads it at its own pace.
        self._samples: Optional[np.ndarray] = None

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
        """Update the status, announcing the link transitions on the way through.

        Every client -- the GUI, ``--headless``, the sim harness -- watches this same dict, so
        saying it here says it once for all of them. A periodic status line cannot: it prints
        "policy idle", which means connected-but-not-running and not-connected-at-all equally
        well. What an operator needs is the moment it changes.
        """
        with self._lock:
            transitions = [
                (key, was, kw[key])
                for key, was in ((k, self._status.get(k)) for k in ("robot_connected", "policy_connected"))
                if key in kw and bool(kw[key]) is not bool(was)
            ]
            self._status.update(kw)
            error = str(kw.get("last_error", self._status.get("last_error", "")))

        for key, _was, now in transitions:
            link = "policy" if key == "policy_connected" else "robot"
            host, port = ((self.cfg.policy_host, self.cfg.policy_port) if link == "policy"
                          else (self.cfg.robot_host, self.cfg.robot_port))
            if now:
                logger.info("%s CONNECTED at %s:%s", link, host, port)
            else:
                logger.warning("%s DISCONNECTED from %s:%s%s", link, host, port,
                               f" ({error})" if error else "")

    def _set_samples(self, samples) -> None:
        """Keep the chunk set the policy just returned, if it returned one.

        Cleared when a reply has none, so a viewer never draws a stale spread over a live
        frame -- a picture that is confidently about the wrong moment.
        """
        with self._lock:
            self._samples = np.asarray(samples, dtype=float) if samples is not None else None
            self._status["num_samples"] = 0 if self._samples is None else int(self._samples.shape[0])

    def get_samples(self) -> Optional[np.ndarray]:
        """``[N, horizon, action_dim]`` from the most recent inference, or None."""
        with self._lock:
            return None if self._samples is None else self._samples.copy()

    def _connect_robot(self) -> None:
        if self.recorder_cfg.mock or self._robot is not None:
            self._set(robot_connected=True)
            return
        from i2rt.serving.robot_client import RobotClient

        self._robot = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0)
        self._set(robot_connected=True)

    def _policy_port_open(self) -> bool:
        """Is the policy server up? Asked over HTTP, because the alternative is noisy.

        This check has to exist at all because `WebsocketClientPolicy` retries a refused
        connection forever, five seconds at a time -- calling it against a server that is not
        up would wedge the control loop instead of reporting the state.

        It used to open a bare TCP connection and close it. That works, but the server is a
        websocket server: a socket that connects and says nothing is a failed opening
        handshake, and it logs one with a full traceback --

            ERROR:websockets.server:opening handshake failed
            EOFError: stream ends after 0 bytes, before end of line

        -- immediately before the real connection succeeds. An operator watching the server
        cannot tell that apart from a genuine fault. openpi's server answers `/healthz` for
        exactly this, so ask it properly.

        Any HTTP answer means something is listening and speaking HTTP, so a non-200 (an older
        server without the route) still counts as up; only a connection-level failure is down.
        """
        url = f"http://{self.cfg.policy_host}:{self.cfg.policy_port}/healthz"
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return True
        except urllib.error.HTTPError:
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
                            result = self._policy.infer(policy_obs)
                            action = result["actions"]
                            self._set_samples(result.get("action_samples"))
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
                        self._set(streaming=False)
                        self._was_streaming = False
                        self._probe_policy()
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

    def _probe_policy(self) -> None:
        """Connect to the policy while idle, so the UI can say whether it is there.

        The connection used to be made only inside the streaming branch, which needs the robot
        to already be running a rollout. Before that, `policy_connected` sat at its initial
        False with `last_error` at its initial "" -- a red dot next to "policy idle" and no
        reason given. Worse, the GUI refuses to start a rollout unless the policy is connected,
        so nothing could ever connect it: the only path to connected ran through a rollout that
        could not be started. Probing while idle is what breaks that circle.

        Throttled, because the probe is a real websocket round-trip (`get_server_metadata`) and
        the loop runs at the control rate -- retrying every tick would hammer a server that is
        still loading its checkpoint.
        """
        if self._policy is not None:
            return
        now = time.monotonic()
        if now - self._last_probe < self._PROBE_PERIOD_S:
            return
        self._last_probe = now
        try:
            self._connect_policy()
            self._set(last_error="")
            self._last_probe_error = ""
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            # _set only announces a *transition*, and a policy that was never up does not
            # transition -- so starting with no server at all would say nothing whatsoever.
            # Report each distinct reason once: enough to see why, without a line every probe.
            # Stay quiet when _set is about to announce the drop itself, or losing a live
            # policy would report the same event twice.
            was_connected = bool(self.get_status().get("policy_connected"))
            if reason != self._last_probe_error and not was_connected:
                self._last_probe_error = reason
                logger.warning(
                    "policy NOT CONNECTED at %s:%s (%s) — retrying every %.0fs",
                    self.cfg.policy_host, self.cfg.policy_port, reason, self._PROBE_PERIOD_S,
                )
            self._set(policy_connected=False, last_error=reason)

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
        # Only sent when actually wanted: a server that supports it does N forward passes for
        # N samples, and an unpatched one ignores an unknown key rather than failing.
        if self.cfg.num_samples > 1:
            obs["num_samples"] = int(self.cfg.num_samples)

        height, width = self._image_shape
        for role, key in self.cfg.image_keys.items():
            if role in images:
                img = image_tools.resize_with_pad(images[role], height, width)
                obs[key] = image_tools.convert_to_uint8(img)
        return obs

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}
