"""Portal bridge: poll the YAM robot server, expose the latest fused snapshot.

Connects (plain TCP) to the robot machine running
``i2rt.serving.run_robot_server`` and polls ``get_observation()``. ``get_snapshot()``
returns the latest fused frame the recorder records:

    {teleop_state, control_mode, state(42,), leader, eef, action, buttons,
     intervention, fine_grained, leader_recentering, recenter_fault, stamp}

``state``/``leader``/``eef``/``action`` are float32 vectors (or None if missing).
``mock=True`` synthesizes a teleop cycle + fake joints so the pipeline runs with no
robot.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, CONTROL_MODE, RecorderConfig

logger = logging.getLogger(__name__)

_EMPTY = {
    "teleop_state": "IDLE",
    "control_mode": CONTROL_MODE["teleop"],
    "state": None,
    "joint_pos": None,
    "leader": None,
    "eef": None,
    "action": None,
    "buttons": {},
    "intervention": False,
    "fine_grained": False,
    "leader_recentering": False,
    "recenter_fault": False,
    "policy_running": False,
    "homing": False,
    "returning": False,
    "dagger_return_configured": False,
    "dagger_return_mode": None,
    "dagger_state": "stopped",
    "last_dagger_event": None,
    "estop": False,
    "stamp": 0.0,
}


class PortalBridge:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._snap = dict(_EMPTY)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._connected = False
        self._estop_req = False
        self._estop_sent: Optional[bool] = None
        self._policy_running_req: Optional[bool] = None
        self._policy_running_sent: Optional[bool] = None
        self._intervention_req: Optional[bool] = None
        self._intervention_sent: Optional[bool] = None
        self._finish_req: Optional[str] = None
        self._return_req = False
        self._return_handoff_pending = False
        self._mock_return_until = 0.0
        self._last_err: Optional[str] = None

    @property
    def connected(self) -> bool:
        """True once the robot server has answered (always True in mock)."""
        return self.cfg.mock or self._connected

    def set_estop(self, flag: bool) -> None:
        """Request a robot e-stop; applied (and re-applied on reconnect) by the poll loop."""
        self._estop_req = bool(flag)
        if flag:
            self._return_req = False
            self._return_handoff_pending = False
            self._mock_return_until = 0.0

    def set_policy_running(self, flag: bool) -> None:
        """Request DAgger policy rollout start/stop."""
        self._policy_running_req = bool(flag)
        if not flag:
            self._return_req = False
            self._return_handoff_pending = False
            self._mock_return_until = 0.0

    def set_intervention(self, flag: bool) -> None:
        """Request DAgger human intervention state."""
        self._intervention_req = bool(flag)
        self._return_handoff_pending = False

    def finish_dagger_run(self, action: str) -> None:
        """Request DAgger keep/discard + homing."""
        action = str(action).lower()
        if action in {"keep", "discard"}:
            self._finish_req = action
            self._return_req = False
            self._return_handoff_pending = False
            self._mock_return_until = 0.0

    def return_to_dagger_pose(self) -> None:
        """Request the configured robot-side recovery move and human handoff."""
        self._return_handoff_pending = True
        self._intervention_req = None
        self._intervention_sent = None
        if self.cfg.mock:
            self._mock_return_until = time.time() + 0.75
        else:
            self._return_req = True

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        target = self._mock_loop if self.cfg.mock else self._poll_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def _server_reachable(self) -> bool:
        """Fast TCP preflight. portal's ``send`` blocks forever waiting to connect, so
        we never construct the client until something is actually listening — this is
        what stops the recorder from hanging silently against a down/unreachable server."""
        import socket

        try:
            with socket.create_connection((self.cfg.robot_host, self.cfg.robot_port), timeout=1.0):
                return True
        except OSError:
            return False

    # ------------------------------------------------------------------ poll
    def _poll_loop(self) -> None:
        # Connect inside the thread so start() never blocks the GUI if the robot
        # server isn't up yet; on a dropped connection, reconnect on the next tick.
        from i2rt.serving.robot_client import RobotClient

        period = 1.0 / max(self.cfg.fps * 2, 1)
        while not self._stop.is_set():
            try:
                if self._client is None:
                    if not self._server_reachable():
                        raise ConnectionError(
                            f"no server listening on {self.cfg.robot_host}:{self.cfg.robot_port}"
                        )
                    # TCP is open; finite timeout guards the (now non-blocking) handshake/reads.
                    self._client = RobotClient(
                        host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0
                    )
                obs = self._client.get_observation()
                with self._lock:
                    self._snap = self._assemble(obs)
                if (
                    self._return_handoff_pending
                    and bool(obs.get("intervention"))
                    and not bool(obs.get("returning"))
                ):
                    # Preserve the robot's post-return intervention state across
                    # a later reconnect without sending intervention early and
                    # racing the return request itself.
                    self._return_handoff_pending = False
                    self._intervention_req = True
                    self._intervention_sent = True
                if not self._connected:
                    logger.info("robot connected (%s:%d)", self.cfg.robot_host, self.cfg.robot_port)
                self._connected = True
                self._last_err = None
                if self._estop_sent != self._estop_req:  # apply e-stop (and re-apply on reconnect)
                    self._client.set_estop(self._estop_req)
                    self._estop_sent = self._estop_req
                if (
                    self._policy_running_req is not None
                    and self._policy_running_sent != self._policy_running_req
                ):
                    self._client.set_policy_running(self._policy_running_req)
                    self._policy_running_sent = self._policy_running_req
                if self._intervention_req is not None and self._intervention_sent != self._intervention_req:
                    self._client.set_intervention(self._intervention_req)
                    self._intervention_sent = self._intervention_req
                if self._finish_req is not None:
                    action = self._finish_req
                    self._finish_req = None
                    self._client.finish_dagger_run(action)
                if self._return_req:
                    self._return_req = False
                    self._client.return_to_dagger_pose()
            except Exception as e:
                self._client = None  # drop and retry next tick
                self._connected = False
                self._estop_sent = None  # force re-apply after reconnect
                self._policy_running_sent = None
                self._intervention_sent = None
                # Surface *why* the link is down instead of failing silently — but only
                # when the reason changes, so a persistently-down server doesn't spam.
                msg = f"{type(e).__name__}: {e}"
                if msg != self._last_err:
                    self._last_err = msg
                    logger.warning("robot link down (%s:%d) — %s", self.cfg.robot_host, self.cfg.robot_port, msg)
            time.sleep(period)

    @staticmethod
    def _fuse(sides: list, fields: tuple, per_arm: Optional[int] = None) -> Optional[np.ndarray]:
        """Concatenate ``fields`` over both arms, or None if any is missing.

        With ``per_arm`` set, also require each arm's vector to match that size
        (used for the fixed-width state/action); otherwise accept whatever dim the
        robot provides (leader / eef, which vary by embodiment).
        """
        parts = []
        for s in sides:
            if not s or any(s.get(f) is None for f in fields):
                return None
            vec = np.concatenate([np.asarray(s[f], dtype=np.float32).reshape(-1) for f in fields])
            if per_arm is not None and vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)

    def _assemble(self, obs: Dict) -> dict:
        sides = [obs.get(a) for a in ARMS]
        state = self._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        joint_pos = self._fuse(sides, ("pos",), ARM_DOF)
        leader = self._fuse(sides, ("leader_pos",))  # variable per-arm dof; saved when present
        eef = self._fuse(sides, ("eef",))  # follower end-effector pose (FK), when the robot provides it
        intervening = bool(obs.get("intervention"))
        returning = bool(obs.get("returning"))
        buttons = {a: (obs.get(a, {}) or {}).get("buttons", []) for a in ARMS}

        if self.cfg.record_source == "dagger":
            # HG-DAgger: an episode = an intervention segment; the label is the
            # human (leader) action, recorded only while intervening.
            teleop_state = "ENGAGED" if intervening or returning else "IDLE"
            action = (
                self._fuse(sides, ("applied",), ARM_DOF)
                if returning
                else self._fuse(sides, ("human",), ARM_DOF)
                if intervening
                else None
            )
            control_mode = (
                CONTROL_MODE["replay"]
                if returning
                else CONTROL_MODE["intervention"]
                if intervening
                else CONTROL_MODE["policy"]
            )
        elif self.cfg.record_source == "eval":
            # Evaluation rollout: record the executed action every tick, labeled by
            # who produced it (policy vs human intervention). Episode = arm..disarm.
            teleop_state = "ENGAGED"
            action = self._fuse(sides, ("applied",), ARM_DOF)
            control_mode = (
                CONTROL_MODE["replay"]
                if returning
                else CONTROL_MODE["intervention"]
                if intervening
                else CONTROL_MODE["policy"]
            )
        else:
            teleop_state = obs.get("teleop_state") or ("ENGAGED" if intervening else "IDLE")
            action = self._fuse(sides, ("applied",), ARM_DOF)
            control_mode = CONTROL_MODE["teleop"]

        return {
            "teleop_state": teleop_state,
            "control_mode": int(control_mode),
            "state": state,
            "joint_pos": joint_pos,
            "leader": leader,
            "eef": eef,
            "action": action,
            "buttons": buttons,
            "intervention": intervening,
            "fine_grained": bool(obs.get("fine_grained")),
            "leader_recentering": bool(obs.get("leader_recentering")),
            "recenter_fault": bool(obs.get("recenter_fault")),
            "policy_running": bool(obs.get("policy_running")),
            "homing": bool(obs.get("homing")),
            "returning": bool(obs.get("returning")),
            "dagger_return_configured": bool(obs.get("dagger_return_configured")),
            "dagger_return_mode": obs.get("dagger_return_mode"),
            "dagger_state": obs.get("dagger_state") or ("intervention" if intervening else "stopped"),
            "last_dagger_event": obs.get("last_dagger_event"),
            "estop": bool(obs.get("estop")),
            "stamp": float(obs.get("t", 0.0)),
        }

    # ------------------------------------------------------------------ mock
    def _mock_loop(self) -> None:
        cycle = [("IDLE", 1.0), ("ENGAGED", 3.0), ("HOMING", 1.5)]
        i = 0
        t0 = time.time()
        seg_start = t0
        while not self._stop.is_set():
            name, dur = cycle[i]
            now = time.time()
            phase = now - t0
            pos = 0.3 * np.sin(phase + np.arange(ARM_DOF))
            vel = 0.3 * np.cos(phase + np.arange(ARM_DOF))
            eff = 0.1 * np.ones(ARM_DOF)
            state = np.concatenate([np.concatenate([pos, vel, eff]) for _ in ARMS]).astype(np.float32)
            action = np.concatenate([pos for _ in ARMS]).astype(np.float32)
            leader = np.concatenate([pos[:6] for _ in ARMS]).astype(np.float32)  # 6-dof leader per arm
            returning = now < self._mock_return_until
            if self._return_handoff_pending and not returning:
                self._return_handoff_pending = False
                self._intervention_req = True
            with self._lock:
                self._snap = {
                    **_EMPTY,
                    "teleop_state": name,
                    "control_mode": CONTROL_MODE["replay"] if returning else CONTROL_MODE["teleop"],
                    "state": state,
                    "joint_pos": action.copy(),
                    "leader": leader,
                    "action": action,
                    "policy_running": bool(self._policy_running_req),
                    "intervention": bool(self._intervention_req) and not returning,
                    "returning": returning,
                    "dagger_return_configured": True,
                    "dagger_return_mode": "relative",
                    "dagger_state": (
                        "returning"
                        if returning
                        else "intervention"
                        if self._intervention_req
                        else "policy"
                        if self._policy_running_req
                        else "stopped"
                    ),
                    "stamp": now,
                }
            if now - seg_start >= dur:
                i = (i + 1) % len(cycle)
                seg_start = now
            time.sleep(0.01)
