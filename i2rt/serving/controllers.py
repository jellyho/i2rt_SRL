"""Bimanual controllers.

Each controller owns the robot pairs and runs the real-time control law in
``step()`` (call it from a fixed-rate loop on the robot). It keeps a thread-safe
``snapshot()`` dict and reads external inputs (policy action, gate override, replay
command) through setters. A :class:`~i2rt.serving.robot_server.RobotServer` wraps any
of these and exposes the snapshot + setters over the network.

Three modes:

* :class:`TeleopController`  — auto home/engage gate, bilateral leader→follower teleop
* :class:`DaggerController`  — HG-DAgger: policy drives, a button hands control to the human
* :class:`WrapperController` — followers track an external command (replay / direct control)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from i2rt.serving import control_config as cc
from i2rt.serving.eef import ArmKinematics
from i2rt.serving.safety import TargetSmoother, clamp_limits, is_finite_vector, max_step_from_speed
from i2rt.serving.state_utils import full_state, to_full_target
from i2rt.serving.teleop_common import (
    ArmPair,
    FineGrainedMapper,
    TeleopStateMachine,
    build_bimanual,
    build_follower_target,
    default_bimanual_specs,
    gate_distance,
    handle_button_pressed,
    read_handle,
)

logger = logging.getLogger(__name__)

_HOME_TOL = 0.05  # rad; homing considered done when the ramp is within this of home
_LEADER_HOME_TOL = 0.1  # rad; the PHYSICAL leader must also be this close (weak home_kp lags)


def _side_state(robot: Any, kin: Optional[ArmKinematics] = None) -> Dict[str, list]:
    pos, vel, eff = full_state(robot)
    out = {"pos": pos.tolist(), "vel": vel.tolist(), "eff": eff.tolist()}
    pose = kin.fk(robot.get_joint_pos()) if kin is not None else None
    if pose is not None:
        out["eef"] = pose.tolist()
    return out


def _build_kin(robots: Dict[str, Any]) -> Dict[str, ArmKinematics]:
    """One ArmKinematics per follower (for EEF FK in the snapshot / IK control)."""
    return {side: ArmKinematics(robot) for side, robot in robots.items()}


def _arm_grav_defaults(arm_type: str) -> Optional[Dict[str, np.ndarray]]:
    """The arm-type's ORIGINAL grav-comp terms (yam.yml), before any leader_* override."""
    try:
        from i2rt.robots.utils import ArmType, _load_arm_config

        hw = _load_arm_config(ArmType(arm_type))
        return {
            "kd": np.asarray(hw.grav_comp_kd, dtype=float),
            "factor": np.asarray(hw.gravity_comp_factor, dtype=float),
            "coulomb": np.asarray(hw.coulomb_friction, dtype=float),
        }
    except Exception:
        return None  # unknown arm (e.g. bare sim) -> no feel switching


class BaseController:
    """Shared transport surface for the controllers.

    Provides the thread-safe ``snapshot()``/``metadata()`` readers and **no-op
    defaults** for every input hook the :class:`~i2rt.serving.robot_server.RobotServer`
    binds; each controller overrides only the hooks it actually supports. Subclasses
    set ``self._lock``, ``self._snap`` and ``self._metadata`` in ``__init__``.
    """

    mode = "base"
    _estop = False
    command_timeout = 0.5  # s; external commands older than this are considered stale (link loss)
    _last_cmd_t = -1e9

    def snapshot(self) -> Dict:
        with self._lock:
            return dict(self._snap)

    def metadata(self) -> Dict:
        return dict(self._metadata)

    def set_estop(self, flag: bool) -> None:
        """Engage/release the e-stop. While engaged, no follower commands are sent."""
        self._estop = bool(flag)

    def _effort_guard(self, robot: Any) -> None:
        """Collision/overload guard: trip the e-stop if a follower arm effort exceeds
        ``FOLLOWER_EFFORT_LIMIT`` (no-op when unset or already e-stopped)."""
        lim = cc.FOLLOWER_EFFORT_LIMIT
        if lim is None or self._estop:
            return
        try:
            _, _, eff = full_state(robot)
            arm = np.abs(np.asarray(eff, dtype=float).reshape(-1)[:-1])  # exclude gripper
            if arm.size and float(arm.max()) > float(lim):
                logger.warning("effort guard tripped (|eff|=%.1f > %.1f Nm) -> e-stop", float(arm.max()), lim)
                self.set_estop(True)
        except Exception:
            pass

    def _touch_cmd(self) -> None:
        """Mark that a fresh external command just arrived (for the staleness watchdog)."""
        self._last_cmd_t = time.monotonic()

    def _cmd_fresh(self) -> bool:
        """True if an external command arrived within ``command_timeout`` — else the link
        is presumed lost and the follower should hold instead of replaying a stale target."""
        return (time.monotonic() - self._last_cmd_t) < self.command_timeout

    def _apply(self, robot: Any, target: np.ndarray) -> Optional[list]:
        """Clamp ``target`` to the workspace limits and command it — unless e-stopped.

        Returns the commanded target as a list (for the snapshot), or None when
        e-stopped (the follower simply holds its last command).
        """
        if self._estop:
            return None
        target = clamp_limits(target, cc.FOLLOWER_JOINT_LIMITS)
        robot.command_joint_pos(target)
        return np.asarray(target, dtype=float).tolist()

    def _init_leader_grav_sets(self, arm_type: str) -> None:
        """Capture the two leader grav-comp feel sets: the yam.yml ORIGINALS (the
        arm's construction state, used whenever no human holds it) and the
        config.yaml leader_* override feel resolved against them (human-held)."""
        from i2rt.robots.get_robot import _override_vec

        self._grav_orig = _arm_grav_defaults(arm_type)
        self._grav_free: Optional[Dict[str, np.ndarray]] = None
        if self._grav_orig is not None:
            ov = cc.leader_arm_overrides()
            self._grav_free = {
                "factor": _override_vec(self._grav_orig["factor"], ov.get("gravity_comp_factor")),
                "coulomb": _override_vec(self._grav_orig["coulomb"], ov.get("coulomb_friction")),
                "kd": _override_vec(self._grav_orig["kd"], ov.get("grav_comp_kd")),
            }
        self._leader_held: Dict[str, Optional[bool]] = {s: None for s in self.pairs}  # None -> set on first tick

    def _sync_leader_feel(self, pair: Any, side: str, held: bool) -> None:
        """Flip the leader's gravity/friction feedforward when human-held changes:
        override feel while held, yam.yml originals otherwise."""
        if self._leader_held.get(side) == held:
            return
        self._leader_held[side] = held
        vals = self._grav_free if held else self._grav_orig
        if vals is None:
            return
        leader = pair.leader
        try:
            if hasattr(leader, "set_gravity_comp_factor"):
                leader.set_gravity_comp_factor(vals["factor"])
            if hasattr(leader, "set_coulomb_friction"):
                leader.set_coulomb_friction(vals["coulomb"])
        except Exception:
            pass

    def _free_kd(self) -> Optional[np.ndarray]:
        """The human-held free-mode damping (leader_grav_comp_kd resolved), or None
        to use the leader's built-in (original) kd."""
        return self._grav_free["kd"] if self._grav_free is not None else None

    @staticmethod
    def _bounded_recenter_leader(
        pair: ArmPair,
        smoother: TargetSmoother,
        destination: np.ndarray,
        kp_scale: float,
        max_following_error: float,
    ) -> None:
        """Advance a leader toward ``destination`` with speed and spring bounds.

        ``smoother`` bounds setpoint velocity. The second clamp keeps that setpoint
        close to the measured arm, bounding PD error/force if the arm is obstructed
        or simply lags. Resetting the smoother to the clamped command prevents
        hidden target windup.
        """
        leader = pair.leader
        if not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            measured = np.asarray(leader.get_joint_pos(), dtype=float)[:m]
            destination = np.asarray(destination, dtype=float)[:m]
            planned = smoother.step(destination)
            destination_error = destination - measured
            planned_error = planned - measured
            # If the physical arm catches or crosses the virtual setpoint, never
            # command it back away from the destination. Restart that joint's
            # setpoint directly from the measured pose instead.
            wrong_direction = planned_error * destination_error <= 0.0
            planned_error = np.where(
                wrong_direction,
                np.clip(destination_error, -smoother.max_step, smoother.max_step),
                planned_error,
            )
            error = np.clip(planned_error, -max_following_error, max_following_error)
            error = np.sign(destination_error) * np.minimum(np.abs(error), np.abs(destination_error))
            command = measured + error
            smoother.reset(command)
            kd = pair.base_kd[:m] if pair.base_kd is not None else np.full(m, 0.5)
            leader.update_kp_kd(pair.base_kp[:m] * kp_scale, kd)
            leader.command_joint_pos(command)
        except Exception:
            pass

    def set_policy_action(self, data: Dict) -> None: ...
    def set_intervention(self, flag: bool) -> None: ...
    def set_policy_running(self, flag: bool) -> None: ...
    def finish_dagger_run(self, action: str) -> None: ...
    def command(self, data: Dict) -> None: ...
    def set_sim_engage(self, flag: bool) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Teleop
# ---------------------------------------------------------------------------
@dataclass
class TeleopConfig:
    sim: bool = False
    home: str = ""
    engage_thr: float = cc.ENGAGE_THR
    release_thr: float = cc.RELEASE_THR
    dwell: float = cc.DWELL_S
    home_kp: float = cc.HOME_KP
    bilateral_kp: float = cc.BILATERAL_KP
    fine_grained_scale: float = cc.FINE_GRAINED_SCALE
    fine_grained_button: str = cc.FINE_GRAINED_BUTTON
    fine_recenter_speed: float = cc.FINE_RECENTER_SPEED
    fine_recenter_kp: float = cc.FINE_RECENTER_KP
    fine_recenter_max_following_error: float = cc.FINE_RECENTER_MAX_FOLLOWING_ERROR
    fine_recenter_tolerance: float = cc.FINE_RECENTER_TOLERANCE
    fine_recenter_dwell: float = cc.FINE_RECENTER_DWELL
    fine_recenter_timeout: float = cc.FINE_RECENTER_TIMEOUT
    button_outcomes: Dict[str, str] = field(default_factory=lambda: dict(cc.DEFAULT_TELEOP_BUTTON_OUTCOMES))
    rate: float = 120.0
    ramp_speed: float = cc.RAMP_SPEED
    engage_time: float = cc.ENGAGE_TIME  # fixed-time engage catch-up (s); 0 = speed-based
    home_speed: float = cc.HOME_SPEED  # slower ramp for the homing return
    gate_joints: str = ",".join(str(j) for j in cc.GATE_JOINTS)
    arm_type: str = "yam"
    leader_gripper: str = "yam_teaching_handle"
    follower_gripper: str = "linear_4310"


class TeleopController(BaseController):
    mode = "teleop"

    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.bilateral_kp = cfg.bilateral_kp
        self.home_kp = cfg.home_kp
        self.fine_grained_button = str(cfg.fine_grained_button).lower()
        self._button_outcomes = {str(key).lower(): str(value).lower() for key, value in cfg.button_outcomes.items()}
        if self.fine_grained_button in self._button_outcomes:
            raise ValueError(
                f"fine-grained button {self.fine_grained_button!r} cannot also be an episode outcome button"
            )
        self._fine_grained = False
        self._fine_button_prev = False
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_started = 0.0
        self._recenter_within_since: Optional[float] = None
        self.pairs = build_bimanual(
            default_bimanual_specs(cfg.sim, arm_type=cfg.arm_type,
                                   leader_gripper=cfg.leader_gripper,
                                   follower_gripper=cfg.follower_gripper),
            sim=cfg.sim)
        self._ramp_step = max_step_from_speed(cfg.ramp_speed, cfg.rate)
        self._home_step = max_step_from_speed(cfg.home_speed, cfg.rate)
        self._gate_joints = [int(x) for x in cfg.gate_joints.split(",") if x.strip() != ""] if cfg.gate_joints else []
        self._caught_up = {s: False for s in self.pairs}
        self._home_d0 = {s: 0.0 for s in self.pairs}  # start distance for the homing cosine profile
        self._engage_d0 = {s: 0.0 for s in self.pairs}  # start distance for the engage cosine approach
        # fixed-time engage blend state (engage_time > 0)
        self._engage_t0 = {s: 0.0 for s in self.pairs}  # blend start time
        self._engage_T = {s: 0.0 for s in self.pairs}  # effective blend duration
        self._engage_start = {s: None for s in self.pairs}  # follower pose at blend start
        self._prev_state = TeleopStateMachine.HOMING

        first = next(iter(self.pairs.values())).follower
        n = int(first.num_dofs())
        self._has_grip = "gripper_pos" in first.get_observations()
        n_arm = n - 1 if self._has_grip else n
        self.home_arm, self.home_grip = self._parse_home(cfg.home, n_arm)
        self.home_full = np.concatenate([self.home_arm, [self.home_grip]]) if self._has_grip else self.home_arm.copy()
        self._fine_mapper = {s: FineGrainedMapper(cfg.fine_grained_scale) for s in self.pairs}
        self._recenter_target = {
            s: np.asarray(p.follower.get_joint_pos(), dtype=float).copy() for s, p in self.pairs.items()
        }
        self._last_applied = {s: target.copy() for s, target in self._recenter_target.items()}
        recenter_step = max_step_from_speed(cfg.fine_recenter_speed, cfg.rate)
        self._recenter_smooth = {
            s: TargetSmoother(
                np.asarray(p.leader.get_joint_pos(), dtype=float)[:n_arm], recenter_step
            )
            for s, p in self.pairs.items()
        }

        self.sm = TeleopStateMachine(cfg.engage_thr, cfg.release_thr, cfg.dwell)
        self._sim_engage = False

        # leader_* override feel only while ENGAGED (human holds the arm);
        # HOMING / IDLE revert to the yam.yml originals
        self._init_leader_grav_sets(cfg.arm_type)
        self._kin = _build_kin({s: p.follower for s, p in self.pairs.items()})
        self._fsmooth, self._lsmooth = {}, {}
        for side, pair in self.pairs.items():
            self._fsmooth[side] = TargetSmoother(pair.follower.get_joint_pos(), self._ramp_step)
            self._lsmooth[side] = TargetSmoother(np.asarray(pair.leader.get_joint_pos())[:n_arm], self._ramp_step)

        self._lock = threading.Lock()
        self._snap: Dict = {"mode": self.mode, "t": 0.0, "teleop_state": "HOMING", "active": False}
        self._metadata = {"mode": self.mode, "sides": list(self.pairs), "has_gripper": self._has_grip}
        gate_desc = f"joints{self._gate_joints}" if self._gate_joints else "L2(all)"
        logger.info(
            "TeleopController up: sides=%s home_arm=%s gate=%s engage>%s release<%s ramp_speed=%s bilateral_kp=%s sim=%s",
            list(self.pairs),
            np.round(self.home_arm, 2).tolist(),
            gate_desc,
            cfg.engage_thr,
            cfg.release_thr,
            cfg.ramp_speed,
            cfg.bilateral_kp,
            cfg.sim,
        )

    @staticmethod
    def _parse_home(home_str: str, n_arm: int) -> "tuple[np.ndarray, float]":
        if not home_str:
            return np.zeros(n_arm), 0.0
        vals = [float(x) for x in home_str.split(",") if x.strip() != ""]
        if len(vals) == n_arm:
            return np.asarray(vals), 0.0
        if len(vals) == n_arm + 1:
            return np.asarray(vals[:n_arm]), float(vals[n_arm])
        raise ValueError(f"home expects {n_arm} or {n_arm + 1} values, got {len(vals)}")

    # ---- external inputs (called from portal handlers) ----------------------
    def set_sim_engage(self, flag: bool) -> None:
        self._sim_engage = bool(flag)

    def _home_button_pressed(self, buttons: Dict[str, list]) -> bool:
        """Return whether a button with a configured terminal outcome is held."""
        for side, btns in buttons.items():
            for idx, value in enumerate(btns):
                if value and f"{side}.{idx}".lower() in self._button_outcomes:
                    return True
        return False

    def _reset_fine_grained(self) -> None:
        self._fine_grained = False
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_started = 0.0
        self._recenter_within_since = None
        for mapper in self._fine_mapper.values():
            mapper.reset()

    def _start_recentering(self, now: float) -> None:
        self._fine_grained = False
        self._leader_recentering = True
        self._recenter_fault = False
        self._recenter_started = now
        self._recenter_within_since = None
        for side, pair in self.pairs.items():
            self._recenter_target[side] = self._last_applied[side].copy()
            self._fsmooth[side].reset(self._recenter_target[side])
            self._recenter_smooth[side].reset(
                np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
            )
        logger.info("fine-grained teleop OFF; aligning leader while follower holds")

    def _cancel_recentering_to_fine(self) -> None:
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_within_since = None
        self._fine_grained = True
        for mapper in self._fine_mapper.values():
            mapper.reset()
        logger.info("leader alignment cancelled; fine-grained teleop ON")

    def _update_recenter_state(self, now: float) -> None:
        if not self._leader_recentering or self._recenter_fault:
            return
        aligned = True
        for side, pair in self.pairs.items():
            target = self._recenter_target[side][: self.home_arm.size]
            leader = np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
            follower = np.asarray(pair.follower.get_joint_pos(), dtype=float)[: self.home_arm.size]
            if max(float(np.max(np.abs(leader - target))), float(np.max(np.abs(follower - target)))) > self.cfg.fine_recenter_tolerance:
                aligned = False
                break
        if aligned:
            if self._recenter_within_since is None:
                self._recenter_within_since = now
            if now - self._recenter_within_since >= self.cfg.fine_recenter_dwell:
                self._leader_recentering = False
                self._recenter_within_since = None
                for side, mapper in self._fine_mapper.items():
                    mapper.reset()
                    mapper.map(
                        np.asarray(self.pairs[side].leader.get_joint_pos(), dtype=float)[
                            : self.home_arm.size
                        ],
                        self._recenter_target[side][: self.home_arm.size],
                        enabled=False,
                    )
                logger.info("leader alignment complete; normal teleop resumed")
                return
        else:
            self._recenter_within_since = None
        if now - self._recenter_started >= self.cfg.fine_recenter_timeout:
            self._recenter_fault = True
            logger.error("leader alignment timed out; follower held and leader freed")

    def _update_fine_grained(self, buttons: Dict[str, list], state: str, now: float) -> None:
        pressed = handle_button_pressed(buttons, self.fine_grained_button)
        rising = pressed and not self._fine_button_prev
        self._fine_button_prev = pressed
        # Mapping starts only after every follower has synchronized with its leader.
        if state == TeleopStateMachine.ENGAGED and all(self._caught_up.values()) and rising:
            if self._leader_recentering:
                self._cancel_recentering_to_fine()
            elif self._fine_grained:
                self._start_recentering(now)
            else:
                self._fine_grained = True
            logger.info(
                "fine-grained teleop %s (scale=%s)",
                "ON" if self._fine_grained else "OFF",
                self.cfg.fine_grained_scale,
            )

    def _homing_done(self) -> bool:
        for side, pair in self.pairs.items():
            if np.linalg.norm(self._fsmooth[side].cur - self.home_full) > _HOME_TOL:
                return False
            if np.linalg.norm(self._lsmooth[side].cur - self.home_arm) > _HOME_TOL:
                return False
            # The virtual ramp reaching home is not enough: the physical leader lags
            # the weak home_kp pull, and flipping to IDLE too early frees it short of
            # home. Keep HOMING until the leader actually arrives.
            try:
                lq = np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
                if np.linalg.norm(lq - self.home_arm) > _LEADER_HOME_TOL:
                    return False
            except Exception:
                pass
        return True

    @staticmethod
    def _ease_vel_scale(p: float) -> float:
        """Raised-cosine speed multiplier as a function of progress ``p`` (0 at the
        start, 1 at the target): ~0.5x at the ends, ~1.28x through the middle,
        averaging ~1x -- a smooth ease-in/out that's quicker in the middle. Used for
        both the homing return and the engage approach."""
        return float(0.5 + 0.785 * np.sin(np.pi * min(max(p, 0.0), 1.0)))

    # ---- one control tick (port of TeleopNode._loop) ------------------------
    def step(self) -> None:
        now = time.monotonic()
        arm_q, grip, buttons, dists = {}, {}, {}, []
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                logger.warning("[%s] handle read failed: %s", side, e)
                a, g, b = np.asarray(pair.leader.get_joint_pos(), dtype=float), None, []
            arm_q[side], grip[side], buttons[side] = a, g, b
            dists.append(gate_distance(a, self.home_arm, self._gate_joints))

        state = self.sm.update(dists, self._homing_done(), now)
        if self._sim_engage:
            state = TeleopStateMachine.ENGAGED
        # a leader "end episode" button (success/fail) forces homing while engaged
        if state == TeleopStateMachine.ENGAGED and self._home_button_pressed(buttons):
            self.sm.state = state = TeleopStateMachine.HOMING
        if state == TeleopStateMachine.ENGAGED and self._prev_state != TeleopStateMachine.ENGAGED:
            self._reset_fine_grained()
            for s in self.pairs:
                self._caught_up[s] = False
                self._fsmooth[s].reset(self.pairs[s].follower.get_joint_pos())
        elif state != TeleopStateMachine.ENGAGED and self._prev_state == TeleopStateMachine.ENGAGED:
            self._reset_fine_grained()
        self._update_fine_grained(buttons, state, now)

        sides_snap: Dict[str, Dict] = {}
        for side, pair in self.pairs.items():
            n = pair.follower.num_dofs()
            fsm, lsm = self._fsmooth[side], self._lsmooth[side]
            applied = None
            try:
                self._effort_guard(pair.follower)
                # override feel only while the human actually holds the leader
                self._sync_leader_feel(
                    pair,
                    side,
                    held=state == TeleopStateMachine.ENGAGED and not self._leader_recentering,
                )
                if state == TeleopStateMachine.ENGAGED:
                    if self._leader_recentering:
                        # Freeze follower arm and gripper at the last command. Human
                        # leader motion is deliberately ignored until alignment ends.
                        applied = self._recenter_target[side].copy()
                        fsm.reset(applied)
                        if self._recenter_fault or self._estop:
                            self._free_leader(pair)
                        else:
                            self._bounded_recenter_leader(
                                pair,
                                self._recenter_smooth[side],
                                applied[: self.home_arm.size],
                                self.cfg.fine_recenter_kp,
                                self.cfg.fine_recenter_max_following_error,
                            )
                        lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])
                    else:
                        desired = build_follower_target(pair.follower, arm_q[side], grip[side])
                    if not self._leader_recentering and self._caught_up[side]:
                        desired[: self.home_arm.size] = self._fine_mapper[side].map(
                            arm_q[side][: self.home_arm.size],
                            fsm.cur[: self.home_arm.size],
                            self._fine_grained,
                        )
                    if not self._leader_recentering and not is_finite_vector(desired, n):
                        desired = fsm.cur
                    if not self._leader_recentering and self._caught_up[side]:
                        applied = desired
                        fsm.reset(applied)
                    elif not self._leader_recentering and self.cfg.engage_time > 0.0:
                        # Fixed-TIME catch-up: cosine blend from the engage pose to the
                        # LIVE leader over engage_time seconds. s'(1)=0, so at handoff
                        # the command's velocity equals the leader's own — direct
                        # tracking takes over with no discontinuity. The duration is
                        # stretched if any joint's peak blend speed (pi/2 * gap / T)
                        # would exceed ramp_speed.
                        if self._prev_state != TeleopStateMachine.ENGAGED:
                            start = fsm.cur.copy()
                            gap = float(np.max(np.abs(start - desired)))
                            t_min = (np.pi / 2.0) * gap / max(self.cfg.ramp_speed, 1e-6)
                            self._engage_start[side] = start
                            self._engage_T[side] = max(self.cfg.engage_time, t_min)
                            self._engage_t0[side] = now
                        tau = (now - self._engage_t0[side]) / max(self._engage_T[side], 1e-6)
                        if tau >= 1.0:
                            self._caught_up[side] = True
                            applied = desired
                            self._fine_mapper[side].map(
                                arm_q[side][: self.home_arm.size],
                                desired[: self.home_arm.size],
                                enabled=False,
                            )
                        else:
                            s_blend = 0.5 * (1.0 - np.cos(np.pi * tau))
                            applied = self._engage_start[side] + s_blend * (desired - self._engage_start[side])
                        fsm.reset(applied)
                    elif not self._leader_recentering:
                        # Cosine ease for the catch-up from home to the (live) leader:
                        # gentle off home, quicker through the middle, gentle on arrival.
                        d = float(np.linalg.norm(fsm.cur - desired))
                        if self._prev_state != TeleopStateMachine.ENGAGED:
                            self._engage_d0[side] = max(d, 1e-6)  # capture the initial gap once
                        p = 1.0 - d / max(self._engage_d0[side], 1e-6)
                        fsm.max_step = self._ramp_step * self._ease_vel_scale(p)
                        applied = fsm.step(desired)
                        if float(np.max(np.abs(fsm.cur - desired))) < _HOME_TOL:
                            self._caught_up[side] = True
                            self._fine_mapper[side].map(
                                arm_q[side][: self.home_arm.size],
                                applied[: self.home_arm.size],
                                enabled=False,
                            )
                    # Only back-drive the leader once the follower has CAUGHT UP. Before
                    # that the follower is still near home while the leader is lifted, so
                    # back-driving would yank the leader toward home — keep it free instead.
                    if self._leader_recentering:
                        pass
                    elif self.bilateral_kp > 0.0 and self._caught_up[side]:
                        self._drive_leader(pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()])
                    else:
                        self._free_leader(pair, kd=self._free_kd())
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])
                elif state == TeleopStateMachine.HOMING:
                    # Cosine velocity profile: ease in/out at the ends, faster through
                    # the middle (avg ≈ home_speed, so total time stays similar but the
                    # return is smooth rather than a constant crawl).
                    d = float(np.linalg.norm(fsm.cur - self.home_full))
                    if self._prev_state != TeleopStateMachine.HOMING:
                        self._home_d0[side] = max(d, 1e-6)  # capture the start distance once
                    p = min(max(1.0 - d / max(self._home_d0[side], 1e-6), 0.0), 1.0)
                    fsm.max_step = lsm.max_step = self._home_step * self._ease_vel_scale(p)
                    applied = fsm.step(self.home_full)
                    self._home_leader(pair, lsm.step(self.home_arm))
                else:  # IDLE — free with the built-in ORIGINAL damping (human not holding yet)
                    fsm.max_step = self._ramp_step
                    applied = fsm.step(self.home_full)
                    self._free_leader(pair)
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])

                applied_list = self._apply(pair.follower, applied)
                if applied_list is not None:
                    self._last_applied[side] = np.asarray(applied_list, dtype=float)
                snap = _side_state(pair.follower, self._kin.get(side))
                snap["leader_pos"] = np.asarray(pair.leader.get_joint_pos(), dtype=float).tolist()
                snap["buttons"] = list(buttons[side])
                snap["gripper_cmd"] = float(grip[side]) if grip[side] is not None else 0.0
                snap["applied"] = applied_list
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] teleop step failed: %s", side, e)

        self._update_recenter_state(now)

        with self._lock:
            self._snap = {
                "mode": self.mode,
                "t": now,
                "teleop_state": state,
                "active": state == TeleopStateMachine.ENGAGED,
                "fine_grained": self._fine_grained,
                "leader_recentering": self._leader_recentering,
                "recenter_fault": self._recenter_fault,
                "estop": self._estop,
                **sides_snap,
            }
        self._prev_state = state

    # ---- leader modes ------------------------------------------------------
    def _home_leader(self, pair: ArmPair, target_arm: np.ndarray) -> None:
        leader = pair.leader
        if not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            kd = pair.base_kd[:m] if pair.base_kd is not None else np.full(m, 0.5)
            leader.update_kp_kd(pair.base_kp[:m] * self.home_kp, kd)
            leader.command_joint_pos(np.asarray(target_arm, dtype=float)[:m])
        except Exception:
            pass

    def _free_leader(self, pair: ArmPair, kd: Optional[np.ndarray] = None) -> None:
        """Grav-comp idle; ``kd=None`` uses the leader's own configured (override)
        damping, an explicit vector (e.g. the yam.yml original) overrides per call."""
        leader = pair.leader
        if hasattr(leader, "enter_gravity_comp_idle"):
            try:
                if kd is None:
                    leader.enter_gravity_comp_idle()
                else:
                    leader.enter_gravity_comp_idle(kd=kd)
            except Exception:
                pass

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray) -> None:
        leader = pair.leader
        if self.bilateral_kp <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * self.bilateral_kp, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def close(self) -> None:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------
@dataclass
class DaggerConfig:
    sim: bool = False
    home: str = ""
    mirror_kp: float = cc.DAGGER_MIRROR_KP
    feedback_kp: float = cc.DAGGER_FEEDBACK_KP
    fine_grained_scale: float = cc.FINE_GRAINED_SCALE
    fine_grained_button: str = cc.FINE_GRAINED_BUTTON
    fine_recenter_speed: float = cc.FINE_RECENTER_SPEED
    fine_recenter_kp: float = cc.FINE_RECENTER_KP
    fine_recenter_max_following_error: float = cc.FINE_RECENTER_MAX_FOLLOWING_ERROR
    fine_recenter_tolerance: float = cc.FINE_RECENTER_TOLERANCE
    fine_recenter_dwell: float = cc.FINE_RECENTER_DWELL
    fine_recenter_timeout: float = cc.FINE_RECENTER_TIMEOUT
    home_kp: float = cc.HOME_KP
    home_speed: float = cc.HOME_SPEED
    rate: float = 120.0
    max_joint_speed: float = 1.5
    command_timeout: float = 0.5  # s; stale policy actions (link loss) are ignored -> hold
    arm_type: str = "yam"
    leader_gripper: str = "yam_teaching_handle"
    follower_gripper: str = "linear_4310"
    button_map: Dict[str, str] = field(
        default_factory=lambda: {
            "left.0": "rollout_toggle",
            "left.1": "intervention_toggle",
            "right.0": "discard_home",
            "right.1": "keep_home",
        }
    )


class DaggerController(BaseController):
    mode = "dagger"

    def __init__(self, cfg: DaggerConfig):
        self.cfg = cfg
        self.command_timeout = cfg.command_timeout
        self.mirror_kp = cfg.mirror_kp
        self.feedback_kp = cfg.feedback_kp
        self.home_kp = cfg.home_kp
        self.fine_grained_button = str(cfg.fine_grained_button).lower()
        self._fine_grained = False
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_started = 0.0
        self._recenter_within_since: Optional[float] = None
        self.pairs = build_bimanual(
            default_bimanual_specs(cfg.sim, arm_type=cfg.arm_type,
                                   leader_gripper=cfg.leader_gripper,
                                   follower_gripper=cfg.follower_gripper),
            sim=cfg.sim)
        self._run_step = max_step_from_speed(cfg.max_joint_speed, cfg.rate)
        self._home_step = max_step_from_speed(cfg.home_speed, cfg.rate)

        self._intervening = False
        self._policy_running = False
        self._homing = False
        self._btn_prev: Dict[str, list] = {}
        self._button_map: Dict[str, str] = {str(k).lower(): str(v).lower() for k, v in cfg.button_map.items()}
        self._event_seq = 0
        self._last_event: Optional[Dict[str, object]] = None
        self._policy_action: Dict[str, Optional[np.ndarray]] = {s: None for s in self.pairs}
        # The config.yaml leader_* overrides are the HUMAN-HELD feel (intervention);
        # in every other phase (mirroring, homing, stopped) all grav-comp terms
        # revert to the yam.yml originals: kd via the mirror PD command, gravity
        # factor + Coulomb feedforward via _sync_leader_feel.
        self._init_leader_grav_sets(cfg.arm_type)
        self._mirror_kd = self._grav_orig["kd"] if self._grav_orig is not None else None
        self._kin = _build_kin({s: p.follower for s, p in self.pairs.items()})
        self._smooth = {s: TargetSmoother(p.follower.get_joint_pos(), self._run_step) for s, p in self.pairs.items()}
        first = next(iter(self.pairs.values())).follower
        n = int(first.num_dofs())
        self._has_grip = "gripper_pos" in first.get_observations()
        n_arm = n - 1 if self._has_grip else n
        self.home_arm, self.home_grip = self._parse_home(cfg.home, n_arm)
        self.home_full = np.concatenate([self.home_arm, [self.home_grip]]) if self._has_grip else self.home_arm.copy()
        self._fine_mapper = {s: FineGrainedMapper(cfg.fine_grained_scale) for s in self.pairs}
        self._recenter_target = {
            s: np.asarray(p.follower.get_joint_pos(), dtype=float).copy() for s, p in self.pairs.items()
        }
        self._last_applied = {s: target.copy() for s, target in self._recenter_target.items()}
        recenter_step = max_step_from_speed(cfg.fine_recenter_speed, cfg.rate)
        self._recenter_smooth = {
            s: TargetSmoother(
                np.asarray(p.leader.get_joint_pos(), dtype=float)[:n_arm], recenter_step
            )
            for s, p in self.pairs.items()
        }
        self._leader_smooth = {
            s: TargetSmoother(np.asarray(p.leader.get_joint_pos())[: self.home_arm.size], self._home_step)
            for s, p in self.pairs.items()
        }
        self._home_d0 = {s: 0.0 for s in self.pairs}

        self._lock = threading.Lock()
        self._snap: Dict = {
            "mode": self.mode,
            "t": 0.0,
            "intervention": False,
            "fine_grained": False,
            "leader_recentering": False,
            "recenter_fault": False,
            "policy_running": False,
            "homing": False,
            "dagger_state": "stopped",
            "last_dagger_event": None,
        }
        self._metadata = {"mode": self.mode, "sides": list(self.pairs), "has_gripper": self._has_grip}
        logger.info(
            "DaggerController up: sides=%s home_arm=%s mirror_kp=%s feedback_kp=%s max_joint_speed=%s sim=%s",
            list(self.pairs),
            np.round(self.home_arm, 2).tolist(),
            cfg.mirror_kp,
            cfg.feedback_kp,
            cfg.max_joint_speed,
            cfg.sim,
        )

    @staticmethod
    def _parse_home(home_str: str, n_arm: int) -> "tuple[np.ndarray, float]":
        if not home_str:
            return np.zeros(n_arm), 0.0
        vals = [float(x) for x in home_str.split(",") if x.strip() != ""]
        if len(vals) == n_arm:
            return np.asarray(vals), 0.0
        if len(vals) == n_arm + 1:
            return np.asarray(vals[:n_arm]), float(vals[n_arm])
        raise ValueError(f"home expects {n_arm} or {n_arm + 1} values, got {len(vals)}")

    @staticmethod
    def _ease_vel_scale(p: float) -> float:
        return float(0.5 + 0.785 * np.sin(np.pi * min(max(p, 0.0), 1.0)))

    # ---- external inputs ----------------------------------------------------
    def set_policy_action(self, data: Dict) -> None:
        """data = {side: position_array}. Stored as full-length follower targets."""
        for side, pos in (data or {}).items():
            if side not in self.pairs:
                continue
            try:
                self._policy_action[side] = to_full_target(np.asarray(pos, dtype=float), self.pairs[side].follower)
            except ValueError as e:
                logger.warning("[%s] bad policy_action: %s", side, e)
        self._touch_cmd()

    def set_intervention(self, flag: bool) -> None:
        if self._homing:
            return
        flag = bool(flag)
        if flag == self._intervening:
            return
        self._intervening = flag
        self._reset_fine_grained()
        # Handoff to policy mirroring starts from the physical leader pose, never
        # from a potentially distant target left by an offset intervention.
        for side, pair in self.pairs.items():
            self._leader_smooth[side].reset(
                np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
            )

    def set_policy_running(self, flag: bool) -> None:
        if self._homing:
            return
        flag = bool(flag)
        if flag and not self._policy_running and not self._intervening:
            for side, pair in self.pairs.items():
                self._leader_smooth[side].reset(
                    np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
                )
        self._policy_running = flag
        if not self._policy_running:
            self.set_intervention(False)

    def finish_dagger_run(self, action: str) -> None:
        action = str(action).lower()
        if action not in {"keep", "discard"}:
            logger.warning("unknown dagger finish action: %s", action)
            return
        self._event_seq += 1
        self._last_event = {"seq": self._event_seq, "action": action}
        self._policy_running = False
        self._intervening = False
        self._reset_fine_grained()
        self._homing = True
        for side, pair in self.pairs.items():
            self._smooth[side].reset(pair.follower.get_joint_pos())
            self._leader_smooth[side].reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])
            self._home_d0[side] = max(float(np.linalg.norm(self._smooth[side].cur - self.home_full)), 1e-6)

    def _toggle_button_action(self, action: str) -> None:
        if action == "rollout_toggle":
            self.set_policy_running(not self._policy_running)
        elif action == "intervention_toggle":
            self.set_intervention(not self._intervening)
        elif action == "keep_home":
            self.finish_dagger_run("keep")
        elif action == "discard_home":
            self.finish_dagger_run("discard")
        else:
            logger.warning("unknown dagger button action: %s", action)

    def _reset_fine_grained(self) -> None:
        self._fine_grained = False
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_started = 0.0
        self._recenter_within_since = None
        for mapper in self._fine_mapper.values():
            mapper.reset()

    def _start_recentering(self, now: float) -> None:
        self._fine_grained = False
        self._leader_recentering = True
        self._recenter_fault = False
        self._recenter_started = now
        self._recenter_within_since = None
        for side, pair in self.pairs.items():
            self._recenter_target[side] = self._last_applied[side].copy()
            self._smooth[side].reset(self._recenter_target[side])
            self._recenter_smooth[side].reset(
                np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
            )
        logger.info("fine-grained intervention OFF; aligning leader while follower holds")

    def _cancel_recentering_to_fine(self) -> None:
        self._leader_recentering = False
        self._recenter_fault = False
        self._recenter_within_since = None
        self._fine_grained = True
        for mapper in self._fine_mapper.values():
            mapper.reset()
        logger.info("leader alignment cancelled; fine-grained intervention ON")

    def _update_recenter_state(self, now: float) -> None:
        if not self._leader_recentering or self._recenter_fault:
            return
        aligned = True
        for side, pair in self.pairs.items():
            target = self._recenter_target[side][: self.home_arm.size]
            leader = np.asarray(pair.leader.get_joint_pos(), dtype=float)[: self.home_arm.size]
            follower = np.asarray(pair.follower.get_joint_pos(), dtype=float)[: self.home_arm.size]
            if max(float(np.max(np.abs(leader - target))), float(np.max(np.abs(follower - target)))) > self.cfg.fine_recenter_tolerance:
                aligned = False
                break
        if aligned:
            if self._recenter_within_since is None:
                self._recenter_within_since = now
            if now - self._recenter_within_since >= self.cfg.fine_recenter_dwell:
                self._leader_recentering = False
                self._recenter_within_since = None
                for side, mapper in self._fine_mapper.items():
                    mapper.reset()
                    mapper.map(
                        np.asarray(self.pairs[side].leader.get_joint_pos(), dtype=float)[
                            : self.home_arm.size
                        ],
                        self._recenter_target[side][: self.home_arm.size],
                        enabled=False,
                    )
                logger.info("leader alignment complete; normal intervention resumed")
                return
        else:
            self._recenter_within_since = None
        if now - self._recenter_started >= self.cfg.fine_recenter_timeout:
            self._recenter_fault = True
            logger.error("leader alignment timed out; follower held and leader freed")

    def _scan_buttons(self, buttons: Dict[str, list]) -> None:
        if self._homing:
            self._btn_prev = {side: list(btns) for side, btns in buttons.items()}
            return
        for side, btns in buttons.items():
            prev = self._btn_prev.get(side, [])
            for idx in range(len(btns)):
                pressed = bool(btns[idx])
                was = idx < len(prev) and bool(prev[idx])
                if pressed and not was:
                    key = f"{side}.{idx}".lower()
                    if key == self.fine_grained_button and self._intervening:
                        if self._leader_recentering:
                            self._cancel_recentering_to_fine()
                        elif self._fine_grained:
                            self._start_recentering(time.monotonic())
                        else:
                            self._fine_grained = True
                        logger.info(
                            "fine-grained intervention %s (scale=%s)",
                            "ON" if self._fine_grained else "OFF",
                            self.cfg.fine_grained_scale,
                        )
                    else:
                        action = self._button_map.get(key)
                        if action:
                            self._toggle_button_action(action)
            self._btn_prev[side] = list(btns)

    def _homing_done(self) -> bool:
        for side in self.pairs:
            if np.linalg.norm(self._smooth[side].cur - self.home_full) > _HOME_TOL:
                return False
            if np.linalg.norm(self._leader_smooth[side].cur - self.home_arm) > _HOME_TOL:
                return False
        return True

    # ---- one control tick (port of DaggerNode._loop) ------------------------
    def step(self) -> None:
        now = time.monotonic()
        arm_q, grip_cmd, buttons, valid = {}, {}, {}, {}
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                logger.warning("[%s] handle read failed: %s", side, e)
                a, g, b = np.zeros(pair.leader.num_dofs()), None, []
            arm_q[side], grip_cmd[side], buttons[side] = a, g, b
            valid[side] = is_finite_vector(a, pair.leader.num_dofs())

        self._scan_buttons(buttons)

        sides_snap: Dict[str, Dict] = {}
        for side, pair in self.pairs.items():
            n = pair.follower.num_dofs()
            smoother = self._smooth[side]
            applied = None
            human = None
            try:
                self._effort_guard(pair.follower)
                # override feel only while the human actually holds the leader
                self._sync_leader_feel(
                    pair, side, held=self._intervening and not self._leader_recentering
                )
                desired = None
                if self._homing:
                    d = float(np.linalg.norm(smoother.cur - self.home_full))
                    p = min(max(1.0 - d / max(self._home_d0[side], 1e-6), 0.0), 1.0)
                    smoother.max_step = self._leader_smooth[side].max_step = self._home_step * self._ease_vel_scale(p)
                    desired = smoother.step(self.home_full)
                    self._home_leader(pair, self._leader_smooth[side].step(self.home_arm))
                elif self._intervening:
                    smoother.max_step = self._run_step
                    if self._leader_recentering:
                        desired = self._recenter_target[side].copy()
                        smoother.reset(desired)
                        if self._recenter_fault or self._estop:
                            self._free_leader(pair)
                        else:
                            self._bounded_recenter_leader(
                                pair,
                                self._recenter_smooth[side],
                                desired[: self.home_arm.size],
                                self.cfg.fine_recenter_kp,
                                self.cfg.fine_recenter_max_following_error,
                            )
                    elif valid[side]:
                        human = build_follower_target(pair.follower, arm_q[side], grip_cmd[side])
                        human[: self.home_arm.size] = self._fine_mapper[side].map(
                            arm_q[side][: self.home_arm.size],
                            smoother.cur[: self.home_arm.size],
                            self._fine_grained,
                        )
                        desired = human
                        if self.feedback_kp > 0.0:
                            self._drive_leader(
                                pair,
                                np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()],
                                self.feedback_kp,
                            )
                        else:
                            # feedback_kp=0: fully free leader — grav-comp idle with ZERO
                            # damping (grav_comp_kd off), not a stale PD hold from mirroring.
                            self._free_leader(pair)
                elif self._policy_running:
                    smoother.max_step = self._run_step
                    act = self._policy_action[side]
                    # ignore a stale policy action (workstation/link down) -> follower holds
                    if is_finite_vector(act, n) and self._cmd_fresh():
                        desired = act[:n]

                if desired is not None:
                    target = desired if self._leader_recentering else smoother.step(desired)
                    if self._intervening and human is not None and self._has_grip:
                        # Match steady teleoperation: the teaching-handle trigger
                        # directly controls the normalized gripper command.  Keep
                        # arm joints rate-limited across the policy/human handoff,
                        # but do not make a full gripper close take 1 / joint-speed
                        # seconds.  Synchronize the smoother so the next handoff
                        # still starts from the command that was actually applied.
                        target[-1] = human[-1]
                        smoother.cur[-1] = human[-1]
                    applied = self._apply(pair.follower, target)
                    if not self._intervening and self._policy_running and applied is not None:
                        # Mirror the command sent after follower smoothing/clamping.
                        # The independent limiter prevents a PD target jump when an
                        # offset human intervention hands control back to the policy.
                        leader_smoother = self._leader_smooth[side]
                        leader_smoother.max_step = self._run_step
                        mirror_target = leader_smoother.step(
                            np.asarray(applied, dtype=float)[: pair.leader.num_dofs()]
                        )
                        self._drive_leader(pair, mirror_target, self.mirror_kp, kd=self._mirror_kd)
                else:
                    smoother.reset(pair.follower.get_joint_pos())

                if applied is not None:
                    self._last_applied[side] = np.asarray(applied, dtype=float)

                snap = _side_state(pair.follower, self._kin.get(side))
                snap["leader_pos"] = np.asarray(pair.leader.get_joint_pos(), dtype=float).tolist()
                snap["buttons"] = list(buttons[side])
                snap["gripper_cmd"] = float(grip_cmd[side]) if grip_cmd[side] is not None else 0.0
                snap["applied"] = applied
                snap["human"] = np.asarray(human, dtype=float).tolist() if human is not None else None
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] dagger step failed: %s", side, e)

        self._update_recenter_state(now)

        if self._homing and self._homing_done():
            self._homing = False

        dagger_state = self._state_name()
        with self._lock:
            self._snap = {
                "mode": self.mode,
                "t": now,
                "intervention": bool(self._intervening),
                "fine_grained": self._fine_grained,
                "leader_recentering": self._leader_recentering,
                "recenter_fault": self._recenter_fault,
                "policy_running": bool(self._policy_running),
                "homing": bool(self._homing),
                "dagger_state": dagger_state,
                "last_dagger_event": dict(self._last_event) if self._last_event is not None else None,
                "estop": self._estop,
                **sides_snap,
            }

    def _state_name(self) -> str:
        if self._estop:
            return "estop"
        if self._homing:
            return "homing"
        if self._intervening:
            return "intervention"
        if self._policy_running:
            return "policy"
        return "stopped"

    def _home_leader(self, pair: ArmPair, target_arm: np.ndarray) -> None:
        leader = pair.leader
        if not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            kd = pair.base_kd[:m] if pair.base_kd is not None else np.full(m, 0.5)
            leader.update_kp_kd(pair.base_kp[:m] * self.home_kp, kd)
            leader.command_joint_pos(np.asarray(target_arm, dtype=float)[:m])
        except Exception:
            pass

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray, kp_scale: float,
                      kd: Optional[np.ndarray] = None) -> None:
        leader = pair.leader
        if kp_scale <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            kd_vec = np.zeros(m) if kd is None else np.asarray(kd, dtype=float)[:m]
            leader.update_kp_kd(pair.base_kp[:m] * kp_scale, kd_vec)
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def _free_leader(self, pair: ArmPair) -> None:
        """Leader free, teleop-identical feel: grav-comp idle with the resolved
        leader_grav_comp_kd override (the leader is BUILT with the original kd)."""
        leader = pair.leader
        if not hasattr(leader, "enter_gravity_comp_idle"):
            return
        try:
            kd = self._free_kd()
            if kd is None:
                leader.enter_gravity_comp_idle()
            else:
                leader.enter_gravity_comp_idle(kd=kd)
        except Exception:
            pass

    def close(self) -> None:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Wrapper / replay — followers track an external command
# ---------------------------------------------------------------------------
@dataclass
class WrapperConfig:
    sim: bool = False
    arm_type: str = "yam"
    gripper: str = "linear_4310"
    rate: float = 100.0
    max_joint_speed: float = 1.5
    control: str = "joint"  # "joint" (rate-limited joint targets) or "eef" (end-effector pose; experimental)
    command_timeout: float = 0.5  # s; stale commands (link loss) are ignored -> hold
    channels: Dict[str, str] = field(default_factory=lambda: {"left": "can_follower_l", "right": "can_follower_r"})


class WrapperController(BaseController):
    mode = "wrapper"

    def __init__(self, cfg: WrapperConfig):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.cfg = cfg
        self.command_timeout = cfg.command_timeout
        self.followers = {}
        for side, channel in cfg.channels.items():
            ch = ("sim_" + channel) if cfg.sim else channel
            self.followers[side] = get_yam_robot(
                channel=ch,
                arm_type=ArmType(cfg.arm_type),
                gripper_type=GripperType(cfg.gripper),
                zero_gravity_mode=False,
                sim=cfg.sim,
            )
            cc.apply_follower_gains(self.followers[side])

        self._kin = _build_kin(self.followers)
        max_step = max_step_from_speed(cfg.max_joint_speed, cfg.rate)
        self._smooth = {s: TargetSmoother(f.get_joint_pos(), max_step) for s, f in self.followers.items()}
        self._command: Dict[str, Optional[np.ndarray]] = {s: None for s in self.followers}
        self._has_grip = "gripper_pos" in next(iter(self.followers.values())).get_observations()
        self._lock = threading.Lock()
        self._snap: Dict = {"mode": self.mode, "t": 0.0}
        self._metadata = {"mode": self.mode, "sides": list(self.followers), "has_gripper": self._has_grip}
        logger.info("WrapperController up: sides=%s sim=%s", list(self.followers), cfg.sim)

    def command(self, data: Dict) -> None:
        """data = {side: target} for each follower (joint positions, or eef pose in eef mode)."""
        if self.cfg.control == "eef":
            self._command_eef(data)
            return
        for side, pos in (data or {}).items():
            if side not in self.followers:
                continue
            try:
                self._command[side] = to_full_target(np.asarray(pos, dtype=float), self.followers[side])
            except ValueError as e:
                logger.warning("[%s] bad command: %s", side, e)
        self._touch_cmd()

    def _command_eef(self, data: Dict) -> None:
        """Safe operational-space (resolved-rate) control: resolve each EE pose target to
        joint positions with the company IK (mink, limits + damping), seeded at the
        current pose, then drive them through the SAME joint path as joint mode — the
        TargetSmoother rate limit, joint clamp, e-stop and watchdog all still apply.
        """
        for side, pose in (data or {}).items():
            follower = self.followers.get(side)
            kin = self._kin.get(side)
            if follower is None or kin is None or not kin.available:
                if not getattr(self, "_eef_warned", False):
                    logger.warning("eef control requested but no IK model is available; ignoring")
                    self._eef_warned = True
                continue
            cur = np.asarray(follower.get_joint_pos(), dtype=float)
            q = kin.ik(np.asarray(pose, dtype=float), init_q=cur)  # full model config (nq) or None
            if q is None:
                continue
            n_arm = cur.size - 1  # gripper is the trailing dof; arm joints lead
            target = cur.copy()
            target[:n_arm] = q[:n_arm]  # IK arm solution; keep the current gripper opening
            self._command[side] = target
        self._touch_cmd()

    # a policy can drive the wrapper too (treated as a direct command)
    set_policy_action = command

    def step(self) -> None:
        now = time.monotonic()
        sides_snap: Dict[str, Dict] = {}
        for side, follower in self.followers.items():
            smoother = self._smooth[side]
            applied = None
            try:
                self._effort_guard(follower)
                cmd = self._command[side]
                # ignore stale commands (link loss) -> hold instead of replaying an old target
                if is_finite_vector(cmd, follower.num_dofs()) and self._cmd_fresh():
                    applied = self._apply(follower, smoother.step(cmd))
                else:
                    smoother.reset(follower.get_joint_pos())
                snap = _side_state(follower, self._kin.get(side))
                snap["applied"] = applied
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] wrapper step failed: %s", side, e)
        with self._lock:
            self._snap = {"mode": self.mode, "t": now, "estop": self._estop, **sides_snap}

    def close(self) -> None:
        for f in self.followers.values():
            try:
                f.close()
            except Exception:
                pass
