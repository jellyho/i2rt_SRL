"""Unified config: one YAML, shared by the robot server and the workstation tools.

A single ``<repo>/config.yaml`` is the source of truth for the whole setup — robot
host/port, control gains/limits, camera serials, recorder defaults, tasks, and the
policy endpoint. It is auto-discovered (no env var, no cwd dependence); ``--config``
can point elsewhere for a one-off. Precedence is

    explicit CLI flag  >  <repo>/config.yaml  >  built-in default

Example:

    robot:    {host: 192.168.1.10, port: 11331}
    policy:   {host: 192.168.1.20, port: 8000}
    control:  {bilateral_kp: 0.15, home_speed: 0.4, follower_effort_limit: 30.0,
               follower_joint_limits: [[-3.0, 3.0], ...]}
    cameras:  {agentview: "1234", wrist_left: "5678", wrist_right: "9012"}
    recorder: {repo_id: user/yam_pick, root: ~/lerobot_data, fps: 60, min_free_gb: 2.0}
    tasks:    ["pick the cube", "stack the blocks"]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

# rig['control'] key -> i2rt.serving.control_config attribute
_CONTROL_MAP = {
    "follower_kp": "FOLLOWER_KP",
    "follower_kd": "FOLLOWER_KD",
    "ramp_speed": "RAMP_SPEED",
    "engage_time": "ENGAGE_TIME",
    "home_speed": "HOME_SPEED",
    "engage_thr": "ENGAGE_THR",
    "release_thr": "RELEASE_THR",
    "dwell": "DWELL_S",
    "gate_joints": "GATE_JOINTS",
    "home_kp": "HOME_KP",
    "bilateral_kp": "BILATERAL_KP",
    "fine_grained_scale": "FINE_GRAINED_SCALE",
    "fine_grained_button": "FINE_GRAINED_BUTTON",
    "fine_recenter_speed": "FINE_RECENTER_SPEED",
    "fine_recenter_kp": "FINE_RECENTER_KP",
    "fine_recenter_max_following_error": "FINE_RECENTER_MAX_FOLLOWING_ERROR",
    "fine_recenter_tolerance": "FINE_RECENTER_TOLERANCE",
    "fine_recenter_dwell": "FINE_RECENTER_DWELL",
    "fine_recenter_timeout": "FINE_RECENTER_TIMEOUT",
    "leader_gravity_comp_factor": "LEADER_GRAVITY_COMP_FACTOR",
    "leader_grav_comp_kd": "LEADER_GRAV_COMP_KD",
    "leader_coulomb_friction": "LEADER_COULOMB_FRICTION",
    "dagger_mirror_kp": "DAGGER_MIRROR_KP",
    "dagger_feedback_kp": "DAGGER_FEEDBACK_KP",
    "follower_joint_limits": "FOLLOWER_JOINT_LIMITS",
    "follower_effort_limit": "FOLLOWER_EFFORT_LIMIT",
    "follower_payload_kg": "FOLLOWER_PAYLOAD_KG",
}

_TELEOP_OUTCOMES = {"success", "fail", "discard"}


def _repo_root() -> str:
    """Absolute repo root (…/i2rt/serving/rig_config.py -> …)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_config_path() -> str:
    """The single in-repo config location: ``<repo>/config.yaml``."""
    return os.path.join(_repo_root(), "config.yaml")


def find_rig(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the config file. It lives IN the repo at ``<repo>/config.yaml`` — no
    environment variables, no cwd dependence. ``--config`` may point elsewhere for a
    one-off. Returns the path, or None if neither exists.
    """
    for cand in (explicit, repo_config_path()):
        if cand and os.path.exists(os.path.expanduser(cand)):
            return os.path.abspath(os.path.expanduser(cand))
    return None


def load_rig(path: Optional[str]) -> Dict[str, Any]:
    """Load the config YAML into a dict, auto-discovering ``<repo>/config.yaml`` if
    ``path`` is None ({} if none)."""
    path = find_rig(path)
    if not path:
        return {}
    import yaml

    with open(path) as f:
        rig = yaml.safe_load(f) or {}
    logging.getLogger(__name__).info("loaded config: %s", path)
    return rig


def apply_control_overrides(rig: Dict[str, Any]) -> Dict[str, Any]:
    """Override ``i2rt.serving.control_config`` constants from ``rig['control']``.

    Call this BEFORE building controllers / argparse defaults so they pick up the
    overrides. Returns the applied {ATTR: value} for logging.
    """
    from i2rt.serving import control_config as cc

    section = (rig or {}).get("control", {}) or {}
    applied: Dict[str, Any] = {}
    for key, value in section.items():
        if key == "home_buttons":
            raise ValueError(
                "control.home_buttons was removed; teleop homing is derived from recorder.buttons"
            )
        attr = _CONTROL_MAP.get(key)
        if attr is None:
            continue
        # JSON/YAML lists -> tuples for limit pairs so they match the documented type
        if attr == "FOLLOWER_JOINT_LIMITS" and value is not None:
            value = [tuple(v) if v is not None else None for v in value]
        setattr(cc, attr, value)
        applied[attr] = value
    return applied


def teleop_button_outcomes(rig: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the one shared teleop button -> terminal outcome mapping.

    Both the robot controller and recorder call this at startup. Every configured
    button therefore has one atomic meaning: label the episode and start homing.
    """
    from i2rt.serving import control_config as cc

    recorder = (rig or {}).get("recorder", {}) or {}
    raw = recorder.get("buttons", cc.DEFAULT_TELEOP_BUTTON_OUTCOMES)
    if not isinstance(raw, dict):
        raise ValueError("recorder.buttons must be a mapping of '<side>.<index>' to success/fail/discard")

    resolved: Dict[str, str] = {}
    for raw_key, raw_outcome in raw.items():
        key = str(raw_key).strip().lower()
        outcome = str(raw_outcome).strip().lower()
        try:
            side, raw_index = key.rsplit(".", 1)
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid recorder button key {raw_key!r}; expected '<side>.<index>'") from exc
        if side not in {"left", "right"} or index < 0:
            raise ValueError(f"invalid recorder button key {raw_key!r}; expected left/right and a non-negative index")
        if outcome not in _TELEOP_OUTCOMES:
            raise ValueError(
                f"invalid outcome {raw_outcome!r} for recorder button {raw_key!r}; "
                "expected success, fail, or discard"
            )
        if key in resolved:
            raise ValueError(f"duplicate recorder button after normalization: {raw_key!r}")
        resolved[key] = outcome
    return resolved


def apply_camera_serials(cameras: List[Any], rig: Dict[str, Any]) -> List[Any]:
    """Set each ``CameraSpec.serial`` from ``rig['cameras']`` (by camera key)."""
    section = (rig or {}).get("cameras", {}) or {}
    for cam in cameras:
        if section.get(cam.key):
            cam.serial = str(section[cam.key])
    return cameras


class Resolver:
    """Resolve a value with precedence: explicit CLI flag > rig section > default."""

    def __init__(self, args: Any, parser: Any, section: Optional[Dict[str, Any]]) -> None:
        self.args = args
        self.parser = parser
        self.section = section or {}

    def get(self, arg: str, key: Optional[str] = None) -> Any:
        key = key or arg
        cli = getattr(self.args, arg, None)
        if cli is not None and cli != self.parser.get_default(arg):
            return cli  # user set it explicitly
        if key in self.section and self.section[key] is not None:
            return self.section[key]
        return cli  # the argparse default
