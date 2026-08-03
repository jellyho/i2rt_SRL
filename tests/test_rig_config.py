"""Unified rig config: load, control overrides, camera serials, CLI>YAML>default."""

from __future__ import annotations

import argparse

import pytest

import i2rt.serving.rig_config as rc
from i2rt.serving import control_config as cc
from i2rt.serving.rig_config import (
    Resolver,
    apply_camera_serials,
    apply_control_overrides,
    find_rig,
    load_rig,
    teleop_button_outcomes,
)
from workstation.lerobot_recorder.__main__ import build_config
from workstation.lerobot_recorder.config import default_cameras
from workstation.lerobot_recorder.deploy_main import build_configs as build_deploy_configs


def _no_repo_rig(monkeypatch, tmp_path):
    """Point the repo root at an empty dir so no in-repo config.yaml is found."""
    monkeypatch.setattr(rc, "_repo_root", lambda: str(tmp_path / "norepo"))


def test_load_rig(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("robot:\n  host: 1.2.3.4\n  port: 9\ntasks:\n  - a\n  - b\n")
    rig = load_rig(str(p))
    assert rig["robot"] == {"host": "1.2.3.4", "port": 9}
    assert rig["tasks"] == ["a", "b"]
    _no_repo_rig(monkeypatch, tmp_path)
    assert load_rig(None) == {}  # no in-repo rig, no env -> empty


def test_record_and_deploy_share_video_backend_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
recorder:
  encoding_backend: pyav
  torchcodec_max_used_vram_gb: 6.5
  streaming_encoding: true
"""
    )
    record_cfg = build_config(["--config", str(config), "--mock"])
    deploy_cfg, _ = build_deploy_configs(["--config", str(config), "--mock"])

    for cfg in (record_cfg, deploy_cfg):
        assert cfg.encoding_backend == "pyav"
        assert cfg.torchcodec_max_used_vram_gb == 6.5
        assert cfg.streaming_encoding is True


def test_find_rig_in_repo(tmp_path, monkeypatch):
    rr = tmp_path / "repo"
    rr.mkdir()
    (rr / "config.yaml").write_text("z: 1\n")
    monkeypatch.setattr(rc, "_repo_root", lambda: str(rr))
    assert find_rig() == str(rr / "config.yaml")  # the in-repo config

    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("x: 1\n")
    assert find_rig(str(explicit)) == str(explicit)  # --config override

    _no_repo_rig(monkeypatch, tmp_path)
    assert find_rig() is None


def test_apply_control_overrides(monkeypatch):
    monkeypatch.setattr(cc, "BILATERAL_KP", 0.0, raising=False)
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", None, raising=False)
    monkeypatch.setattr(cc, "FOLLOWER_JOINT_LIMITS", None, raising=False)
    applied = apply_control_overrides(
        {"control": {"bilateral_kp": 0.2, "follower_effort_limit": 25.0, "follower_joint_limits": [[-1, 1], [-2, 2]]}}
    )
    assert cc.BILATERAL_KP == 0.2
    assert cc.FOLLOWER_EFFORT_LIMIT == 25.0
    assert cc.FOLLOWER_JOINT_LIMITS == [(-1, 1), (-2, 2)]  # lists -> tuples
    assert "BILATERAL_KP" in applied
    assert apply_control_overrides({}) == {}


def test_apply_fine_grained_control_overrides(monkeypatch):
    monkeypatch.setattr(cc, "FINE_GRAINED_SCALE", 0.2, raising=False)
    monkeypatch.setattr(cc, "FINE_GRAINED_BUTTON", "left.0", raising=False)
    monkeypatch.setattr(cc, "FINE_RECENTER_SPEED", 0.15, raising=False)
    monkeypatch.setattr(cc, "FINE_RECENTER_MAX_FOLLOWING_ERROR", 0.05, raising=False)
    monkeypatch.setattr(cc, "FINE_RECENTER_TOLERANCE", 0.03, raising=False)
    applied = apply_control_overrides(
        {
            "control": {
                "fine_grained_scale": 0.1,
                "fine_grained_button": "left.0",
                "fine_recenter_speed": 0.12,
                "fine_recenter_max_following_error": 0.04,
                "fine_recenter_tolerance": 0.02,
            }
        }
    )
    assert cc.FINE_GRAINED_SCALE == 0.1
    assert cc.FINE_GRAINED_BUTTON == "left.0"
    assert cc.FINE_RECENTER_SPEED == 0.12
    assert cc.FINE_RECENTER_MAX_FOLLOWING_ERROR == 0.04
    assert cc.FINE_RECENTER_TOLERANCE == 0.02
    assert "FINE_GRAINED_SCALE" in applied


def test_teleop_button_outcomes_are_shared_and_validated():
    assert teleop_button_outcomes({}) == cc.DEFAULT_TELEOP_BUTTON_OUTCOMES
    assert teleop_button_outcomes({"recorder": {"buttons": {"LEFT.2": "SUCCESS"}}}) == {
        "left.2": "success"
    }
    assert teleop_button_outcomes({"recorder": {"buttons": {}}}) == {}

    for buttons in (["left.1"], {"middle.0": "success"}, {"left.0": "stop"}):
        try:
            teleop_button_outcomes({"recorder": {"buttons": buttons}})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid button mapping was accepted: {buttons!r}")


def test_deprecated_home_buttons_fail_fast():
    try:
        apply_control_overrides({"control": {"home_buttons": ["left.1"]}})
    except ValueError as exc:
        assert "recorder.buttons" in str(exc)
    else:
        raise AssertionError("deprecated control.home_buttons was accepted")


def test_apply_control_overrides_leader_keys(monkeypatch):
    for attr in ("LEADER_GRAVITY_COMP_FACTOR", "LEADER_GRAV_COMP_KD", "LEADER_COULOMB_FRICTION"):
        monkeypatch.setattr(cc, attr, None, raising=False)
    applied = apply_control_overrides(
        {
            "control": {
                "leader_gravity_comp_factor": 1.1,
                "leader_grav_comp_kd": [0.05, 0.05, 0.05, 0.15, 0.03, 0.03],
                "leader_coulomb_friction": 0.4,
            }
        }
    )
    assert cc.LEADER_GRAVITY_COMP_FACTOR == 1.1
    assert cc.LEADER_GRAV_COMP_KD == [0.05, 0.05, 0.05, 0.15, 0.03, 0.03]
    assert cc.LEADER_COULOMB_FRICTION == 0.4
    assert "LEADER_GRAVITY_COMP_FACTOR" in applied


def test_apply_camera_serials():
    cams = apply_camera_serials(default_cameras(), {"cameras": {"agentview": "AAA", "wrist_left": "BBB"}})
    by = {c.key: c.serial for c in cams}
    assert by["agentview"] == "AAA" and by["wrist_left"] == "BBB"
    assert by["wrist_right"] == ""  # untouched


def test_apply_camera_options():
    cams = apply_camera_serials(
        default_cameras(),
        {
            "cameras": {
                "agentview": {"serial": "AAA", "options": {"enable_auto_exposure": 0, "exposure": 300}},
                "wrist_left": "BBB",  # plain-string form still supported alongside
            }
        },
    )
    by = {c.key: c for c in cams}
    assert by["agentview"].serial == "AAA"
    assert by["agentview"].options == {"enable_auto_exposure": 0.0, "exposure": 300.0}
    assert by["wrist_left"].serial == "BBB" and by["wrist_left"].options == {}


def test_apply_camera_options_bool_and_serial_only_map():
    cams = apply_camera_serials(
        default_cameras(),
        {"cameras": {"agentview": {"options": {"enable_auto_exposure": False}}}},
    )
    by = {c.key: c for c in cams}
    assert by["agentview"].serial == ""  # no serial in the map -> left unpinned
    assert by["agentview"].options == {"enable_auto_exposure": 0.0}


def test_apply_camera_options_rejects_bad_shapes():
    with pytest.raises(ValueError):
        apply_camera_serials(default_cameras(), {"cameras": {"agentview": {"options": ["exposure"]}}})
    with pytest.raises(ValueError):
        apply_camera_serials(default_cameras(), {"cameras": {"agentview": {"options": {"exposure": "dark"}}}})


def test_resolver_precedence():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dflt")

    r_yaml = Resolver(p.parse_args([]), p, {"root": "yaml"})
    assert r_yaml.get("root") == "yaml"  # YAML beats the default

    r_cli = Resolver(p.parse_args(["--root", "cli"]), p, {"root": "yaml"})
    assert r_cli.get("root") == "cli"  # explicit CLI beats YAML

    r_def = Resolver(p.parse_args([]), p, {})
    assert r_def.get("root") == "dflt"  # nothing -> default


def test_resolver_key_alias():
    p = argparse.ArgumentParser()
    p.add_argument("--robot-host", default="127.0.0.1")
    r = Resolver(p.parse_args([]), p, {"host": "10.0.0.5"})
    assert r.get("robot_host", key="host") == "10.0.0.5"


def test_recorder_config_uses_first_yaml_task_as_active_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("tasks:\n  - pick the cube\n  - stack the blocks\n")

    cfg = build_config(["--config", str(cfg_path)])

    assert cfg.tasks == ["pick the cube", "stack the blocks"]
    assert cfg.task == "pick the cube"


def test_recorder_config_task_precedence(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "recorder:\n"
        "  task: open the drawer\n"
        "tasks:\n"
        "  - pick the cube\n"
        "  - stack the blocks\n"
    )

    cfg = build_config(["--config", str(cfg_path)])
    assert cfg.task == "open the drawer"

    cfg = build_config(["--config", str(cfg_path), "--task", "close the drawer"])
    assert cfg.task == "close the drawer"


def test_recorder_config_uses_shared_button_outcomes(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("recorder:\n  buttons:\n    left.2: success\n")

    cfg = build_config(["--config", str(cfg_path)])

    assert cfg.button_map == {"left.2": "success"}
