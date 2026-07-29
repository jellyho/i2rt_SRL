"""Exposure tuner logic: luma metrics, exposure suggestion, config.yaml splicing."""

from __future__ import annotations

import numpy as np
import pytest

from workstation.lerobot_recorder.exposure_tuner import (
    brightness_report,
    clipped_fraction,
    format_options_yaml,
    mean_luma,
    same_model_groups,
    splice_camera_options,
    suggest_exposure,
)


def _gray(value, h=8, w=8):
    return np.full((h, w, 3), value, np.uint8)


# ------------------------------------------------------------------ luma / clipping
def test_mean_luma_of_flat_gray_is_that_gray():
    assert mean_luma(_gray(128)) == pytest.approx(128.0, abs=0.5)
    assert mean_luma(_gray(0)) == 0.0


def test_mean_luma_weights_green_most():
    red = np.zeros((4, 4, 3), np.uint8); red[..., 0] = 255
    green = np.zeros((4, 4, 3), np.uint8); green[..., 1] = 255
    blue = np.zeros((4, 4, 3), np.uint8); blue[..., 2] = 255
    assert mean_luma(green) > mean_luma(red) > mean_luma(blue)


def test_mean_luma_handles_empty():
    assert mean_luma(None) == 0.0
    assert mean_luma(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_clipped_fraction_counts_blown_highlights():
    frame = _gray(255)
    assert clipped_fraction(frame) == 1.0
    assert clipped_fraction(_gray(100)) == 0.0
    half = np.concatenate([_gray(255, h=4), _gray(10, h=4)], axis=0)
    assert clipped_fraction(half) == pytest.approx(0.5)


def test_brightness_report_deltas_against_reference():
    frames = {"agentview": _gray(200), "wrist_left": _gray(100), "wrist_right": _gray(100)}
    rep = brightness_report(frames, reference="wrist_left")
    assert rep["wrist_left"]["delta"] == pytest.approx(0.0, abs=0.5)
    assert rep["agentview"]["delta"] == pytest.approx(100.0, abs=1.0)


def test_brightness_report_without_reference_has_no_deltas():
    rep = brightness_report({"agentview": _gray(120)}, reference=None)
    assert rep["agentview"]["delta"] is None
    rep_missing = brightness_report({"agentview": _gray(120)}, reference="nope")
    assert rep_missing["agentview"]["delta"] is None


# ------------------------------------------------------------------- suggestion loop
def test_suggest_exposure_raises_when_too_dark_and_lowers_when_too_bright():
    assert suggest_exposure(100, luma=50, target=100, rng=(1, 10000)) > 100
    assert suggest_exposure(100, luma=200, target=100, rng=(1, 10000)) < 100


def test_suggest_exposure_clamps_to_range():
    assert suggest_exposure(9000, luma=10, target=200, rng=(1, 10000)) <= 10000
    assert suggest_exposure(2, luma=250, target=10, rng=(1, 10000)) >= 1


def test_suggest_exposure_survives_black_frame():
    out = suggest_exposure(100, luma=0.0, target=120, rng=(1, 10000))
    assert out > 100 and out <= 10000  # no div-by-zero, no infinity


def test_suggest_exposure_converges_toward_target():
    """Simulated linear sensor: repeated suggestions should approach the target."""
    exposure, k = 100.0, 1.0  # luma = k * exposure
    for _ in range(12):
        luma = min(255.0, k * exposure)
        exposure = suggest_exposure(exposure, luma, target=120.0, rng=(1, 10000))
    assert min(255.0, k * exposure) == pytest.approx(120.0, rel=0.05)


# ------------------------------------------------------------------------- yaml emit
def test_format_options_yaml_shape():
    out = format_options_yaml("agentview", {"serial": "AAA", "enable_auto_exposure": 0.0, "exposure": 300.0})
    assert "agentview:" in out and 'serial: "AAA"' in out
    assert "enable_auto_exposure: 0" in out and "exposure: 300" in out
    assert "300.0" not in out  # integral values print cleanly


# ---------------------------------------------------------------------- yaml splicing
CONFIG = """\
robot:
  host: 1.2.3.4

cameras:                     # serials by role
  agentview:                             # D455
    serial: "246322303794"
    options:
      enable_auto_exposure: 0
      exposure: 300
  wrist_left: "352122271652"             # D405
  wrist_right: "409122274199"

recorder:
  fps: 30
"""


def test_splice_replaces_existing_options_block():
    out = splice_camera_options(CONFIG, "agentview", {"enable_auto_exposure": 0, "exposure": 800, "gain": 50}, serial="246322303794")
    assert "exposure: 800" in out and "gain: 50" in out
    assert "exposure: 300" not in out
    # everything else survives untouched, comments included
    assert "recorder:" in out and "fps: 30" in out
    assert 'wrist_left: "352122271652"             # D405' in out
    assert "# serials by role" in out
    assert "# D455" in out  # the key-line comment is carried over


def test_splice_upgrades_a_plain_string_entry_to_a_map():
    out = splice_camera_options(CONFIG, "wrist_left", {"enable_auto_exposure": 0, "exposure": 12000}, serial="352122271652")
    assert 'serial: "352122271652"' in out
    assert "exposure: 12000" in out
    # sibling entries and the following section are intact
    assert 'wrist_right: "409122274199"' in out
    assert "recorder:" in out
    assert "# D405" in out


def test_splice_preserves_other_cameras_options():
    once = splice_camera_options(CONFIG, "wrist_left", {"exposure": 12000}, serial="352122271652")
    twice = splice_camera_options(once, "wrist_right", {"exposure": 11000}, serial="409122274199")
    assert "exposure: 12000" in twice and "exposure: 11000" in twice
    assert "exposure: 300" in twice  # agentview's original block untouched


def test_splice_is_idempotent():
    once = splice_camera_options(CONFIG, "agentview", {"exposure": 500}, serial="246322303794")
    twice = splice_camera_options(once, "agentview", {"exposure": 500}, serial="246322303794")
    assert once == twice


def test_splice_keeps_trailing_newline():
    assert splice_camera_options(CONFIG, "agentview", {"exposure": 500}).endswith("\n")


def test_splice_rejects_missing_section_or_key():
    with pytest.raises(ValueError):
        splice_camera_options("robot:\n  host: x\n", "agentview", {"exposure": 1})
    with pytest.raises(ValueError):
        splice_camera_options(CONFIG, "nonexistent_cam", {"exposure": 1})


def test_spliced_config_still_parses_and_round_trips():
    yaml = pytest.importorskip("yaml")
    out = splice_camera_options(CONFIG, "agentview", {"enable_auto_exposure": 0, "exposure": 640, "gain": 40}, serial="246322303794")
    parsed = yaml.safe_load(out)
    cam = parsed["cameras"]["agentview"]
    assert cam["serial"] == "246322303794"
    assert cam["options"] == {"enable_auto_exposure": 0, "exposure": 640, "gain": 40}
    # untouched neighbours keep their original scalar form
    assert parsed["cameras"]["wrist_left"] == "352122271652"
    assert parsed["recorder"]["fps"] == 30


# --------------------------------------------------------------- same-model grouping
def test_same_model_groups_pairs_identical_models():
    groups = same_model_groups(
        {
            "wrist_left": "Intel RealSense D405",
            "wrist_right": "Intel RealSense D405",
            "agentview": "Intel RealSense D455",
        }
    )
    assert list(groups) == ["Intel RealSense D405"]
    assert sorted(groups["Intel RealSense D405"]) == ["wrist_left", "wrist_right"]


def test_same_model_groups_ignores_singletons_and_unbound():
    assert same_model_groups({"a": "D455", "b": "D405"}) == {}
    assert same_model_groups({"a": "", "b": ""}) == {}  # unbound cameras never group
    assert same_model_groups({}) == {}


def test_same_model_groups_handles_three_of_a_kind():
    groups = same_model_groups({"a": "D405", "b": "D405", "c": "D405"})
    assert sorted(groups["D405"]) == ["a", "b", "c"]
