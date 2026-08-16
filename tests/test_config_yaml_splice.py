"""Splicing calibration results into config.yaml -- same technique, and same test style, as
tests/test_exposure_tuner.py's splice_camera_options tests: real config.yaml-shaped text in,
byte-for-byte-except-the-touched-block text out.

No cv2/mujoco needed here at all -- this is pure text manipulation, deliberately kept
independent of the detection/FK machinery the rest of charuco.py needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from workstation.lerobot_recorder.charuco import (
    ArmOffsetResult,
    CalibrationResult,
    format_arm_offset_yaml,
    format_extrinsic_yaml,
    splice_agentview_extrinsic,
    splice_arm_offset,
)

CONFIG = """\
robot:
  host: 192.168.0.42
  port: 11331
  arm_type: yam

cameras:                     # serials by role
  agentview:                             # D455
    serial: "246322303794"
    fps: 30
    options:
      enable_auto_exposure: 0
      exposure: 115
  wrist_left: "352122271652"             # D405
  wrist_right: "409122274199"

recorder:
  fps: 30
"""


def _result(t=(1.0, 0.2, 0.5)) -> CalibrationResult:
    m = np.eye(4)
    m[:3, 3] = t
    return CalibrationResult(
        arm="left",
        base_t_agentview=m,
        n_captures=4,
        translation_rms_mm=1.234,
        rotation_rms_deg=0.5678,
        per_capture_translation_mm=[1.0, 1.5],
        per_capture_rotation_deg=[0.5, 0.6],
    )


def _offset(t=(0.6, 0.0, 0.0)) -> ArmOffsetResult:
    m = np.eye(4)
    m[:3, 3] = t
    return ArmOffsetResult(
        left_t_right=m,
        distance_m=float(np.linalg.norm(t)),
        n_captures=3,
        translation_rms_mm=0.8,
        rotation_rms_deg=0.2,
        per_capture_translation_mm=[0.5, 0.9],
        per_capture_rotation_deg=[0.1, 0.2],
    )


# --------------------------------------------------------------------------------------- #
# splice_agentview_extrinsic
# --------------------------------------------------------------------------------------- #
def test_inserts_a_new_extrinsic_section_when_none_exists():
    out = splice_agentview_extrinsic(CONFIG, "left", _result(), calibrated_at="2026-08-17 10:00:00")
    assert "extrinsic:" in out
    assert "left:" in out
    assert "n_captures: 4" in out
    assert "translation_rms_mm: 1.234" in out
    assert '"2026-08-17 10:00:00"' in out


def test_leaves_the_rest_of_the_file_untouched_including_comments():
    out = splice_agentview_extrinsic(CONFIG, "left", _result(), calibrated_at="t")
    for line in ("# D405", "# serials by role", "host: 192.168.0.42", "fps: 30"):
        assert line in out


def test_adding_the_right_arm_does_not_erase_the_previously_saved_left_arm():
    """The failure mode this exists to prevent: a naive 'replace the whole extrinsic: block'
    would silently drop whichever arm was not just captured."""
    step1 = splice_agentview_extrinsic(CONFIG, "left", _result((1.0, 0.0, 0.0)), calibrated_at="t1")
    step2 = splice_agentview_extrinsic(step1, "right", _result((-1.0, 0.0, 0.0)), calibrated_at="t2")
    assert "left:" in step2 and "right:" in step2
    assert '"t1"' in step2  # left's timestamp survived the right-arm write
    assert '"t2"' in step2


def test_re_capturing_the_same_arm_replaces_it_not_duplicates_it():
    step1 = splice_agentview_extrinsic(CONFIG, "left", _result(), calibrated_at="t1")
    step2 = splice_agentview_extrinsic(step1, "left", _result(), calibrated_at="t2")
    # count the "left:" ARM key specifically, not "wrist_left:" (which also ends in "left:")
    assert sum(1 for ln in step2.splitlines() if ln.strip() == "left:") == 1
    assert '"t1"' not in step2
    assert '"t2"' in step2


def test_the_written_matrix_round_trips_through_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    out = splice_agentview_extrinsic(CONFIG, "left", _result((1.5, -0.25, 0.75)), calibrated_at="t")
    parsed = yaml.safe_load(out)
    matrix = parsed["cameras"]["agentview"]["extrinsic"]["left"]["matrix"]
    got = np.asarray(matrix, dtype=np.float64)
    assert np.allclose(got[:3, 3], [1.5, -0.25, 0.75], atol=1e-6)
    assert np.allclose(got, np.eye(4), atol=1e-6, rtol=0) is False  # sanity: not silently identity
    # the rest of the file must still parse -- e.g. camera serials survived as strings
    assert parsed["cameras"]["agentview"]["serial"] == "246322303794"
    assert parsed["robot"]["host"] == "192.168.0.42"


def test_refuses_when_agentview_is_missing_from_the_file():
    broken = "robot:\n  host: x\ncameras:\n  wrist_left: y\n"
    with pytest.raises(ValueError, match="agentview"):
        splice_agentview_extrinsic(broken, "left", _result(), calibrated_at="t")


def test_preserves_trailing_newline_state():
    # CONFIG itself ends with "\n" -- splicing it must keep that (matches
    # splice_camera_options's own convention: only add a trailing newline if the input had one).
    assert splice_agentview_extrinsic(CONFIG, "left", _result(), calibrated_at="t").endswith("\n")
    # and an input with NO trailing newline must not gain one either -- "preserve", not "add".
    no_trailing = CONFIG.rstrip("\n")
    assert not splice_agentview_extrinsic(no_trailing, "left", _result(), calibrated_at="t").endswith("\n")


# --------------------------------------------------------------------------------------- #
# splice_arm_offset
# --------------------------------------------------------------------------------------- #
def test_inserts_arm_offset_under_the_top_level_robot_section():
    out = splice_arm_offset(CONFIG, _offset(), calibrated_at="t")
    assert "arm_offset:" in out
    assert "distance_m: 0.6000" in out
    # still a sibling of the other robot: keys, not swallowing them
    assert "host: 192.168.0.42" in out
    assert "arm_type: yam" in out


def test_re_solving_the_offset_replaces_not_duplicates():
    step1 = splice_arm_offset(CONFIG, _offset(), calibrated_at="t1")
    step2 = splice_arm_offset(step1, _offset(), calibrated_at="t2")
    assert step2.count("arm_offset:") == 1
    assert '"t2"' in step2 and '"t1"' not in step2


def test_arm_offset_round_trips_through_yaml():
    yaml = pytest.importorskip("yaml")
    out = splice_arm_offset(CONFIG, _offset((0.3, 0.4, 0.0)), calibrated_at="t")
    parsed = yaml.safe_load(out)
    got = np.asarray(parsed["robot"]["arm_offset"]["matrix"], dtype=np.float64)
    assert np.allclose(got[:3, 3], [0.3, 0.4, 0.0], atol=1e-6)
    assert parsed["robot"]["arm_offset"]["distance_m"] == pytest.approx(0.5, abs=1e-3)


def test_refuses_when_robot_section_is_missing():
    with pytest.raises(ValueError, match="robot"):
        splice_arm_offset("cameras:\n  agentview: {}\n", _offset(), calibrated_at="t")


# --------------------------------------------------------------------------------------- #
# format_* helpers, in isolation
# --------------------------------------------------------------------------------------- #
def test_format_extrinsic_yaml_includes_every_field():
    lines = format_extrinsic_yaml(_result(), calibrated_at="2026-01-01 00:00:00", indent=8)
    text = "\n".join(lines)
    for expected in ("matrix:", "n_captures: 4", "translation_rms_mm: 1.234", "rotation_rms_deg: 0.5678"):
        assert expected in text
    assert all(ln.startswith(" " * 8) for ln in lines if ln.strip() and not ln.strip().startswith("-"))


def test_format_arm_offset_yaml_includes_distance():
    lines = format_arm_offset_yaml(_offset(), calibrated_at="t", indent=4)
    assert any("distance_m:" in ln for ln in lines)
