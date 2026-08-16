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
    BoardSpec,
    CalibrationResult,
    UnifiedRigCalibration,
    WristExtrinsicResult,
    format_arm_offset_yaml,
    format_extrinsic_yaml,
    format_unified_yaml,
    splice_agentview_extrinsic,
    splice_agentview_unified,
    splice_arm_offset,
    splice_board,
    splice_wrist_extrinsic,
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


def _unified(t=(1.2, 0.1, 0.4), cross_t=0.9, cross_r=0.15) -> UnifiedRigCalibration:
    m = np.eye(4)
    m[:3, 3] = t
    return UnifiedRigCalibration(
        left_t_agentview=m,
        left_t_right=np.eye(4),
        distance_m=0.6,
        cross_check_translation_mm=cross_t,
        cross_check_rotation_deg=cross_r,
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
# splice_agentview_unified -- the fused answer, safe to store alongside left/right (see its
# own docstring for why: every save recomputes and writes all three together)
# --------------------------------------------------------------------------------------- #
def test_unified_sits_alongside_left_and_right_not_in_place_of_them():
    out = splice_agentview_extrinsic(CONFIG, "left", _result((1.0, 0.0, 0.0)), calibrated_at="t1")
    out = splice_agentview_extrinsic(out, "right", _result((-1.0, 0.0, 0.0)), calibrated_at="t2")
    out = splice_agentview_unified(out, _unified(), calibrated_at="t3")
    assert "left:" in out and "right:" in out and "unified:" in out
    assert '"t1"' in out and '"t2"' in out and '"t3"' in out


def test_unified_works_even_when_extrinsic_did_not_exist_yet():
    """A rig calibrated in one go (both arms + offset, never saved individually first) must
    still be able to write `unified` -- it should not require left/right to already be there."""
    out = splice_agentview_unified(CONFIG, _unified(), calibrated_at="t")
    assert "unified:" in out


def test_re_solving_replaces_unified_not_duplicates():
    step1 = splice_agentview_unified(CONFIG, _unified(), calibrated_at="t1")
    step2 = splice_agentview_unified(step1, _unified(), calibrated_at="t2")
    assert sum(1 for ln in step2.splitlines() if ln.strip().startswith("unified:")) == 1
    assert '"t1"' not in step2 and '"t2"' in step2


def test_unified_does_not_duplicate_left_t_right_or_distance_m():
    """robot.arm_offset already carries these -- the unified block should carry only what is
    NOT available anywhere else (the fused matrix + the cross-check numbers)."""
    out = splice_arm_offset(CONFIG, _offset(), calibrated_at="t")
    out = splice_agentview_unified(out, _unified(), calibrated_at="t")
    unified_lines = format_unified_yaml(_unified(), calibrated_at="t")
    assert not any("distance_m" in ln for ln in unified_lines)


def test_unified_round_trips_through_yaml_with_cross_check_numbers():
    yaml = pytest.importorskip("yaml")
    out = splice_agentview_unified(CONFIG, _unified((2.0, -0.5, 0.3), cross_t=1.234, cross_r=0.056), calibrated_at="t")
    parsed = yaml.safe_load(out)
    node = parsed["cameras"]["agentview"]["extrinsic"]["unified"]
    got = np.asarray(node["matrix"], dtype=np.float64)
    assert np.allclose(got[:3, 3], [2.0, -0.5, 0.3], atol=1e-6)
    assert node["cross_check_translation_mm"] == pytest.approx(1.234, abs=1e-3)
    assert node["cross_check_rotation_deg"] == pytest.approx(0.056, abs=1e-4)


def test_unified_omits_cross_check_fields_when_none():
    """A rig calibrated from only one arm's captures has no independent second route to cross-
    check against -- the field should be absent, not written as null/0, which would read as a
    real (and reassuringly small) measurement that was never actually taken."""
    lines = format_unified_yaml(UnifiedRigCalibration(np.eye(4), np.eye(4), 0.0, None, None), calibrated_at="t")
    text = "\n".join(lines)
    assert "cross_check" not in text


# --------------------------------------------------------------------------------------- #
# splice_wrist_extrinsic -- the hand-eye mount, under cameras.wrist_<arm>.extrinsic
# --------------------------------------------------------------------------------------- #
def _wrist(t=(0.02, -0.03, 0.05)) -> WristExtrinsicResult:
    m = np.eye(4)
    m[:3, 3] = t
    return WristExtrinsicResult(
        arm="left",
        gripper_t_camera=m,
        n_captures=8,
        translation_rms_mm=0.4,
        rotation_rms_deg=0.05,
        per_capture_translation_mm=[0.3, 0.5],
        per_capture_rotation_deg=[0.04, 0.06],
    )


def test_wrist_extrinsic_expands_a_scalar_camera_and_adds_the_child():
    """CONFIG's wrist_left is the scalar shorthand ("<serial>"); the splice must convert it to a
    mapping (serial preserved) so an extrinsic child is valid YAML, not a scalar with children."""
    yaml = pytest.importorskip("yaml")
    out = splice_wrist_extrinsic(CONFIG, "left", _wrist((0.01, 0.02, 0.03)), calibrated_at="t")
    parsed = yaml.safe_load(out)
    node = parsed["cameras"]["wrist_left"]
    assert node["serial"] == "352122271652"  # serial survived the scalar->mapping expansion
    got = np.asarray(node["extrinsic"]["matrix"], dtype=np.float64)
    assert np.allclose(got[:3, 3], [0.01, 0.02, 0.03], atol=1e-6)


def test_wrist_extrinsic_works_on_a_mapping_form_camera_too():
    yaml = pytest.importorskip("yaml")
    out = splice_wrist_extrinsic(CONFIG, "left", _wrist(), calibrated_at="t")  # agentview is mapping-form
    out = splice_wrist_extrinsic(out, "left", _wrist((0.09, 0.0, 0.0)), calibrated_at="t2")  # now wrist_left is too
    parsed = yaml.safe_load(out)
    got = np.asarray(parsed["cameras"]["wrist_left"]["extrinsic"]["matrix"], dtype=np.float64)
    assert np.allclose(got[:3, 3], [0.09, 0.0, 0.0], atol=1e-6)  # re-splice replaced, not duplicated


def test_wrist_extrinsic_leaves_the_other_wrist_and_agentview_alone():
    out = splice_wrist_extrinsic(CONFIG, "left", _wrist(), calibrated_at="t")
    assert "409122274199" in out  # wrist_right serial untouched
    assert "246322303794" in out  # agentview serial untouched


# --------------------------------------------------------------------------------------- #
# splice_board -- the ChArUco geometry, under a top-level calibration.board
# --------------------------------------------------------------------------------------- #
def test_board_round_trips_through_config_and_from_config():
    yaml = pytest.importorskip("yaml")
    spec = BoardSpec(squares_x=7, squares_y=5, square_length_m=0.025, marker_length_m=0.018, dictionary="DICT_5X5_100")
    out = splice_board(CONFIG, spec, calibrated_at="t")
    parsed = yaml.safe_load(out)
    assert BoardSpec.from_config(parsed["calibration"]["board"]) == spec


def test_board_creates_the_calibration_section_when_absent_then_reuses_it():
    out = splice_board(CONFIG, BoardSpec(), calibrated_at="t1")
    out = splice_board(out, BoardSpec(squares_x=9), calibrated_at="t2")
    assert sum(1 for ln in out.splitlines() if ln.strip().startswith("calibration:")) == 1
    assert sum(1 for ln in out.splitlines() if ln.strip().startswith("board:")) == 1
    assert '"t1"' not in out and '"t2"' in out


def test_board_omits_calibrated_at_when_not_given():
    """A caller that just wants to persist the geometry (not record a solve) should not get a
    misleading timestamp."""
    out = splice_board(CONFIG, BoardSpec())
    assert "calibrated_at" not in out.split("board:", 1)[1]


def test_from_config_falls_back_to_defaults_for_missing_keys():
    partial = BoardSpec.from_config({"squares_x": 10})
    assert partial.squares_x == 10
    assert partial.squares_y == BoardSpec().squares_y  # the rest defaulted
    assert BoardSpec.from_config(None) == BoardSpec()  # absent block -> all defaults


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
