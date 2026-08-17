"""Splicing agentview calibration results into config.yaml -- same technique, and same test style,
as tests/test_exposure_tuner.py's splice_camera_options tests: real config.yaml-shaped text in,
byte-for-byte-except-the-touched-block text out.

No cv2/mujoco needed here at all -- this is pure text manipulation, deliberately kept
independent of the detection/FK machinery the rest of charuco.py needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from workstation.lerobot_recorder.charuco import (
    AgentviewEyeToHandResult,
    BoardSpec,
    format_extrinsic_yaml,
    splice_agentview_extrinsic,
    splice_board,
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


def _result(t=(1.0, 0.2, 0.5)) -> AgentviewEyeToHandResult:
    m = np.eye(4)
    m[:3, 3] = t
    return AgentviewEyeToHandResult(
        arm="left",
        base_t_agentview=m,
        flange_t_board=np.eye(4),
        n_captures=4,
        translation_rms_mm=1.234,
        rotation_rms_deg=0.5678,
        per_capture_translation_mm=[1.0, 1.5],
        per_capture_rotation_deg=[0.5, 0.6],
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
    assert 'method: "eye_to_hand"' in out  # provenance: how it was produced


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


def test_the_written_matrix_round_trips_through_yaml():
    yaml = pytest.importorskip("yaml")
    out = splice_agentview_extrinsic(CONFIG, "left", _result((1.5, -0.25, 0.75)), calibrated_at="t")
    parsed = yaml.safe_load(out)
    matrix = parsed["cameras"]["agentview"]["extrinsic"]["left"]["matrix"]
    got = np.asarray(matrix, dtype=np.float64)
    assert np.allclose(got[:3, 3], [1.5, -0.25, 0.75], atol=1e-6)
    # the rest of the file must still parse -- e.g. camera serials survived as strings
    assert parsed["cameras"]["agentview"]["serial"] == "246322303794"
    assert parsed["robot"]["host"] == "192.168.0.42"


def test_expands_a_scalar_agentview_camera_so_it_can_take_an_extrinsic_child():
    """A camera written in the shorthand `agentview: "serial"` form cannot carry child keys; the
    splice must convert it to the mapping form first, or it would produce invalid YAML."""
    yaml = pytest.importorskip("yaml")
    scalar = 'cameras:\n  agentview: "246322303794"\n'
    out = splice_agentview_extrinsic(scalar, "left", _result(), calibrated_at="t")
    parsed = yaml.safe_load(out)
    assert parsed["cameras"]["agentview"]["serial"] == "246322303794"  # serial survived the expand
    assert "matrix" in parsed["cameras"]["agentview"]["extrinsic"]["left"]


def test_refuses_when_agentview_is_missing_from_the_file():
    broken = "robot:\n  host: x\ncameras:\n  wrist_left: y\n"
    with pytest.raises(ValueError, match="agentview"):
        splice_agentview_extrinsic(broken, "left", _result(), calibrated_at="t")


def test_preserves_trailing_newline_state():
    # CONFIG itself ends with "\n" -- splicing it must keep that (only add a trailing newline if
    # the input had one).
    assert splice_agentview_extrinsic(CONFIG, "left", _result(), calibrated_at="t").endswith("\n")
    no_trailing = CONFIG.rstrip("\n")
    assert not splice_agentview_extrinsic(no_trailing, "left", _result(), calibrated_at="t").endswith("\n")


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
    lines = format_extrinsic_yaml(_result(), calibrated_at="2026-01-01 00:00:00", indent=8, method="eye_to_hand")
    text = "\n".join(lines)
    for expected in ("matrix:", 'method: "eye_to_hand"', "n_captures: 4", "translation_rms_mm: 1.234"):
        assert expected in text
    assert all(ln.startswith(" " * 8) for ln in lines if ln.strip() and not ln.strip().startswith("-"))
