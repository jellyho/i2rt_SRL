"""Capture triggers: leader-handle button edge detection, and the convergence-trend text.

The board-detection/solve math itself is tests/test_charuco.py's job; this covers the GUI-level
wiring on top of it -- specifically the two things that are easy to get subtly wrong and hard to
notice by eye (a level check firing every tick instead of once per press; a "converged" verdict
that never actually needs the last few solves to agree).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from workstation.lerobot_recorder.calibrate_agentview_gui import CalibrateAgentviewWindow, _convergence_note
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.charuco import BoardSpec, Detection
from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeGeometry:
    def camera_pose(self, q):
        return np.eye(4)


@pytest.fixture
def window(qapp):
    cfg = RecorderConfig(cameras=[CameraSpec("agentview", "", 64, 64, 5)], mock=True)
    cams = CameraManager(cfg)
    cams.start()
    win = CalibrateAgentviewWindow(
        cams,
        object(),  # non-None "robot" -- only _check_capture_button's obs dict matters here
        {"left": _FakeGeometry(), "right": _FakeGeometry()},
        board=BoardSpec(),
        out_path="/tmp/test_calibrate_agentview.json",
        mock=False,  # False so the button path is actually exercised (True short-circuits it)
    )
    det = Detection(cam_t_board=np.eye(4), n_corners=10, reproj_error_px=0.2, corners_px=np.zeros((10, 2)))
    win._last_agent_det = det
    win._last_wrist_det = {"left": det, "right": det}
    win._last_q = {"left": np.zeros(7), "right": np.zeros(7)}
    win.capture_btn.setEnabled(True)
    yield win
    cams.stop()


def _obs(left=False, right=False) -> dict:
    return {"left": {"buttons": [0, int(left)]}, "right": {"buttons": [0, int(right)]}}


# --------------------------------------------------------------------------------------- #
# _check_capture_button: rising-edge detection, not a level check
# --------------------------------------------------------------------------------------- #
def test_button_up_does_not_capture(window):
    window._check_capture_button(_obs())
    assert len(window.captures) == 0


def test_rising_edge_captures_exactly_once(window):
    window._check_capture_button(_obs())  # up
    window._check_capture_button(_obs(left=True))  # rising edge
    # one capture per ready arm (left + right), from the single press
    assert len(window.captures) == 2


def test_holding_the_button_does_not_capture_again(window):
    window._check_capture_button(_obs(left=True))  # rising edge -> captures
    n_after_first = len(window.captures)
    window._check_capture_button(_obs(left=True))  # still held
    window._check_capture_button(_obs(left=True))  # still held
    assert len(window.captures) == n_after_first


def test_release_then_re_press_captures_again(window):
    window._check_capture_button(_obs(left=True))  # rising edge -> captures
    n_after_first = len(window.captures)
    window._check_capture_button(_obs())  # release
    window._check_capture_button(_obs(left=True))  # rising edge again
    assert len(window.captures) == n_after_first + 2


def test_either_configured_button_fires_the_same_single_edge(window):
    """Pressing left AND right at once must still fire once, not twice -- the whole configured
    set is one aggregate signal (see _check_capture_button's docstring), not one edge test per
    button, or a two-handed press would double-capture the same instant."""
    window._check_capture_button(_obs())  # up
    window._check_capture_button(_obs(left=True, right=True))  # both edges at once
    assert len(window.captures) == 2  # one press's worth (left-arm + right-arm captures), not 4


def test_a_button_not_in_capture_buttons_is_ignored(window):
    window.capture_buttons = ["left.0"]  # "upper", not "lower" -- obs() only sets index 1
    window._check_capture_button(_obs())
    window._check_capture_button(_obs(left=True))  # sets buttons[1], not buttons[0]
    assert len(window.captures) == 0


def test_an_empty_capture_buttons_list_disables_the_trigger_entirely(window):
    window.capture_buttons = []
    window._check_capture_button(_obs(left=True))
    assert len(window.captures) == 0


def test_capture_button_respects_the_readiness_gate(window):
    """A rising edge while the button is disabled (board not seen, robot link down, ...) must
    not capture -- same gate Space and the on-screen click go through."""
    window.capture_btn.setEnabled(False)
    window._check_capture_button(_obs(left=True))
    assert len(window.captures) == 0


# --------------------------------------------------------------------------------------- #
# _convergence_note
# --------------------------------------------------------------------------------------- #
def test_needs_at_least_two_entries_to_say_anything():
    assert "need more" in _convergence_note([])
    assert "need more" in _convergence_note([(2, 10.0, 0.1)])


def test_flat_recent_history_reads_as_converged():
    history = [(2, 10.0, 0.1), (3, 9.0, 0.1), (4, 8.9, 0.09), (5, 8.95, 0.1)]
    note = _convergence_note(history)
    assert "converged" in note
    assert "keep capturing" not in note


def test_still_dropping_history_does_not_falsely_claim_convergence():
    history = [(2, 40.0, 1.0), (3, 25.0, 0.6), (4, 15.0, 0.3)]
    note = _convergence_note(history)
    assert "still moving" in note


def test_the_trend_string_shows_the_actual_numbers_in_order():
    history = [(2, 10.0, 0.0), (3, 5.5, 0.0)]
    assert "10.0 -> 5.5" in _convergence_note(history)
