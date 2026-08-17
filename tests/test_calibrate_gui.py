"""GUI-level wiring on top of the agentview eye-to-hand calibration.

The board-detection/solve math is tests/test_charuco.py's job; this covers the things easy to get
subtly wrong and hard to notice by eye: a capture-button level check firing every tick instead of
once per press, the hands-free auto-capture state machine, and that Save writes only the agentview
extrinsic (never a wrist mount or arm offset -- those are gone).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from workstation.lerobot_recorder.calibrate_gui import CalibrateAgentviewWindow, _convergence_note
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.charuco import BoardSpec, Detection
from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


class _MovableGeometry:
    """flange_pose returns whatever the test last set -- so a capture records a varied FK pose (an
    identity would not drive the eye-to-hand solve)."""

    def __init__(self):
        self.cur = np.eye(4)

    def flange_pose(self, q):
        return self.cur


@pytest.fixture
def window(qapp):
    cfg = RecorderConfig(cameras=[CameraSpec("agentview", "", 64, 64, 5)], mock=True)
    cams = CameraManager(cfg)
    cams.start()
    win = CalibrateAgentviewWindow(
        cams,
        object(),  # non-None "robot" -- only _check_capture_button's obs dict matters here
        {"left": _MovableGeometry(), "right": _MovableGeometry()},
        board=BoardSpec(),
        config_path=None,  # per-test fixtures set this where the save path matters
        mock=False,  # False so the button path is actually exercised (True short-circuits it)
    )
    win.capture_buttons = ["left.1", "right.1"]  # handle trigger off by default; enable for the tests
    win._last_agent_det = Detection(np.eye(4), 10, 0.2, np.zeros((10, 2)))
    win._last_q = {"left": np.zeros(7), "right": np.zeros(7)}
    win.capture_btn.setEnabled(True)
    yield win
    cams.stop()


def _obs(left=False, right=False) -> dict:
    return {"left": {"buttons": [0, int(left)]}, "right": {"buttons": [0, int(right)]}}


# --------------------------------------------------------------------------------------- #
# _check_capture_button: rising-edge detection, not a level check (captures the ACTIVE arm)
# --------------------------------------------------------------------------------------- #
def test_button_up_does_not_capture(window):
    window._check_capture_button(_obs())
    assert len(window.captures) == 0


def test_rising_edge_captures_exactly_once(window):
    window._check_capture_button(_obs())  # up
    window._check_capture_button(_obs(left=True))  # rising edge
    assert len(window.captures) == 1  # one capture for the active arm
    assert window.captures[0].arm == "left"


def test_holding_the_button_does_not_capture_again(window):
    window._check_capture_button(_obs(left=True))
    window._check_capture_button(_obs(left=True))
    window._check_capture_button(_obs(left=True))
    assert len(window.captures) == 1


def test_release_then_re_press_captures_again(window):
    window._check_capture_button(_obs(left=True))
    window._check_capture_button(_obs())  # release
    window._check_capture_button(_obs(left=True))  # rising edge again
    assert len(window.captures) == 2


def test_an_empty_capture_buttons_list_disables_the_trigger_entirely(window):
    window.capture_buttons = []
    window._check_capture_button(_obs(left=True))
    assert len(window.captures) == 0


def test_capture_button_respects_the_readiness_gate(window):
    window.capture_btn.setEnabled(False)
    window._check_capture_button(_obs(left=True))
    assert len(window.captures) == 0


def test_capture_attributes_to_the_selected_arm(window):
    window.arm_selector.setCurrentText("right")
    window._check_capture_button(_obs(left=True))  # any configured button fires the ACTIVE arm
    assert window.captures[-1].arm == "right"


# --------------------------------------------------------------------------------------- #
# _auto_capture_check: hands-free hold-still capture (the primary trigger)
# --------------------------------------------------------------------------------------- #
def _hold(window, *, engaged=True, t0=100.0):
    """Hold still across the dwell; return (fired_only_after_dwell, n_captures_after)."""
    window._auto_capture_check(["left"], engaged, t0)  # opens the window
    window._auto_capture_check(["left"], engaged, t0 + window.auto_dwell_s - 0.1)  # not yet
    before = len(window.captures)
    window._auto_capture_check(["left"], engaged, t0 + window.auto_dwell_s + 0.01)  # fires
    return before == 0, len(window.captures)


def test_auto_capture_fires_after_holding_still_through_the_dwell(window):
    fired_only_after_dwell, n = _hold(window)
    assert fired_only_after_dwell  # nothing fired before the dwell elapsed
    assert n == 1


def test_auto_capture_does_not_refire_while_still_held(window):
    _hold(window)
    window._auto_capture_check(["left"], True, 200.0)
    window._auto_capture_check(["left"], True, 205.0)
    assert len(window.captures) == 1


def test_auto_capture_rearms_after_moving_away(window):
    _hold(window)
    window._last_q = {"left": np.full(7, 0.2), "right": np.full(7, 0.2)}
    _, n = _hold(window, t0=300.0)
    assert n == 2  # a second capture after moving away and holding again


def test_auto_capture_does_not_fire_when_not_engaged(window):
    window._auto_capture_check(["left"], False, 100.0)
    window._auto_capture_check(["left"], False, 102.0)
    assert len(window.captures) == 0


def test_auto_capture_off_never_fires(window):
    window.auto_capture = False
    _hold(window)
    assert len(window.captures) == 0


# --------------------------------------------------------------------------------------- #
# _convergence_note
# --------------------------------------------------------------------------------------- #
def test_needs_at_least_two_entries_to_say_anything():
    assert "more captures" in _convergence_note([(3, 5.0, 0.5)])


def test_flat_recent_history_reads_as_converged():
    hist = [(3, 5.0, 0.5), (4, 4.9, 0.5), (5, 4.95, 0.5)]
    assert "converged" in _convergence_note(hist)


def test_still_dropping_history_does_not_falsely_claim_convergence():
    hist = [(3, 20.0, 2.0), (4, 12.0, 1.5), (5, 5.0, 0.8)]
    assert "keep capturing" in _convergence_note(hist)


# --------------------------------------------------------------------------------------- #
# solve + save (agentview eye-to-hand only)
# --------------------------------------------------------------------------------------- #
def _feed(win, arm, x_true, y_true, *, n, seed):
    win.arm_selector.setCurrentText(arm)
    rng = np.random.default_rng(seed)
    geos = win.geometries
    for _ in range(n):
        base_t_flange = _T(
            _rot("x", rng.uniform(-40, 40)) @ _rot("y", rng.uniform(-40, 40)), rng.uniform(-0.2, 0.2, 3)
        )
        geos[arm].cur = base_t_flange
        win._last_agent_det = Detection(np.linalg.inv(x_true) @ base_t_flange @ y_true, 10, 0.2, np.zeros((10, 2)))
        win._last_q = {"left": np.zeros(7), "right": np.zeros(7)}
        win._on_capture()


_X_LEFT = _T(_rot("x", 150.0), [0.4, -0.1, 0.9])
_Y = _T(_rot("y", 30.0), [0.03, 0.0, 0.12])

_CONFIG_YAML = """\
robot:
  host: 192.168.0.42

cameras:
  agentview:
    serial: "246322303794"
    fps: 30
  wrist_left: "352122271652"
  wrist_right: "409122274199"
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG_YAML)
    return path


def _confirm_yes(monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))


def test_solve_recovers_each_arms_agentview_pose(window):
    _feed(window, "left", _X_LEFT, _Y, n=6, seed=1)
    _feed(window, "right", np.linalg.inv(_T(_rot("z", 5.0), [0.03, -0.53, 0.05])) @ _X_LEFT, _Y, n=5, seed=2)
    res = window._last_results
    assert set(res) == {"left", "right"}
    assert np.allclose(res["left"].base_t_agentview, _X_LEFT, atol=1e-6)


def test_save_button_disabled_until_three_captures(window):
    _feed(window, "left", _X_LEFT, _Y, n=2, seed=1)  # 2 < the eye-to-hand minimum of 3
    assert window.save_btn.isEnabled() is False


def test_save_writes_agentview_extrinsic_marked_eye_to_hand(window, config_file, monkeypatch):
    _confirm_yes(monkeypatch)
    window.config_path = str(config_file)
    _feed(window, "left", _X_LEFT, _Y, n=6, seed=1)

    window._on_save()

    written = config_file.read_text()
    assert 'method: "eye_to_hand"' in written
    assert "agentview:" in written and "extrinsic:" in written
    assert "calibration:" in written and "board:" in written
    # a .bak of the original is kept
    assert (config_file.parent / (config_file.name + ".bak")).read_text() == _CONFIG_YAML
    # eye-to-hand never fabricates a wrist extrinsic or an arm offset (those are gone)
    assert "gripper_T_camera" not in written
    assert "arm_offset" not in written


def test_save_with_no_config_path_shows_an_error_not_a_crash(window, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a)))
    window.config_path = None
    _feed(window, "left", _X_LEFT, _Y, n=6, seed=1)
    window._on_save()  # must not raise
    assert shown  # the "no config.yaml" dialog fired
