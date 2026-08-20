"""The render GUI: what it reads off a dataset, and the args it hands the renderer."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from misc.render_gui import RenderGUI


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_an_empty_root_says_so_instead_of_failing(qapp, tmp_path):
    gui = RenderGUI(str(tmp_path))
    try:
        assert gui.dataset_combo.count() == 0
        assert "No dataset folders" in gui.info.text()
        assert gui.episode_combo.count() == 0
    finally:
        gui.close()


def test_it_leaves_the_recorded_numbers_for_the_renderer_to_recover(qapp, tmp_path):
    """The point of the window: nothing the recording already knows is typed in. `candidates` and
    `horizon` stay None so render() reads them off the dataset -- a mistyped candidate count is a
    reshape error and a mistyped horizon silently draws every chunk in the wrong place."""
    gui = RenderGUI(str(tmp_path))
    try:
        args = gui._build_args()
        assert args.candidates is None and args.horizon is None
        assert args.replans == 0 and args.hold == 1
        # Speed is the written frame rate, not a re-encode: 1x is the renderer's own default of 10.
        gui.speed_spin.setValue(1.0)
        assert gui._build_args().fps == 10
        gui.speed_spin.setValue(4.0)
        assert gui._build_args().fps == 40
    finally:
        gui.close()


def test_render_is_refused_without_an_episode(qapp, tmp_path):
    """An empty root has no episode to render; pressing the button must say so rather than start a
    worker that fails deep inside the renderer."""
    gui = RenderGUI(str(tmp_path))
    try:
        gui._on_render()
        assert "episode" in gui.status.text().lower()
        assert gui.render_btn.isEnabled()
    finally:
        gui.close()
