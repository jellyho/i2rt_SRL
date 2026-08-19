"""ChunkLengthPlot — the deploy GUI's live plot of the horizon each reply carried."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtGui, QtWidgets

from workstation.lerobot_recorder.chunk_plot import ChunkLengthPlot


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _paint(widget, width=420, height=90):
    """Render offscreen: painting is the whole behaviour, so a test that never paints tests
    nothing -- an index error or a divide-by-zero only happens in paintEvent."""
    widget.resize(width, height)
    pixmap = QtGui.QPixmap(widget.size())
    widget.render(pixmap)
    return pixmap


def test_it_paints_before_any_reply_has_arrived(qapp):
    plot = ChunkLengthPlot()
    assert not _paint(plot).isNull()


def test_it_paints_a_constant_horizon_and_a_changing_one(qapp):
    plot = ChunkLengthPlot()
    plot.set_values([30] * 40)  # pi05: fixed 30 every replan
    assert not _paint(plot).isNull()
    plot.set_values([30, 30, 12, 7, 25, 9])  # an adaptive server
    assert not _paint(plot).isNull()


def test_more_replans_than_pixels_still_paint(qapp):
    """The history outgrows the strip; the oldest bars scroll off rather than shrinking to zero
    width (a zero-width bar is a divide-by-zero, not a drawing)."""
    plot = ChunkLengthPlot()
    plot.set_values(list(range(1, 500)))
    assert not _paint(plot, width=200).isNull()


def test_a_degenerate_size_does_not_crash(qapp):
    plot = ChunkLengthPlot()
    plot.set_values([4, 4])
    assert not _paint(plot, width=8, height=8).isNull()


def test_non_positive_lengths_are_ignored(qapp):
    """0 means "no chunk yet", not a reply of length zero -- plotting it would draw a false gap."""
    plot = ChunkLengthPlot()
    plot.set_values([0, 5, -1, 7])
    assert plot._values == [5, 7]
