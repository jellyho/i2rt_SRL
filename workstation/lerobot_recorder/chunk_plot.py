"""Live plot of the action-chunk length the server answers with, one bar per replan.

The chunk is adaptive: the broker takes the horizon from every reply rather than from a setting,
so a policy is free to answer with a different number of steps each time (a prefix-guided/RTC
server returns only what is still worth executing; a sampler under load may shorten). That number
exists nowhere but the reply, and it decides how often the policy is re-queried -- and therefore
how often a synchronous rollout stalls. Watching it is how you notice the server quietly changing
its plan length.

Deliberately painted by hand rather than pulling in a plotting library: it is one bounded series
of small integers on a strip a few centimetres tall, redrawn at the UI's 10 Hz.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from workstation.lerobot_recorder import theme


class ChunkLengthPlot(QtWidgets.QWidget):
    """Bar-per-reply chart of recent chunk lengths, newest on the right."""

    #: Bars thinner than this are not worth drawing separately; the view scrolls instead.
    MIN_BAR_PX = 3

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._values: List[int] = []
        # No sizeHint() is defined and the vertical policy is Fixed, so the layout hands this
        # widget exactly its *minimum* -- the minimum is the height, and the maximum only guards
        # against a stray stretch. At 72px the bars were a few pixels apart and a change in plan
        # length was not readable off them; the room comes from the log pane below.
        self.setMinimumHeight(160)
        self.setMaximumHeight(240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setToolTip(
            "Length of each action chunk the policy returned, one bar per replan (newest right).\n"
            "The horizon comes from the reply, not from a setting, so it can change per replan."
        )

    def set_values(self, values: Sequence[int]) -> None:
        values = [int(v) for v in values if int(v) > 0]
        if values != self._values:
            self._values = values
            self.update()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QtGui.QColor("#0d1117"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#30363d")))
        painter.drawRect(rect)

        # pad_b carries the caption on its own line: any less and the descenders clip against the
        # widget edge and the text crowds the bars sitting directly above it.
        pad_l, pad_r, pad_t, pad_b = 44, 8, 12, 22
        plot = rect.adjusted(pad_l, pad_t, -pad_r, -pad_b)
        if plot.width() <= 0 or plot.height() <= 0:
            painter.end()
            return

        muted = QtGui.QColor(theme.MUTED)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)

        if not self._values:
            painter.setPen(muted)
            painter.drawText(rect, QtCore.Qt.AlignCenter, "chunk length — waiting for the first reply")
            painter.end()
            return

        # Only the bars that fit, newest first; the rest scroll off the left.
        capacity = max(1, plot.width() // self.MIN_BAR_PX)
        shown = self._values[-capacity:]
        top = max(shown)
        # Round the axis up so a constant series does not paint as a full-height slab with no scale.
        axis_max = max(1, int(top * 1.15) + 1)

        # Horizontal guides at 0 and the axis maximum.
        painter.setPen(QtGui.QPen(QtGui.QColor("#21262d")))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawLine(plot.left(), plot.top(), plot.right(), plot.top())
        painter.setPen(muted)
        painter.drawText(
            QtCore.QRect(0, plot.top() - 6, pad_l - 6, 12),
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            str(axis_max),
        )
        painter.drawText(
            QtCore.QRect(0, plot.bottom() - 6, pad_l - 6, 12), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, "0"
        )

        # Bars. A length that differs from the previous reply is highlighted: a changing horizon is
        # the thing worth noticing, and it is easy to miss in a field of equal bars.
        slot = plot.width() / len(shown)
        bar_w = max(1.0, slot - 1.0)
        steady = QtGui.QColor(theme.ACCENT)
        changed = QtGui.QColor(theme.WARN)
        previous = None
        for i, value in enumerate(shown):
            height = max(1, round(plot.height() * value / axis_max))
            x = plot.left() + i * slot
            bar = QtCore.QRectF(x, plot.bottom() - height, bar_w, height)
            painter.fillRect(bar, changed if (previous is not None and value != previous) else steady)
            previous = value

        # Caption: the current value, and the range actually seen (constant horizons say so).
        low, high = min(self._values), max(self._values)
        span = f"{low}" if low == high else f"{low}-{high}"
        painter.setPen(muted)
        painter.drawText(
            QtCore.QRect(plot.left(), plot.bottom() + 2, plot.width(), pad_b - 3),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"chunk length · now {self._values[-1]} · seen {span} · {len(self._values)} replans",
        )
        painter.end()
