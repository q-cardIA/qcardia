"""Results panel: LV and RV function from the CINE_SAX segmentation, as numbers and a volume curve,
and scar from the LGE_SAX segmentation."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from qcardia_app.view import theme

_SERIES = (("LV", "volumes", 1), ("RV", "rv_volumes", 3))  # name, analysis key, segmentation label


def _ranges(numbers: list[int]) -> str:
    """[3, 4, 5, 8] -> '3-5, 8'."""
    parts, start = [], None
    for prev, cur in zip([None, *numbers], [*numbers, None]):
        if start is None:
            start = cur
        elif cur != prev + 1:
            parts.append(f"{start}–{prev}" if prev > start else str(start))
            start = cur
    return ", ".join(parts)


def _section(title: str) -> QLabel:
    label = QLabel(title)
    label.setProperty("role", "sectionTitle")
    return label


def _secondary(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("secondary", "true")
    label.setWordWrap(True)
    return label


def _headline(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    label.setStyleSheet(f"font-size: {theme.scaled(24)}px; font-weight: bold;")  # the theme's font-size wins over setFont
    return label


def _percent(value: float, decimals: int = 0) -> str:
    return f"{value:.{decimals}f}<span style='font-size: {theme.scaled(13)}px; font-weight: normal;'> %</span>"


class ResultsWidget(QFrame):
    """LV and RV function from the CINE_SAX segmentation, and scar from the LGE_SAX segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        body = QWidget()
        body.setAutoFillBackground(False)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(theme.SPACING["sp3"], theme.SPACING["sp2"], theme.SPACING["sp3"],
                                  theme.SPACING["sp3"])
        layout.setSpacing(theme.SPACING["sp2"])

        # CINE SAX function
        layout.addWidget(_section("CINE SAX FUNCTION"))
        self._cine_empty = _secondary("Segment the CINE SAX to see the LV and RV function here.")
        layout.addWidget(self._cine_empty)
        self._cine = QWidget()
        cine = QVBoxLayout(self._cine)
        cine.setContentsMargins(0, 0, 0, 0)
        cine.setSpacing(theme.SPACING["sp2"])
        grid = QGridLayout()
        grid.setHorizontalSpacing(theme.SPACING["sp4"])
        grid.setVerticalSpacing(theme.SPACING["sp1"])
        self._values: dict[tuple[str, str], QLabel] = {}
        for col, (name, _, label) in enumerate(_SERIES, start=1):
            # The swatch carries the colour of the segmentation label; the text stays in text colour.
            head = QWidget()
            row = QHBoxLayout(head)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(theme.SPACING["sp1"])
            swatch = QFrame()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(f"background-color: {theme.label_color(label).name()}; border-radius: 2px;")
            row.addWidget(swatch, 0, Qt.AlignVCenter)
            row.addWidget(QLabel(f"<b>{name}</b>"))
            grid.addWidget(head, 0, col, Qt.AlignRight)
        for row, (metric, text) in enumerate((("ef", "EF"), ("sv", "SV"), ("edv", "EDV"), ("esv", "ESV")), start=1):
            grid.addWidget(_secondary(text), row, 0)
            for col, (name, _, _) in enumerate(_SERIES, start=1):
                value = _headline() if metric == "ef" else QLabel()
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._values[name, metric] = value
                grid.addWidget(value, row, col)
        grid.setColumnStretch(0, 1)
        cine.addLayout(grid)
        self._phases = _secondary()
        cine.addWidget(self._phases)
        cine.addWidget(_section("VOLUME OVER TIME"))
        self._chart = VolumeChart()
        cine.addWidget(self._chart)
        layout.addWidget(self._cine)

        # LGE scar
        layout.addSpacing(theme.SPACING["sp2"])
        layout.addWidget(_section("LGE SCAR"))
        self._lge_empty = _secondary("Segment the LGE SAX to see the scar here.")
        layout.addWidget(self._lge_empty)
        self._lge = QWidget()
        lge = QGridLayout(self._lge)
        lge.setContentsMargins(0, 0, 0, 0)
        lge.setHorizontalSpacing(theme.SPACING["sp4"])
        lge.setVerticalSpacing(theme.SPACING["sp1"])
        self._scar_pct, self._scar, self._myo, self._scar_slices = _headline(), QLabel(), QLabel(), QLabel()
        rows = (("Scar of myocardium", self._scar_pct), ("Scar", self._scar), ("Myocardium", self._myo),
                ("Slices with scar", self._scar_slices))
        for row, (text, value) in enumerate(rows):
            lge.addWidget(_secondary(text), row, 0)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lge.addWidget(value, row, 1)
        lge.setColumnStretch(0, 1)
        layout.addWidget(self._lge)

        layout.addSpacing(theme.SPACING["sp2"])
        layout.addWidget(_secondary("From the automatic segmentations; check the contours before you use "
                                    "these numbers."))
        layout.addStretch(1)
        self.show_analysis({})

    def show_analysis(self, analysis: dict):
        """The current patient's analyses by series key (DICOMModel.analysis); {} clears the panel."""
        self._show_function(analysis.get("CINE_SAX"))
        self._show_scar(analysis.get("LGE_SAX"))

    def _show_function(self, analysis: dict | None):
        """Per ventricle EDV and ESV are its largest and smallest volume; ED and ES come from the LV."""
        shown = bool(analysis and "volumes" in analysis)
        self._cine_empty.setVisible(not shown)
        self._cine.setVisible(shown)
        if not shown:
            self._chart.set_curves({}, None, None)
            return
        curves = {name: analysis[key] for name, key, _ in _SERIES if analysis.get(key) and max(analysis[key]) > 0}
        for name, _, _ in _SERIES:
            edv, esv = (max(curves[name]), min(curves[name])) if name in curves else (None, None)
            for metric, value in (("ef", 100 * (edv - esv) / edv if edv else None), ("sv", edv and edv - esv),
                                  ("edv", edv), ("esv", esv)):
                text = "–" if value is None else _percent(value) if metric == "ef" else f"{value:.0f} ml"
                self._values[name, metric].setText(text)
        self._phases.setText(f"ED  frame {analysis['ed'] + 1}      ES  frame {analysis['es'] + 1}")
        self._chart.set_curves(curves, analysis["ed"], analysis["es"])

    def _show_scar(self, analysis: dict | None):
        """Scar as % of the myocardium (which includes it), volumes and masses at 1.05 g/ml."""
        shown = bool(analysis and "scar_ml" in analysis)
        self._lge_empty.setVisible(not shown)
        self._lge.setVisible(shown)
        if not shown:
            return
        myo, scar = analysis["myo_ml"], analysis["scar_ml"]
        self._scar_pct.setText(_percent(100 * scar / myo if myo else 0.0, 1))
        self._scar.setText(f"{scar:.1f} ml  ·  {scar * 1.05:.1f} g")
        self._myo.setText(f"{myo:.0f} ml  ·  {myo * 1.05:.0f} g")
        self._scar_slices.setText(_ranges([z + 1 for z in analysis["scar_slices"]]) or "none")


class VolumeChart(QWidget):
    """LV and RV volume per frame on one ml axis, with the ED and ES frames marked. Hover shows the
    volumes of one frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curves: dict[str, list[float]] = {}
        self._ed = self._es = None
        self._hover: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(theme.scaled(220))

    def set_curves(self, curves: dict[str, list[float]], ed: int | None, es: int | None):
        self._curves, self._ed, self._es, self._hover = curves, ed, es, None
        self.update()

    def _frames(self) -> int:
        return max((len(c) for c in self._curves.values()), default=0)

    def _plot_rect(self) -> QRectF:
        fm = self.fontMetrics()
        left = fm.horizontalAdvance("000 ml") + theme.SPACING["sp2"]
        top = fm.height() * 2 + theme.SPACING["sp2"]  # legend row, then the ED/ES labels
        bottom = fm.height() + theme.SPACING["sp2"]
        return QRectF(left, top, max(1, self.width() - left - theme.SPACING["sp2"]),
                      max(1, self.height() - top - bottom))

    def _scale(self) -> tuple[float, list[int]]:
        """Top of the ml axis and its ticks: 0 and round steps, 3 to 5 lines."""
        peak = max((max(c) for c in self._curves.values()), default=0) or 1
        step = next(s for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000) if peak / s <= 4)
        top = step * (int(peak // step) + 1)
        return top, list(range(0, top + 1, step))

    def _point(self, rect: QRectF, frame: int, volume: float, top: float) -> QPointF:
        n = self._frames()
        x = rect.left() + (rect.width() * frame / (n - 1) if n > 1 else rect.width() / 2)
        return QPointF(x, rect.bottom() - rect.height() * volume / top)

    def paintEvent(self, event):
        n = self._frames()
        if not n:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect, (top, ticks) = self._plot_rect(), self._scale()
        fm = self.fontMetrics()
        muted, faint = theme.color("text_muted"), theme.color("text_faint")

        # Recessive grid and axis labels.
        for tick in ticks:
            y = self._point(rect, 0, tick, top).y()
            p.setPen(QPen(theme.color("line"), 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(faint)
            p.drawText(QRectF(0, y - fm.height() / 2, rect.left() - theme.SPACING["sp1"], fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, f"{tick} ml" if tick == ticks[-1] else str(tick))
        for frame in sorted({0, (n - 1) // 2, n - 1}):
            x = self._point(rect, frame, 0, top).x()
            p.drawText(QRectF(x - 30, rect.bottom() + theme.SPACING["sp1"], 60, fm.height()),
                       Qt.AlignHCenter | Qt.AlignTop, str(frame + 1))

        # ED and ES frames.
        for name, frame in (("ED", self._ed), ("ES", self._es)):
            if frame is None:
                continue
            x = self._point(rect, frame, 0, top).x()
            p.setPen(QPen(faint, 1, Qt.DashLine))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.setPen(muted)
            p.drawText(QRectF(x - 30, rect.top() - fm.height() - 2, 60, fm.height()), Qt.AlignHCenter, name)

        # The curves, with a legend above the plot.
        legend_x = rect.left()
        for name, key, label in _SERIES:
            curve = self._curves.get(name)
            if not curve:
                continue
            colour = theme.label_color(label)
            path = QPainterPath(self._point(rect, 0, curve[0], top))
            for t, v in enumerate(curve[1:], start=1):
                path.lineTo(self._point(rect, t, v, top))
            p.setPen(QPen(colour, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            p.drawLine(QPointF(legend_x, fm.height() / 2 + 1), QPointF(legend_x + 14, fm.height() / 2 + 1))
            p.setPen(muted)
            p.drawText(QPointF(legend_x + 18, fm.ascent()), name)
            legend_x += 18 + fm.horizontalAdvance(name) + theme.SPACING["sp4"]

        if self._hover is not None:
            self._paint_hover(p, rect, top)
        p.end()

    def _paint_hover(self, p: QPainter, rect: QRectF, top: float):
        t = self._hover
        x = self._point(rect, t, 0, top).x()
        p.setPen(QPen(theme.color("text_faint"), 1))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        lines = [f"Frame {t + 1}"]
        for name, _, label in _SERIES:
            curve = self._curves.get(name)
            if curve:
                point = self._point(rect, t, curve[t], top)
                p.setPen(QPen(theme.color("bg1"), 2))
                p.setBrush(theme.label_color(label))
                p.drawEllipse(point, 4.5, 4.5)
                lines.append(f"{name}  {curve[t]:.0f} ml")
        fm = self.fontMetrics()
        w = max(fm.horizontalAdvance(s) for s in lines) + 2 * theme.SPACING["sp2"]
        h = fm.height() * len(lines) + 2 * theme.SPACING["sp1"]
        box = QRectF(x + 10 if x + 10 + w < rect.right() else x - 10 - w, rect.top(), w, h)
        p.setPen(QPen(theme.color("line"), 1))
        p.setBrush(theme.color("bg2"))
        p.drawRoundedRect(box, theme.RADIUS, theme.RADIUS)
        for i, text in enumerate(lines):
            p.setPen(theme.color("text") if i == 0 else theme.color("text_muted"))
            p.drawText(QPointF(box.left() + theme.SPACING["sp2"],
                               box.top() + theme.SPACING["sp1"] + fm.ascent() + i * fm.height()), text)

    def mouseMoveEvent(self, event):
        n, rect = self._frames(), self._plot_rect()
        if n and rect.left() - 8 <= event.position().x() <= rect.right() + 8:
            frac = (event.position().x() - rect.left()) / rect.width()
            hover = min(n - 1, max(0, round(frac * (n - 1))))
        else:
            hover = None
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, event):
        self._hover = None
        self.update()
