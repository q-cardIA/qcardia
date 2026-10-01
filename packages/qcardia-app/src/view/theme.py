from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QListView, QProgressBar


def _rgb(r, g, b) -> QColor:
    return QColor(r, g, b)


def thin_loading_bar() -> QProgressBar:
    """Hairline activity strip for under a control. Indeterminate: Ollama reports no
    load progress, so a percentage would be invented."""
    bar = QProgressBar()
    bar.setProperty("role", "thinLoadingBar")
    bar.setRange(0, 0)
    bar.setTextVisible(False)
    bar.setFixedHeight(3)
    bar.hide()
    return bar


class PopupComboBox(QComboBox):
    """Combo whose popup shows its items without scrolling.

    Qt sizes the popup from the view's size hint, which this app's padded
    QAbstractItemView rule can collapse to one row. Setting a minimum from the row
    count is a floor, not a ceiling, so a list past MAX_ROWS still scrolls.
    """

    MAX_ROWS = 12
    _FALLBACK_ROW_H = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        # A real QListView honours the ::item metrics in theme.qss; the default does not.
        self.setView(QListView())
        self.setMaxVisibleItems(self.MAX_ROWS)

    def showPopup(self):
        rows = min(self.count(), self.MAX_ROWS)
        if rows:
            view = self.view()
            row_h = view.sizeHintForRow(0) or self._FALLBACK_ROW_H
            # Plus the view's own top/bottom inset, so the last row isn't clipped.
            view.setMinimumHeight(rows * row_h + 10)
        super().showPopup()


def _build_palette() -> QPalette:
    p = QPalette()
    r = QPalette.ColorRole
    p.setColor(r.Window,          _rgb(40, 44, 52))
    p.setColor(r.WindowText,      _rgb(221, 221, 221))
    p.setColor(r.Base,            _rgb(27, 29, 35))
    p.setColor(r.AlternateBase,   _rgb(33, 37, 43))
    p.setColor(r.Text,            _rgb(221, 221, 221))
    p.setColor(r.BrightText,      _rgb(255, 255, 255))
    p.setColor(r.Button,          _rgb(52, 59, 72))
    p.setColor(r.ButtonText,      _rgb(221, 221, 221))
    p.setColor(r.ToolTipBase,     _rgb(33, 37, 43))
    p.setColor(r.ToolTipText,     _rgb(221, 221, 221))
    p.setColor(r.Highlight,       _rgb(189, 147, 249))
    p.setColor(r.HighlightedText, _rgb(255, 255, 255))
    p.setColor(r.Link,            _rgb(189, 147, 249))
    p.setColor(r.PlaceholderText, _rgb(113, 126, 149))
    g = QPalette.ColorGroup.Disabled
    p.setColor(g, r.WindowText, _rgb(80, 85, 95))
    p.setColor(g, r.Text,       _rgb(80, 85, 95))
    p.setColor(g, r.ButtonText, _rgb(80, 85, 95))
    p.setColor(g, r.Highlight,  _rgb(44, 49, 58))
    return p


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(_build_palette())

    assets = Path(__file__).parent.parent.parent / "assets" / "fonts"
    family = None
    for ttf in sorted(assets.glob("Inter*.ttf")):
        fid = QFontDatabase.addApplicationFont(str(ttf))
        if fid != -1 and family is None:
            names = QFontDatabase.applicationFontFamilies(fid)
            if names:
                family = names[0]

    font = QFont(family) if family else app.font()
    font.setPointSize(10)
    app.setFont(font)

    qss_path = Path(__file__).with_name("theme.qss")
    app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
