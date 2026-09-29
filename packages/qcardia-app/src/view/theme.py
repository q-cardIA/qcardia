"""The app's look: design tokens, the stylesheet built from them, and the appearance dialog.

Every colour, font size, spacing and radius lives in the token tables below. theme.qss
refers to them as @name and apply_theme() fills them in, so a change here reaches the
whole app. The accent colour and font size are user settings; everything else is fixed.
"""
from functools import cache
from pathlib import Path
import re
import tempfile

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QProgressBar,
    QProxyStyle,
    QPushButton,
    QStyle,
    QVBoxLayout,
)

from utils.app_config import SETTINGS

# Neutral greys, no hue: tinted surroundings shift how grey values in an image are seen.
GREYS = {
    "viewport": "#000000",  # behind the image itself, so the image has no visible edge
    "base": "#242428",  # window background, behind the floating cards
    "bg0": "#101012",   # image card around the viewports
    "bg1": "#18181b",   # panels: top bar, sidebars, chat
    "bg2": "#212125",   # controls, cards, inputs
    "bg3": "#2c2c31",   # hover
    "line": "#36363c",  # borders and dividers
    "text": "#e6e6e8",
    "text_muted": "#a2a2aa",   # secondary text, 6.3:1 on bg2
    "text_faint": "#8e8e97",   # hints and metadata, 4.9:1 on bg2
    "text_disabled": "#56565d",
}

STATUS = {"success": "#5ec27e", "warning": "#e8b04a", "error": "#ef6b73"}

# Okabe-Ito colours: told apart with the common forms of colour blindness.
LABEL_COLORS = {1: "#e69f00", 2: "#56b4e9", 3: "#cc79a7"}

ACCENTS = {
    "Blue": "#85aaff",
    "Purple": "#bd93f9",
    "Teal": "#4fc8c0",
    "Green": "#72c98f",
    "Amber": "#f0b45a",
    "Pink": "#f28bc7",
}
DEFAULT_ACCENT = "Blue"

FONT_SIZES = {"Small": 9, "Default": 10, "Large": 11.5, "Extra large": 13}  # body size in pt
DEFAULT_FONT_SIZE = "Default"

SPACING = {"sp1": 4, "sp2": 8, "sp3": 12, "sp4": 16, "sp5": 24}  # px
RADIUS = 6  # px, the corner radius of controls; pills are the exception
PANEL_RADIUS = 10  # px, the corner radius of the floating cards

_QSS_PATH = Path(__file__).with_name("theme.qss")
_FONTS_DIR = Path(__file__).parents[2] / "assets" / "fonts"
_TOKEN = re.compile(r"@([a-z][a-z0-9_]*)")
_ICON_DIR = Path(tempfile.gettempdir()) / "qcardia-theme"

# Stylesheet images must be files, so these are written out in the current token colours.
_SVG_ICONS = {
    "chevron": ('<path d="M3 6l5 5 5-5" fill="none" stroke="{c}" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"/>', "text_muted"),
    "check": ('<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{c}" stroke-width="2.2" '
              'stroke-linecap="round" stroke-linejoin="round"/>', "on_accent"),
    "check_menu": ('<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{c}" stroke-width="2" '
                   'stroke-linecap="round" stroke-linejoin="round"/>', "accent"),
}


class _Notifier(QObject):
    changed = Signal()


# Emitted after the accent or font size changes, for the few widgets that paint in code.
notifier = _Notifier()


def accent_name() -> str:
    name = SETTINGS.value("appearance/accent", DEFAULT_ACCENT)
    return name if name in ACCENTS else DEFAULT_ACCENT


def font_size_name() -> str:
    name = SETTINGS.value("appearance/font_size", DEFAULT_FONT_SIZE)
    return name if name in FONT_SIZES else DEFAULT_FONT_SIZE


def contrast(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio; body text needs 4.5 or more."""
    def luminance(c: QColor) -> float:
        channels = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                    for v in (c.redF(), c.greenF(), c.blueF())]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _on_accent(accent: QColor) -> str:
    """Text colour on an accent fill: whichever of near-black and white reads better."""
    return max(("#111113", "#ffffff"), key=lambda c: contrast(QColor(c), accent))


_cache: dict[str, dict] = {}  # tokens and QColors for the current appearance; _refresh clears it


def tokens() -> dict[str, str]:
    if "tokens" not in _cache:
        _cache["tokens"] = _build_tokens()
    return _cache["tokens"]


def _build_tokens() -> dict[str, str]:
    accent = QColor(ACCENTS[accent_name()])
    soft = QColor(accent)
    soft.setAlpha(46)
    body = FONT_SIZES[font_size_name()]
    values = {
        **GREYS,
        **STATUS,
        "accent": accent.name(),
        "accent_hover": accent.lighter(112).name(),
        "accent_soft": f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, {soft.alpha()})",
        "on_accent": _on_accent(accent),
        "fs_small": f"{body * 0.9:.1f}pt",
        "fs_body": f"{body:.1f}pt",
        "fs_large": f"{body * 1.3:.1f}pt",
        "fs_display": f"{body * 2:.1f}pt",
        "radius": f"{RADIUS}px",
        "radius_panel": f"{PANEL_RADIUS}px",
        "scrim": "rgba(16, 16, 18, 210)",           # loading overlay over images
        "banner": "#b3101012",                      # name banner over a thumbnail (#AARRGGBB, painted in code)
        "hover_on_image": "rgba(255, 255, 255, 28)",
    }
    values.update({k: f"{v}px" for k, v in SPACING.items()})
    return values


def _icon_files(values: dict[str, str]) -> dict[str, str]:
    _ICON_DIR.mkdir(exist_ok=True)
    files = {}
    for name, (shape, token) in _SVG_ICONS.items():
        hex_value = values[token]
        path = _ICON_DIR / f"{name}-{hex_value.lstrip('#')}.svg"
        if not path.exists():
            path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                            f'{shape.format(c=hex_value)}</svg>', encoding="utf-8")
        files[f"icon_{name}"] = path.as_posix()
    return files


def scaled(px: int) -> int:
    """A size in px that grows with the text size, for widgets sized in code."""
    return round(px * FONT_SIZES[font_size_name()] / FONT_SIZES[DEFAULT_FONT_SIZE])


def color(name: str) -> QColor:
    """A token as a QColor, for widgets that paint in code."""
    colors = _cache.setdefault("colors", {})
    if name not in colors:
        colors[name] = QColor(tokens()[name])
    return QColor(colors[name])  # a copy: changing it must not change the cached token


def label_color(class_id: int) -> QColor:
    return QColor(LABEL_COLORS.get(class_id, GREYS["text"]))


def tinted_icon(path: str, token: str = "text_muted") -> QIcon:
    """A single-colour icon from any PNG, recoloured with a token: the asset colours are ignored."""
    source = QPixmap(path)
    if source.isNull():
        return QIcon()
    source = source.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 2x a 20 px icon
    tinted = QPixmap(source.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, source)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color(token))
    painter.end()
    return QIcon(tinted)


def _draw_mic(painter: QPainter):
    painter.drawRoundedRect(QRectF(12, 4, 8, 15), 4, 4)
    painter.drawArc(QRectF(7, 8, 18, 16), 200 * 16, 140 * 16)
    painter.drawLine(QPointF(16, 24), QPointF(16, 28))


def _draw_arrow_up(painter: QPainter):
    painter.drawLine(QPointF(16, 26), QPointF(16, 7))
    painter.drawPolyline([QPointF(8.5, 14.5), QPointF(16, 7), QPointF(23.5, 14.5)])


def _draw_chevron_down(painter: QPainter):
    painter.drawPolyline([QPointF(9, 12.5), QPointF(16, 19.5), QPointF(23, 12.5)])


def _draw_chevron_right(painter: QPainter):
    painter.drawPolyline([QPointF(12.5, 9), QPointF(19.5, 16), QPointF(12.5, 23)])


def _draw_close(painter: QPainter):
    painter.drawLine(QPointF(9, 9), QPointF(23, 23))
    painter.drawLine(QPointF(23, 9), QPointF(9, 23))


_LINE_ICONS = {"mic": _draw_mic, "arrow_up": _draw_arrow_up, "close": _draw_close,
               "chevron_down": _draw_chevron_down, "chevron_right": _draw_chevron_right}


def line_icon(kind: str, token: str) -> QIcon:
    """A drawn line icon in a token colour, greyed out when its button is disabled."""
    return _line_icon(kind, color(token).name(), color("text_disabled").name())


@cache  # keyed on the colours, so an accent change simply draws new ones
def _line_icon(kind: str, normal: str, disabled: str) -> QIcon:
    icon = QIcon()
    for mode, hex_value in ((QIcon.Normal, normal), (QIcon.Disabled, disabled)):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(2, 2)  # drawn on a 32 px grid, stored at 2x so it stays sharp when scaled
        pen = QPen(QColor(hex_value), 2.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        _LINE_ICONS[kind](painter)
        painter.end()
        icon.addPixmap(pixmap, mode)
    return icon


def stylesheet() -> str:
    values = dict(tokens())
    values.update(_icon_files(values))
    return _TOKEN.sub(lambda m: values[m.group(1)], _QSS_PATH.read_text(encoding="utf-8"))


def _build_palette() -> QPalette:
    t = {k: QColor(v) for k, v in tokens().items() if not v.endswith(("pt", "px", ")"))}
    p = QPalette()
    r = QPalette.ColorRole
    for role, name in (
        (r.Window, "bg1"), (r.WindowText, "text"), (r.Base, "bg2"), (r.AlternateBase, "bg1"),
        (r.Text, "text"), (r.BrightText, "text"), (r.Button, "bg2"), (r.ButtonText, "text"),
        (r.ToolTipBase, "bg2"), (r.ToolTipText, "text"), (r.Highlight, "accent"),
        (r.HighlightedText, "on_accent"), (r.Link, "accent"), (r.PlaceholderText, "text_faint"),
    ):
        p.setColor(role, t[name])
    for role in (r.WindowText, r.Text, r.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, t["text_disabled"])
    p.setColor(QPalette.ColorGroup.Disabled, r.Highlight, t["bg3"])
    return p


def _font_family() -> str | None:
    family = None
    for ttf in sorted(_FONTS_DIR.glob("Inter*.ttf")):
        fid = QFontDatabase.addApplicationFont(str(ttf))
        if fid != -1 and family is None:
            names = QFontDatabase.applicationFontFamilies(fid)
            family = names[0] if names else None
    return family


class _Style(QProxyStyle):
    """Fusion, without the always-on underline under menu access keys; Alt+letter still works."""

    def styleHint(self, hint, option=None, widget=None, data=None):
        if hint == QStyle.SH_UnderlineShortcut:
            return 0
        return super().styleHint(hint, option, widget, data)


def apply_theme(app: QApplication) -> None:
    app.setStyle(_Style("Fusion"))
    family = _font_family()
    font = QFont(family) if family else app.font()
    app.setProperty("themeFontFamily", font.family())
    _refresh(app)


def _refresh(app: QApplication) -> None:
    _cache.clear()
    app.setPalette(_build_palette())
    font = QFont(app.property("themeFontFamily"))
    font.setPointSizeF(FONT_SIZES[font_size_name()])
    app.setFont(font)
    app.setStyleSheet(stylesheet())
    notifier.changed.emit()


def set_appearance(accent: str | None = None, font_size: str | None = None) -> None:
    if accent is not None:
        SETTINGS.setValue("appearance/accent", accent)
    if font_size is not None:
        SETTINGS.setValue("appearance/font_size", font_size)
    SETTINGS.sync()
    _refresh(QApplication.instance())


def appearance_summary() -> str:
    return f"Appearance: {accent_name()}, {font_size_name().lower()} text"


class AppearanceDialog(QDialog):
    """Pick the accent colour and text size. Changes show at once; Cancel restores them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Appearance")
        self._original = (accent_name(), font_size_name())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING["sp5"], SPACING["sp4"], SPACING["sp5"], SPACING["sp4"])
        layout.setSpacing(SPACING["sp3"])

        layout.addWidget(self._title("Accent colour"))
        swatches = QHBoxLayout()
        swatches.setSpacing(SPACING["sp2"])
        self._accents = QButtonGroup(self)
        for name, hex_value in ACCENTS.items():
            swatch = QPushButton()
            swatch.setProperty("role", "accentSwatch")
            swatch.setCheckable(True)
            swatch.setToolTip(name)
            swatch.setAccessibleName(f"{name} accent")
            swatch.setIcon(self._swatch_icon(hex_value))
            swatch.setIconSize(QSize(22, 22))
            swatch.setChecked(name == self._original[0])
            swatch.clicked.connect(lambda _, n=name: set_appearance(accent=n))
            self._accents.addButton(swatch)
            swatches.addWidget(swatch)
        swatches.addStretch(1)
        layout.addLayout(swatches)

        layout.addWidget(self._title("Text size"))
        sizes = QHBoxLayout()
        size_buttons = []
        for name in FONT_SIZES:
            button = QPushButton(name)
            button.clicked.connect(lambda _, n=name: set_appearance(font_size=n))
            size_buttons.append(button)
        sizes.addWidget(segment_group(size_buttons))
        for name, button in zip(FONT_SIZES, size_buttons):
            button.setChecked(name == self._original[1])
        sizes.addStretch(1)
        layout.addLayout(sizes)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setProperty("role", "primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(SPACING["sp2"])
        layout.addWidget(buttons)

    @staticmethod
    def _title(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "sectionTitle")
        return label

    @staticmethod
    def _swatch_icon(hex_value: str) -> QIcon:
        pixmap = QPixmap(44, 44)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(hex_value))
        painter.drawEllipse(2, 2, 40, 40)
        painter.end()
        return QIcon(pixmap)

    def reject(self):
        set_appearance(*self._original)
        super().reject()


class StickyMenu(QMenu):
    """Stays open when a checkable entry is clicked, so several can be set in one visit."""

    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.isCheckable() and action.menu() is None:
            action.trigger()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        action = self.activeAction()
        if (event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and action is not None
                and action.isEnabled() and action.isCheckable() and action.menu() is None):
            action.trigger()
            return
        super().keyPressEvent(event)


def segment_group(buttons: list[QPushButton]) -> QFrame:
    """Exclusive checkable buttons in one outlined group."""
    frame = QFrame()
    frame.setProperty("role", "segmentGroup")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)
    group = QButtonGroup(frame)
    for button in buttons:
        button.setCheckable(True)
        group.addButton(button)
        layout.addWidget(button)
    return frame


def set_style_property(widget, name: str, value) -> None:
    """Set a property that stylesheet selectors read; Qt only re-matches them after a repolish."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


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
