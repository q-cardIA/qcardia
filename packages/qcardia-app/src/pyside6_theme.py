from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QWidget

THEME_TOKENS = {
    "bg0": "#1C2A3D",
    "bg1": "#24354D",
    "text": "#EAF0FF",
    "muted": "#B6C2E2",
    "muted2": "#8FA0C8",
    "brand": "#3A4F6B",
    "brand2": "#5A6F8D",
    "red": "#FF2D2D",
    "orange": "#D5A07A",
    "link": "#89A7CF",
    "linkHover": "#B4C7E2",
}

_FONT_CANDIDATES = (
    "Plus Jakarta Sans",
    "Segoe UI",
    "SF Pro Text",
    "Helvetica Neue",
    "Arial",
    "Roboto",
)


def _available_font_map() -> dict[str, str]:
    database = QFontDatabase()
    return {family.casefold(): family for family in database.families()}


def _choose_ui_font_family(
    app: QApplication,
    available: dict[str, str],
) -> str:
    for candidate in _FONT_CANDIDATES:
        resolved = available.get(candidate.casefold())
        if resolved:
            return resolved
    return app.font().family()


def _build_font_stack(primary_family: str, available: dict[str, str]) -> str:
    stack: list[str] = []
    for candidate in (primary_family, *_FONT_CANDIDATES):
        resolved = available.get(candidate.casefold())
        if resolved and resolved not in stack:
            stack.append(resolved)
    if not stack:
        stack.append(primary_family)
    quoted = ", ".join(f'"{family}"' for family in stack)
    return f"{quoted}, sans-serif"


def _apply_runtime_font_stack(
    stylesheet: str,
    primary_family: str,
    available: dict[str, str],
) -> str:
    font_stack = _build_font_stack(primary_family, available)
    return re.sub(
        r"font-family\s*:\s*[^;]+;",
        f"font-family: {font_stack};",
        stylesheet,
        count=1,
    )


def apply_theme(app: QApplication, qss_path: str | None = None) -> None:
    available_fonts = _available_font_map()
    chosen_family = _choose_ui_font_family(app, available_fonts)
    font = QFont(chosen_family)
    font.setPointSize(10)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(THEME_TOKENS["bg0"]))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2A3D58"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2E4462"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2E4462"))
    palette.setColor(
        QPalette.ColorRole.ToolTipBase,
        QColor(THEME_TOKENS["bg1"]),
    )
    palette.setColor(
        QPalette.ColorRole.WindowText,
        QColor(THEME_TOKENS["text"]),
    )
    palette.setColor(
        QPalette.ColorRole.Text,
        QColor(THEME_TOKENS["text"]),
    )
    palette.setColor(
        QPalette.ColorRole.ButtonText,
        QColor(THEME_TOKENS["text"]),
    )
    palette.setColor(
        QPalette.ColorRole.BrightText,
        QColor(THEME_TOKENS["text"]),
    )
    palette.setColor(QPalette.ColorRole.Link, QColor(THEME_TOKENS["link"]))
    palette.setColor(
        QPalette.ColorRole.Highlight,
        QColor(THEME_TOKENS["brand"]),
    )
    palette.setColor(
        QPalette.ColorRole.HighlightedText,
        QColor(THEME_TOKENS["text"]),
    )
    app.setPalette(palette)

    if qss_path:
        stylesheet_path = Path(qss_path)
    else:
        stylesheet_path = Path(__file__).with_name("pyside6_theme.qss")
    if stylesheet_path.exists():
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
        app.setStyleSheet(
            _apply_runtime_font_stack(
                stylesheet,
                chosen_family,
                available_fonts,
            )
        )


def set_widget_variant(widget: QWidget, name: str, value: str) -> None:
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def mark_card(widget: QWidget) -> None:
    set_widget_variant(widget, "role", "card")


def mark_muted(widget: QWidget) -> None:
    set_widget_variant(widget, "muted", "true")


def mark_secondary(widget: QWidget) -> None:
    set_widget_variant(widget, "secondary", "true")


def set_button_variant(widget: QWidget, variant: str) -> None:
    set_widget_variant(widget, "variant", variant)
