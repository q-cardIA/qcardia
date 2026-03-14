from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MiraViewerColors:
    bg_primary: str = "#0a0a0f"
    bg_secondary: str = "#12121a"
    bg_tertiary: str = "#1a1a24"
    border: str = "#2a2a3a"
    text_primary: str = "#f0f0f5"
    text_secondary: str = "#8a8a9a"
    text_tertiary: str = "#6f7084"
    accent: str = "#3b82f6"
    accent_hover: str = "#2563eb"


def _require_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PySide6 is required to use this theme. "
            "Install it with `pip install PySide6`."
        ) from exc


def build_palette(colors: MiraViewerColors = MiraViewerColors()) -> Any:
    qtgui = _require_module("PySide6.QtGui")
    palette = qtgui.QPalette()
    qpalette = qtgui.QPalette
    qcolor = qtgui.QColor

    palette.setColor(qpalette.ColorRole.Window, qcolor(colors.bg_primary))
    palette.setColor(
        qpalette.ColorRole.WindowText,
        qcolor(colors.text_primary),
    )
    palette.setColor(qpalette.ColorRole.Base, qcolor(colors.bg_primary))
    palette.setColor(
        qpalette.ColorRole.AlternateBase,
        qcolor(colors.bg_secondary),
    )
    palette.setColor(
        qpalette.ColorRole.ToolTipBase,
        qcolor(colors.bg_secondary),
    )
    palette.setColor(
        qpalette.ColorRole.ToolTipText,
        qcolor(colors.text_primary),
    )
    palette.setColor(qpalette.ColorRole.Text, qcolor(colors.text_primary))
    palette.setColor(qpalette.ColorRole.Button, qcolor(colors.bg_tertiary))
    palette.setColor(
        qpalette.ColorRole.ButtonText,
        qcolor(colors.text_secondary),
    )
    palette.setColor(qpalette.ColorRole.BrightText, qcolor("white"))
    palette.setColor(qpalette.ColorRole.Highlight, qcolor(colors.accent))
    palette.setColor(qpalette.ColorRole.HighlightedText, qcolor("white"))
    palette.setColor(qpalette.ColorRole.Link, qcolor(colors.accent))
    palette.setColor(
        qpalette.ColorRole.PlaceholderText,
        qcolor(colors.text_secondary),
    )

    palette.setColor(
        qpalette.ColorGroup.Disabled,
        qpalette.ColorRole.WindowText,
        qcolor(colors.text_tertiary),
    )
    palette.setColor(
        qpalette.ColorGroup.Disabled,
        qpalette.ColorRole.Text,
        qcolor(colors.text_tertiary),
    )
    palette.setColor(
        qpalette.ColorGroup.Disabled,
        qpalette.ColorRole.ButtonText,
        qcolor(colors.text_tertiary),
    )
    palette.setColor(
        qpalette.ColorGroup.Disabled,
        qpalette.ColorRole.Highlight,
        qcolor(colors.border),
    )

    return palette


def load_stylesheet(path: str | Path | None = None) -> str:
    qss_path = Path(path) if path else Path(__file__).with_name("miraviewer.qss")
    return qss_path.read_text(encoding="utf-8")


def apply_miraviewer_theme(
    app: Any,
    *,
    stylesheet_path: str | Path | None = None,
    colors: MiraViewerColors = MiraViewerColors(),
    force_fusion: bool = True,
) -> None:
    _require_module("PySide6.QtWidgets")

    if force_fusion:
        app.setStyle("Fusion")

    app.setPalette(build_palette(colors))
    app.setStyleSheet(load_stylesheet(stylesheet_path))


def set_variant(widget: Any, variant: str) -> None:
    widget.setProperty("variant", variant)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
