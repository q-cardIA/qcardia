# MiraViewer theme for PySide6

This folder extracts the visual style from MiraViewer's React app into reusable PySide6 assets.

## Included files

- `miraviewer.qss`: widget styling (buttons, inputs, sliders, scrollbars, tabs, tables, menus)
- `miraviewer_theme.py`: palette + stylesheet application helper
- `__init__.py`: package exports
- `demo_theme.py`: standalone visual preview window

## Theme preview demo

From the repository root:

```bash
python -m pyside6_theme.demo_theme
```

Alternative:

```bash
python pyside6_theme/demo_theme.py
```

## Quick start

```python
import sys

from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPushButton
from pyside6_theme import apply_miraviewer_theme, set_variant

app = QApplication(sys.argv)
apply_miraviewer_theme(app)

panel = QFrame()
panel.setProperty("role", "panel")

title = QLabel("MiraViewer-style Panel")
caption = QLabel("Secondary text")
caption.setProperty("textRole", "secondary")

button = QPushButton("Import")
set_variant(button, "primary")
```

## Dynamic properties used by the QSS

- `role="panel"` on `QFrame` for card/panel styling
- `textRole="secondary"` on `QLabel` for muted text
- `textRole="tertiary"` on `QLabel` for disabled/aux text
- `textRole="accent"` on `QLabel` for accent highlights
- `variant="primary"` on `QPushButton` for blue CTA
- `variant="ghost"` on `QPushButton` for low-emphasis controls
- `variant="danger"` on `QPushButton` for destructive actions

## Extracted color tokens

- `bg_primary`: `#0a0a0f`
- `bg_secondary`: `#12121a`
- `bg_tertiary`: `#1a1a24`
- `border`: `#2a2a3a`
- `text_primary`: `#f0f0f5`
- `text_secondary`: `#8a8a9a`
- `text_tertiary`: `#6f7084`
- `accent`: `#3b82f6`
- `accent_hover`: `#2563eb`

`text_tertiary` is included as a practical fallback because the React source uses it widely but does not define it in the root CSS variables.
