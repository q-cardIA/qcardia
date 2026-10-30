from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from qcardia_app.utils import bindings

_KIND_TITLES = {
    "drag": ("Mouse drag", "a mouse button (with Ctrl, Shift or Alt), or a key to hold while dragging with left"),
    "wheel": ("Mouse wheel", "the wheel (with Ctrl, Shift or Alt), or a key to hold while scrolling"),
    "key": ("Single actions", "a key, or a mouse click (also the side buttons)"),
    "hold": ("Held down", "any key to hold, alone or with Ctrl, Shift or Alt, a modifier on its own, "
                           "or the middle or a side mouse button"),
}
_REJECT = {
    "drag": "Drag actions need a mouse button or a key to hold.",
    "wheel": "Scroll actions need the wheel or a key to hold.",
    "key": "Single actions need a key or a mouse click.",
    "hold": "This action needs a key, or the middle or a side mouse button.",
}


class _BindingButton(QPushButton):
    """Shows a binding. Left-click it to edit, then press a key, click or scroll (Esc cancels)."""

    captured = Signal(str, object)  # (action, binding); None = cancelled, "" = input does not fit

    def __init__(self, action: str, kind: str):
        super().__init__()
        self.action, self.kind = action, kind
        self._capturing = self._swallow_release = False
        self._modifier: str | None = None
        self.setMinimumWidth(180)
        self.clicked.connect(self._start)

    def show_binding(self, binding: str):
        self._capturing = False
        self.setText(bindings.label(binding))

    def _start(self):
        self._capturing, self._modifier = True, None
        self.setText("Press a key, click or scroll…")
        self.setFocus()

    def _finish(self, binding: str | None):
        self._capturing = False
        self.captured.emit(self.action, binding)

    def mousePressEvent(self, event):
        if not self._capturing:
            super().mousePressEvent(event)
        elif self.kind == "wheel":
            self._finish("")
        else:
            drag = bindings.drag_binding(event.button(), event.modifiers())
            if self.kind == "hold" and event.button() not in bindings.HOLD_BUTTONS:
                drag = None
            self._finish((drag if self.kind == "drag" else bindings.click_of(drag)) or "")

    def wheelEvent(self, event):
        if self._capturing:
            self._finish(bindings.wheel_binding(event.modifiers()) if self.kind == "wheel" else "")

    def keyPressEvent(self, event):
        if not self._capturing:
            super().keyPressEvent(event)
        elif event.key() == Qt.Key_Escape:
            self._finish(None)
        elif event.key() in bindings.MODIFIER_KEYS:
            self._modifier = bindings.MODIFIER_KEYS[event.key()] if self.kind == "hold" else None
        else:
            key = bindings.bare_key(event)
            if self.kind in ("key", "hold"):
                self._finish(bindings.key_binding(event))
            elif key.isalnum() or len(key) > 1:  # a key to hold: letters, digits or named keys like Space
                self._finish(f"{key}+LeftDrag" if self.kind == "drag" else f"{key}+Wheel")
            else:
                self._finish("")
            self._swallow_release = True

    def focusOutEvent(self, event):
        if self._capturing:
            self._finish(None)
        super().focusOutEvent(event)

    def keyReleaseEvent(self, event):
        if self._capturing and self._modifier and bindings.hold_key(event) == self._modifier:
            self._finish(self._modifier)
            return
        if self._swallow_release:  # else releasing Space would click the button again
            self._swallow_release = False
            return
        super().keyReleaseEvent(event)


class ControlsDialog(QDialog):
    """Remap the viewer's mouse, wheel and keyboard controls; saved when OK is clicked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mouse and keyboard controls")
        self._bindings = bindings.current()
        self._buttons: dict[str, _BindingButton] = {}

        layout = QVBoxLayout(self)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset = QComboBox()
        self._preset.addItems([*bindings.PRESETS, "Custom"])
        self._preset.activated.connect(self._on_preset)
        preset_row.addWidget(self._preset, 1)
        layout.addLayout(preset_row)

        grid = QGridLayout()
        row = 0
        for kind, (head, hint) in _KIND_TITLES.items():
            if not any(action_kind == kind for _, action_kind in bindings.ACTIONS.values()):
                continue  # e.g. "hold" without a plugin that registers such an action
            grid.addWidget(QLabel(f"<b>{head}</b>: {hint}"), row, 0, 1, 2)
            row += 1
            for action, (label, action_kind) in bindings.ACTIONS.items():
                if action_kind != kind:
                    continue
                button = _BindingButton(action, kind)
                button.captured.connect(self._on_captured)
                self._buttons[action] = button
                grid.addWidget(QLabel(label), row, 0)
                grid.addWidget(button, row, 1)
                row += 1
        layout.addLayout(grid)

        self._note = QLabel("Double-click a panel to reset its view.")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = buttons.addButton("Reset to defaults", QDialogButtonBox.ResetRole)
        reset.clicked.connect(lambda: self._set_all(bindings.PRESETS[bindings.DEFAULT_PRESET]))
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._set_all(self._bindings)

    def _set_all(self, new: dict[str, str]):
        self._bindings = dict(new)
        for action, button in self._buttons.items():
            button.show_binding(self._bindings.get(action, ""))
        self._preset.setCurrentText(bindings.preset_name(self._bindings))

    def _on_preset(self):
        name = self._preset.currentText()
        if name in bindings.PRESETS:
            self._set_all(bindings.PRESETS[name])

    def _on_captured(self, action: str, binding: str | None):
        if not binding:  # cancelled, or an input that does not fit this action
            if binding == "":
                self._note.setText(_REJECT[bindings.ACTIONS[action][1]])
            self._buttons[action].show_binding(self._bindings[action])
            return
        for other, b in self._bindings.items():
            if other != action and bindings.conflicts(b, binding):  # one input does one thing
                self._bindings[other] = ""
                self._note.setText(f"{bindings.label(b)} was used for '{bindings.ACTIONS[other][0]}', "
                                   "which now has no binding.")
        self._bindings[action] = binding
        self._set_all(self._bindings)

    def _save(self):
        bindings.save(self._bindings)
        self.accept()
