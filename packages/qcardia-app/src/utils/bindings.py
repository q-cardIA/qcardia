"""Mouse, wheel and key bindings for the image viewers, remappable on the welcome screen.

A binding is a string: "RightDrag", "Ctrl+LeftDrag" or "Space+LeftDrag" (hold Space, drag with left)
for drags; "Shift+Wheel" or "Z+Wheel" for scrolls; a QKeySequence string such as "Up" or "Space",
or a click such as "MiddleClick" or "BackClick", for single actions.
"""
from __future__ import annotations

import json

from PySide6.QtCore import QEvent, QKeyCombination, QObject, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from utils.app_config import SETTINGS

# action -> (label, kind)
ACTIONS: dict[str, tuple[str, str]] = {
    "crosshair": ("Move crosshair", "drag"),
    "pan": ("Pan", "drag"),
    "window": ("Window / level", "drag"),
    "zoom_drag": ("Zoom", "drag"),
    "slice_scroll": ("Scroll slices", "wheel"),
    "frame_scroll": ("Scroll time frames", "wheel"),
    "zoom_wheel": ("Zoom", "wheel"),
    "slice_prev": ("Previous slice", "key"),
    "slice_next": ("Next slice", "key"),
    "first_slice": ("First slice", "key"),
    "last_slice": ("Last slice", "key"),
    "frame_prev": ("Previous time frame", "key"),
    "frame_next": ("Next time frame", "key"),
    "play": ("Play / pause cine", "key"),
    "zoom_in": ("Zoom in", "key"),
    "zoom_out": ("Zoom out", "key"),
    "reset": ("Reset view", "key"),
    "link": ("Link zoom, pan and scrolling of a panel", "key"),
    "contours": ("Show / hide contours in a panel", "key"),
    "push_to_talk": ("Push to talk", "hold"),
}

_KEYS = {
    "slice_prev": "Up", "slice_next": "Down", "first_slice": "Home", "last_slice": "End",
    "frame_prev": "Left", "frame_next": "Right", "play": "Space",
    "zoom_in": "+", "zoom_out": "-", "reset": "R", "link": "L", "contours": "C", "push_to_talk": "Ctrl+Space",
}
# Cardiac: left stays free for crosshair and contours, right-drag windows (Medis, QMass, Segment).
# PACS: left-drag windows, right-drag zooms (OHIF, Horos).
PRESETS: dict[str, dict[str, str]] = {
    "Cardiac": {"crosshair": "LeftDrag", "pan": "MiddleDrag", "window": "RightDrag",
                "zoom_drag": "Ctrl+RightDrag", "slice_scroll": "Wheel",
                "frame_scroll": "Shift+Wheel", "zoom_wheel": "Ctrl+Wheel", **_KEYS},
    "PACS": {"crosshair": "Shift+LeftDrag", "pan": "MiddleDrag", "window": "LeftDrag",
             "zoom_drag": "RightDrag", "slice_scroll": "Wheel",
             "frame_scroll": "Shift+Wheel", "zoom_wheel": "Ctrl+Wheel", **_KEYS},
}
DEFAULT_PRESET = "Cardiac"


def _load() -> dict[str, str]:
    try:
        saved = json.loads(SETTINGS.value("bindings", "{}"))
    except (TypeError, ValueError):
        saved = {}
    return {a: saved.get(a, b) for a, b in PRESETS[DEFAULT_PRESET].items()}


_current = _load()


def current() -> dict[str, str]:
    return dict(_current)


def save(bindings: dict[str, str]):
    _current.clear()
    _current.update(bindings)
    SETTINGS.setValue("bindings", json.dumps(_current))


def preset_name(bindings: dict[str, str]) -> str:
    return next((name for name, p in PRESETS.items() if p == bindings), "Custom")


def action_for(binding: str | None) -> str | None:
    return next((a for a, b in _current.items() if b == binding), None) if binding else None


def conflicts(a: str, b: str) -> bool:
    """Same input, a click and a drag of the same button, or a key that is also held for a drag or scroll."""
    if not a or not b:
        return False
    if a == b or click_of(a) == b or click_of(b) == a:
        return True
    return any(y.startswith(x + "+") and y.endswith(("Drag", "Wheel")) and not x.endswith(("Drag", "Click", "Wheel"))
               for x, y in ((a, b), (b, a)))


def hold_keys() -> set[str]:
    """Keys held for a drag or scroll, e.g. Space in 'Space+LeftDrag'."""
    return {t for b in _current.values() if b.endswith(("Drag", "Wheel"))
            for t in b.split("+")[:-1] if t not in ("Ctrl", "Shift", "Alt")}


held: str | None = None  # the hold key that is down now


class _HeldKeyTracker(QObject):
    def eventFilter(self, obj, event):
        global held
        if event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and not event.isAutoRepeat():
            key = bare_key(event)
            if event.type() == QEvent.KeyPress and key in hold_keys():
                held = key
            elif key == held:
                held = None
        elif event.type() == QEvent.ApplicationDeactivate:
            held = None
        return False


_tracker: _HeldKeyTracker | None = None


def track_held_keys():
    global _tracker
    if _tracker is None:
        _tracker = _HeldKeyTracker()
        QApplication.instance().installEventFilter(_tracker)


def label(binding: str) -> str:
    """'Ctrl+RightDrag' -> 'Ctrl + right-drag'."""
    if not binding:
        return "—"
    if binding == "+" or binding.endswith("++"):  # the '+' key itself, e.g. 'Ctrl++'
        head, last = binding[:-2], "+"
    else:
        head, _, last = binding.rpartition("+")
    for suffix, word in (("Drag", "-drag"), ("Click", "-click")):
        if last.endswith(suffix) and last != suffix:
            last = last.removesuffix(suffix).lower() + word
    last = "wheel" if last == "Wheel" else last
    return " + ".join([*head.split("+"), last]) if head else last


def _mods(modifiers) -> str:
    return "".join(name for flag, name in ((Qt.ControlModifier, "Ctrl+"), (Qt.ShiftModifier, "Shift+"),
                                           (Qt.AltModifier, "Alt+")) if modifiers & flag)


_BUTTONS = {Qt.LeftButton: "Left", Qt.MiddleButton: "Middle", Qt.RightButton: "Right",
            Qt.BackButton: "Back", Qt.ForwardButton: "Forward"}


def click_of(drag: str | None) -> str | None:
    return drag.removesuffix("Drag") + "Click" if drag and drag.endswith("Drag") else None


def drag_binding(button, modifiers, hold: str | None = None) -> str | None:
    if button not in _BUTTONS:
        return None
    return f"{_mods(modifiers)}{hold + '+' if hold else ''}{_BUTTONS[button]}Drag"


def wheel_binding(modifiers, hold: str | None = None) -> str:
    return f"{_mods(modifiers)}{hold + '+' if hold else ''}Wheel"


def bare_key(event) -> str:
    return QKeySequence(Qt.Key(event.key())).toString()


def key_binding(event) -> str:
    mods = event.modifiers() & ~Qt.KeypadModifier  # macOS flags arrow keys as keypad keys
    text = event.text()
    if text and not text.isalnum() and not text.isspace():
        mods &= ~Qt.ShiftModifier  # '+' needs Shift on some layouts
    return QKeySequence(QKeyCombination(mods, Qt.Key(event.key()))).toString()


MODIFIER_KEYS = {Qt.Key_Control: "Ctrl", Qt.Key_Shift: "Shift", Qt.Key_Alt: "Alt", Qt.Key_Meta: "Meta"}
HOLD_BUTTONS = (Qt.MiddleButton, Qt.BackButton, Qt.ForwardButton)


def hold_binding(event) -> str:
    return MODIFIER_KEYS.get(event.key()) or key_binding(event)


def hold_key(event) -> str:
    return MODIFIER_KEYS.get(event.key()) or bare_key(event)


def last_key(binding: str) -> str:
    return binding.rpartition("+")[2] or "+"
