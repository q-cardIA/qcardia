"""Logs every button, menu, combo box, slider, dialog, mouse drag, wheel turn and key press, so new widgets need no
logging code. Each record names the nearest widget with an objectName, e.g. "panel_2"."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QApplication,
    QComboBox,
    QDialog,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QSlider,
    QTextEdit,
    QWidget,
)

from qcardia_app.utils import study_log

_WANTED = {QEvent.Polish, QEvent.Show, QEvent.Hide, QEvent.Wheel, QEvent.KeyPress, QEvent.MouseButtonPress,
           QEvent.MouseButtonRelease}
_MODIFIERS = {Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta}
TASK_SHORTCUT = "Ctrl+Shift+T"


def _target(widget) -> str:
    w = widget
    while w is not None:
        name = w.objectName()
        if name and not name.startswith("qt_"):
            return name
        w = w.parentWidget()
    return type(widget).__name__


def _label(widget) -> str:
    for candidate in (getattr(widget, "text", str)(), widget.toolTip(), widget.property("role")):
        if candidate:
            return str(candidate).strip().splitlines()[0]
    return type(widget).__name__


def _on_slider_action(slider, action):
    if action != QAbstractSlider.SliderMove:
        study_log.log("slider", name=_label(slider), target=_target(slider), value=slider.sliderPosition())


def _watch(widget):
    if not isinstance(widget, (QAbstractButton, QComboBox, QSlider, QMenuBar)) or widget.property("study_logged"):
        return
    widget.setProperty("study_logged", True)
    if isinstance(widget, QMenuBar):  # its triggered also fires for actions in submenus
        widget.triggered.connect(lambda action: study_log.log(
            "menu", name=action.text().replace("&", ""), **({"checked": action.isChecked()} if action.isCheckable() else {})))
    elif isinstance(widget, QAbstractButton):
        widget.clicked.connect(lambda checked, w=widget: study_log.log(
            "button", name=_label(w), target=_target(w), **({"checked": checked} if w.isCheckable() else {})))
    elif isinstance(widget, QComboBox):
        widget.activated.connect(lambda _, w=widget: study_log.log("combo", target=_target(w), value=w.currentText()))
    else:
        widget.actionTriggered.connect(lambda action, w=widget: _on_slider_action(w, action))
        widget.sliderReleased.connect(lambda w=widget: study_log.log(
            "slider", name=_label(w), target=_target(w), value=w.value()))


class _UiLogger(QObject):
    def __init__(self, parent, window):
        super().__init__(parent)
        self._window = window
        self._press = None
        self._away = None

    def on_state_changed(self, state):
        if state != Qt.ApplicationActive and self._away is None:
            self._away = study_log.Span("away")
        elif state == Qt.ApplicationActive and self._away is not None:
            self._away.end()
            self._away = None

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.MouseButtonRelease, QEvent.KeyRelease, QEvent.Wheel):
            self._window.schedule_screen_log()  # what the input changed on screen
        if kind not in _WANTED or not isinstance(obj, QWidget):
            return False
        # A mouse, wheel or key event that a widget ignores goes on to its parent: log it at the first widget only.
        if kind == QEvent.Polish:
            _watch(obj)
        elif kind == QEvent.Show and isinstance(obj, QDialog):
            study_log.log("dialog_open", title=obj.windowTitle(),
                          **({"text": obj.text()} if isinstance(obj, QMessageBox) else {}))
        elif kind == QEvent.Hide and isinstance(obj, QDialog):
            clicked = obj.clickedButton() if isinstance(obj, QMessageBox) else None
            study_log.log("dialog_close", title=obj.windowTitle(),
                          **({"button": clicked.text()} if clicked else {"accepted": obj.result() == QDialog.Accepted}))
        elif kind == QEvent.Wheel:
            if not isinstance(obj, QAbstractSlider) and obj is QApplication.widgetAt(event.globalPosition().toPoint()):
                study_log.log("wheel", target=_target(obj), delta=event.angleDelta().y())
        elif kind == QEvent.KeyPress:
            if not event.isAutoRepeat() and event.key() not in _MODIFIERS and obj is QApplication.focusWidget():
                if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                    study_log.log("keystroke", target=_target(obj))
                else:
                    study_log.log("key", target=_target(obj), key=QKeySequence(event.keyCombination()).toString())
        elif kind == QEvent.MouseButtonPress:
            if not isinstance(obj, QAbstractButton) and obj is QApplication.widgetAt(event.globalPosition().toPoint()):
                self._press = (obj, event.timestamp(), event.position())
        elif kind == QEvent.MouseButtonRelease and self._press is not None and self._press[0] is obj:
            _, pressed_at, pos = self._press
            self._press = None
            study_log.log("mouse", target=_target(obj), button=event.button().name,
                          duration_ms=event.timestamp() - pressed_at,
                          distance_px=round((event.position() - pos).manhattanLength()))
        return False


def install_ui_logger(window):
    """Starts logging the UI for the current study session; Ctrl+Shift+T then marks task start and end."""
    app = QApplication.instance()
    if app.property("ui_logger_installed"):
        return  # a second session after Return to Start Screen; one logger is enough
    app.setProperty("ui_logger_installed", True)
    logger = _UiLogger(app, window)
    for widget in app.allWidgets():
        _watch(widget)
    app.installEventFilter(logger)
    app.applicationStateChanged.connect(logger.on_state_changed)
    shortcut = QShortcut(QKeySequence(TASK_SHORTCUT), window)
    shortcut.setContext(Qt.ApplicationShortcut)
    shortcut.activated.connect(study_log.toggle_task)
