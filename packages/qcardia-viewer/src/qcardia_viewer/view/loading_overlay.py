from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an indeterminate progress bar + status label.

    Parented to (and covering) the widget it should block, e.g. the central
    display area during patient loading.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "loadingOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        self._label = QLabel("Loading…")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setProperty("role", "loadingOverlayLabel")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setFixedWidth(220)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, alignment=Qt.AlignCenter)

        self.hide()

    def show_message(self, text: str):
        self._label.setText(text)
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def hide_overlay(self):
        self.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
