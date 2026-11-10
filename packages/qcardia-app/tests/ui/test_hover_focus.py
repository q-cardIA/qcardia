"""Hovering a panel gives it the keyboard focus, unless a text field has it. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget
from qcardia_app.view.custom_graphics_view import CustomQGraphicsView
from qcardia_app.view.theme import apply_theme


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    host = QWidget()
    layout = QVBoxLayout(host)
    host.panel = CustomQGraphicsView()
    host.panel.setEnabled(True)
    host.text = QLineEdit()
    layout.addWidget(host.panel)
    layout.addWidget(host.text)
    host.show()
    host.activateWindow()
    assert QTest.qWaitForWindowActive(host)
    return host


def _hover(panel):
    point = QPointF(5, 5)
    QApplication.sendEvent(panel, QEnterEvent(point, point, point))


def test_hover_focuses_the_panel(window):
    _hover(window.panel)
    assert window.panel.hasFocus()


def test_hover_keeps_a_text_field_focused(window):
    window.text.setFocus()
    _hover(window.panel)
    assert window.text.hasFocus()
