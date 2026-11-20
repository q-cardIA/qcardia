"""Rotating a panel turns the image about the panel centre and refits it. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from qcardia_viewer.view.custom_graphics_view import CustomQGraphicsView
from qcardia_viewer.view.theme import apply_theme


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    panel = CustomQGraphicsView()
    panel.setEnabled(True)
    panel.resize(200, 100)
    panel.show()
    panel.update_slice_view(np.zeros((20, 40), np.uint8))  # a wide image
    app.processEvents()
    return panel


def test_quarter_turn_is_clockwise_and_refits(panel):
    scale = panel.transform().m11()
    panel.rotate_by(90)
    turned = panel.transform()
    edge = turned.map(QPointF(1, 0))  # the right edge goes down
    assert (edge.x(), edge.y()) == pytest.approx((0, scale / 2), abs=1e-6)
    assert turned.m12() == pytest.approx(scale / 2)  # the wide image is now tall, so it is half as large


def test_reset_view_removes_the_rotation(panel):
    panel.rotate_by(90)
    panel.reset_view()
    assert panel.transform().m12() == pytest.approx(0)


def test_rotate_drag_turns_freely_and_snaps_with_shift(panel):
    drag = panel._CustomQGraphicsView__rotate_drag
    drag(60, snap=False)  # half a degree per pixel
    assert panel._rotation == pytest.approx(30)
    drag(40, snap=True)  # 20 degrees, snapped to 15
    assert panel._rotation == pytest.approx(15)


def test_default_bindings_do_not_conflict():
    from qcardia_viewer.utils import bindings
    for preset in bindings.PRESETS.values():
        pairs = [(a, b) for a in preset for b in preset if a < b]
        assert not [(a, b) for a, b in pairs if bindings.conflicts(preset[a], preset[b])]
