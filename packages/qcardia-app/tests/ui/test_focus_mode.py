"""Focus mode: one panel fills the area, and the grid comes back. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from qcardia_app.view.central_display_area_widget import CentralDisplayAreaWidget
from qcardia_app.view.theme import apply_theme


@pytest.fixture
def display():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    host = QWidget()
    host.resize(800, 400)
    display = CentralDisplayAreaWidget(host)
    display.set_grid_size(1, 2)
    for i, key in enumerate(("CINE_SAX", "LGE_SAX")):
        display.register_series(key, np.zeros((3, 8, 8), np.float32), False, (100.0, 200.0))
        display.set_panel_series(i, key)
    host.show()
    display.show()
    app.processEvents()
    return display


def test_focus_hides_the_other_panels(display):
    display.toggle_focus(1)
    QApplication.processEvents()
    display.layout().activate()
    shown = [p.container.isVisibleTo(display) for p in display._panels]
    assert shown == [False, True]
    assert display._panels[1].container.width() > 0.9 * display.width()
    assert display.panel_assignments()[0]["series_key"] == "CINE_SAX"  # the series stay


def test_second_toggle_restores_the_grid(display):
    display.toggle_focus(0)
    display.toggle_focus(0)
    assert all(p.container.isVisibleTo(display) for p in display._panels)


def test_new_layout_leaves_focus(display):
    display.toggle_focus(0)
    display.set_grid_size(2, 2)
    QApplication.processEvents()
    assert all(p.container.isVisibleTo(display) for p in display._panels)
