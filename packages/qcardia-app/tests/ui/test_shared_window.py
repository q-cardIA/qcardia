"""A window change applies to all series of one sequence. Runs without a display (QT_QPA_PLATFORM=offscreen)."""
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
    display = CentralDisplayAreaWidget(host)
    for key in ("CINE_SAX", "CINE_2CH", "LGE_SAX"):
        display.register_series(key, np.zeros((3, 8, 8), np.float32), False, (100.0, 200.0))
    return display


def test_window_applies_to_the_sequence(display):
    display.set_window("CINE_SAX", (120.0, 300.0))
    assert display.window("CINE_2CH") == (120.0, 300.0)
    assert display.window("LGE_SAX") == (100.0, 200.0)


def test_panel_window_stays_own(display):
    display.set_panel_series(0, "CINE_SAX")
    display.set_window("CINE_SAX", (50.0, 80.0), panels=[0])
    assert display.window("CINE_2CH") == (100.0, 200.0)
