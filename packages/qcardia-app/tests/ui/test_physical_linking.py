"""Linked panels of different series scroll by distance in mm, not by slice. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from qcardia_app.view.central_display_area_widget import CentralDisplayAreaWidget
from qcardia_app.view.theme import apply_theme
from qcardia_app.view.volume_geometry import VolumeGeometry


def _stack(spacing, n):
    ipps = [np.array([0.0, 0.0, spacing * k]) for k in range(n)]
    return VolumeGeometry(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), ipps, (1.0, 1.0), 16, 16)


@pytest.fixture
def display():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    display = CentralDisplayAreaWidget(QWidget())
    display.set_grid_size(2, 2)
    # Both stacks are shown twice, so the panels are placed by slice and not by the crosshair.
    for key, spacing, n in (("CINE_SAX", 10.0, 5), ("LGE_SAX", 5.0, 9)):
        display.register_series(key, np.zeros((n, 16, 16), np.float32), False, (100.0, 200.0), _stack(spacing, n))
    for panel, key, z in ((0, "CINE_SAX", 0), (1, "LGE_SAX", 0), (2, "CINE_SAX", 1), (3, "LGE_SAX", 2)):
        display.set_panel_series(panel, key, slice_idx=z)
    return display


def test_other_series_moves_by_the_same_distance(display):
    display._scroll(0, 1)  # CINE_SAX: one slice is 10 mm
    slices = [display._panel_slice_idx[i] for i in range(4)]
    assert slices == [1, 2, 2, 4]  # LGE_SAX has 5 mm slices, so it moves two, keeping its own offset
