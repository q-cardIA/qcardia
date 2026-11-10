"""The plane of the selected panel is drawn as a line in the panels it cuts. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from qcardia_app.view.central_display_area_widget import CentralDisplayAreaWidget
from qcardia_app.view.theme import apply_theme
from qcardia_app.view.volume_geometry import VolumeGeometry


def _geometry(row_dir, col_dir, ipps):
    row_dir, col_dir = np.array(row_dir, float), np.array(col_dir, float)
    return VolumeGeometry(row_dir, col_dir, np.cross(row_dir, col_dir), [np.array(p, float) for p in ipps],
                          (1.0, 1.0), 64, 64)


@pytest.fixture
def display():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    display = CentralDisplayAreaWidget(QWidget())
    display.set_grid_size(1, 2)
    # A short-axis stack (slices 10 mm apart along z) and a long-axis plane at y = 32 that cuts it.
    stacks = {"CINE_SAX": _geometry((1, 0, 0), (0, 1, 0), [(0, 0, 0), (0, 0, 10), (0, 0, 20)]),
              "CINE_2CH": _geometry((1, 0, 0), (0, 0, 1), [(0, 32, 0)])}
    for i, (key, geom) in enumerate(stacks.items()):
        display.register_series(key, np.zeros((len(geom.ipp_list), 64, 64), np.float32), False, (100.0, 200.0), geom)
        display.set_panel_series(i, key)
    return display


def _line(display, panel):
    line = display._panels[panel].viewer.plane_line.line()
    return [v for point in sorted([(line.x1(), line.y1()), (line.x2(), line.y2())]) for v in point]


def test_long_axis_plane_is_a_horizontal_line_in_the_short_axis(display):
    display._select_time_axis(1)
    assert _line(display, 0) == pytest.approx([0.5, 32.5, 63.5, 32.5])  # y = 32 mm: row 32 across all columns
    assert _line(display, 1) == pytest.approx([0, 0, 0, 0])  # the selected panel has no line


def test_short_axis_slice_is_a_line_at_its_depth_in_the_long_axis(display):
    display._select_time_axis(0)  # the middle slice, z = 10 mm
    assert _line(display, 1) == pytest.approx([0.5, 10.5, 63.5, 10.5])
