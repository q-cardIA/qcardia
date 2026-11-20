"""The crosshair of a panel that the selected panel's plane cuts is rotated. Runs without a display."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from qcardia_viewer.view.central_display_area_widget import CentralDisplayAreaWidget
from qcardia_viewer.view.custom_graphics_view import CustomQGraphicsView
from qcardia_viewer.view.theme import apply_theme
from qcardia_viewer.view.volume_geometry import VolumeGeometry


def _geometry(row_dir, col_dir, ipps):
    row_dir, col_dir = np.array(row_dir, float), np.array(col_dir, float)
    return VolumeGeometry(row_dir, col_dir, np.cross(row_dir, col_dir), [np.array(p, float) for p in ipps],
                          (1.0, 1.0), 64, 64)


@pytest.fixture
def display():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    # Class-wide View menu states; a MainWindow in another test may have set them from the user's settings.
    CustomQGraphicsView.crosshair_shown, CustomQGraphicsView.stack_shown = True, False
    display = CentralDisplayAreaWidget(QWidget())
    display.set_grid_size(1, 2)
    # A short-axis stack (slices 10 mm apart along z) and a long-axis plane at y = 32 that cuts it.
    stacks = {"CINE_SAX": _geometry((1, 0, 0), (0, 1, 0), [(0, 0, 0), (0, 0, 10), (0, 0, 20)]),
              "CINE_2CH": _geometry((1, 0, 0), (0, 0, 1), [(0, 32, 0)])}
    for i, (key, geom) in enumerate(stacks.items()):
        display.register_series(key, np.zeros((len(geom.ipp_list), 64, 64), np.float32), False, (100.0, 200.0), geom)
        display.set_panel_series(i, key)
    return display


def _lines(display, panel):
    """Both crosshair lines of a panel, each as its sorted end points."""
    viewer = display._panels[panel].viewer
    out = []
    for item in (viewer.horizontal_line, viewer.vertical_line):
        line = item.line()
        out.append(sorted([(round(line.x1(), 3), round(line.y1(), 3)), (round(line.x2(), 3), round(line.y2(), 3))]))
    return out


def test_selected_panel_has_a_plain_crosshair(display):
    display._world_crosshair = np.array([20.0, 32.0, 10.0])
    display._select_time_axis(1)
    horizontal, vertical = _lines(display, 1)
    assert horizontal[0][1] == horizontal[1][1] and vertical[0][0] == vertical[1][0]


def test_long_axis_selected_gives_the_short_axis_a_line_at_its_plane_and_one_at_the_position(display):
    display._world_crosshair = np.array([20.0, 32.0, 10.0])
    display._select_time_axis(1)
    plane, position = _lines(display, 0)
    assert sum(plane, ()) == pytest.approx((0, 32.5, 64, 32.5))  # y = 32 mm: row 32 across all columns
    assert position[0][0] == pytest.approx(20.5) and position[1][0] == pytest.approx(20.5)  # x = 20 mm


def test_short_axis_selected_gives_the_long_axis_the_slice_depth_and_the_position(display):
    display._world_crosshair = np.array([20.0, 40.0, 10.0])
    display._select_time_axis(0)  # the middle slice, z = 10 mm
    plane, position = _lines(display, 1)
    assert sum(plane, ()) == pytest.approx((0, 10.5, 64, 10.5))
    assert position[0][0] == pytest.approx(20.5) and position[1][0] == pytest.approx(20.5)


def test_crosshair_toggle_key_action(display):
    shown = []
    display.crosshair_toggled.connect(shown.append)
    display._on_viewer_action(0, "crosshair_toggle", 0)
    display._on_viewer_action(0, "crosshair_toggle", 0)
    assert shown == [False, True]


def test_rotated_crosshair_has_a_plane_and_a_position_colour(display):
    from qcardia_viewer.view.theme import CROSSHAIR_COLORS
    display._world_crosshair = np.array([20.0, 32.0, 10.0])
    display._select_time_axis(1)
    colour = lambda item: item.pen().color().name()
    rotated, plain = display._panels[0].viewer, display._panels[1].viewer
    assert colour(rotated.horizontal_line) == CROSSHAIR_COLORS["plane"]
    assert colour(rotated.vertical_line) == CROSSHAIR_COLORS["position"]
    assert colour(plain.horizontal_line) == colour(plain.vertical_line) == CROSSHAIR_COLORS["position"]


def test_short_axis_slices_on_the_long_axis(display):
    shown = []
    display.stack_toggled.connect(shown.append)
    display._on_viewer_action(1, "sax_planes_toggle", 0)
    display._select_time_axis(0)  # the middle slice, z = 10 mm
    items = display._panels[1].viewer._stack_items
    assert shown == [True] and len(items) == 3 and all(item.isVisible() for item, _ in items)
    assert sorted(round(item.line().y1(), 3) for item, _ in items) == [0.5, 10.5, 20.5]
    widths = {round(item.line().y1(), 3): item.pen().width() for item, _ in items}
    assert widths == {0.5: 1, 10.5: 2, 20.5: 1}  # the current slice thicker
    assert display._panels[0].viewer._stack_items == []  # none in the short axis itself
    display._on_viewer_action(1, "sax_planes_toggle", 0)
    assert not any(item.isVisible() for item, _ in display._panels[1].viewer._stack_items)
