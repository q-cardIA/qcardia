from functools import cache

import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsOpacityEffect,
    QGraphicsView,
    QToolButton,
)

from qcardia_app.utils import bindings
from qcardia_app.view import theme

# Actions the viewer handles itself; the rest go to the display area as (action, amount).
_KEY_ACTIONS = {
    "slice_prev": ("slice", -1), "slice_next": ("slice", 1),
    "first_slice": ("first", 0), "last_slice": ("last", 0),
    "frame_prev": ("frame", -1), "frame_next": ("frame", 1), "play": ("play", 0),
    "link": ("link", 0), "own_frame": ("own_frame", 0), "contours": ("contours", 0),
}
_REPEATING = {"slice_prev", "slice_next", "frame_prev", "frame_next", "zoom_in", "zoom_out"}


@cache
def _icon(draw, off: str, on: str) -> QIcon:
    """16 px line icon in the off and on colours; draw paints it in a 32 x 32 box."""
    icon = QIcon()
    for state, color in ((QIcon.Off, QColor(off)), (QIcon.On, QColor(on))):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(color, 3))
        draw(painter)
        painter.end()
        icon.addPixmap(pixmap, QIcon.Normal, state)
    return icon


def _draw_link(painter):  # two chain links at 45 degrees
    painter.translate(16, 16)
    painter.rotate(-45)
    painter.drawRoundedRect(QRectF(-14, -5, 16, 10), 5, 5)
    painter.drawRoundedRect(QRectF(-2, -5, 16, 10), 5, 5)


def _draw_clock(painter):  # a clock face at ten past
    painter.drawEllipse(QPointF(16, 16), 12, 12)
    painter.drawLine(QPointF(16, 16), QPointF(16, 8))
    painter.drawLine(QPointF(16, 16), QPointF(22, 19))


def _draw_contours(painter):  # short-axis cine segmentation: thick LV ring, RV crescent at the upper left
    rv = QPainterPath(QPointF(23.6, 10.0))  # LV at 70 degrees
    rv.cubicTo(QPointF(12, -4), QPointF(-4, 11), QPointF(13.6, 24.3))  # to 220 degrees
    painter.drawPath(rv)
    pen = painter.pen()
    pen.setWidthF(5)
    painter.setPen(pen)
    painter.drawEllipse(QPointF(20.5, 18.5), 9, 9)


class CustomQGraphicsView(QGraphicsView):
    """One panel's image. Scene coordinates are image pixels (x = col + 0.5, y = row + 0.5);
    zoom and pan are the view transform, so a slice is drawn once at native resolution."""

    coordinates_changed = Signal(int, int)
    contours_updated = Signal(list)
    action_requested = Signal(str, int)  # ("slice"|"frame", step), or (first|last|play|reset|link|own_frame|contours, 0)
    window_dragged = Signal(float, float)  # screen pixels moved (dx, dy)
    view_changed = Signal(float, float, float)
    focused = Signal()

    def __init__(self, view_type="axial", size=QSize(150, 150), parent=None):
        super(CustomQGraphicsView, self).__init__()
        self.parent = parent
        self.view_type = view_type
        self.edit_mode = False
        self._contour_definitions = []
        self._contour_items = []
        self._handle_items = []
        self._active_handle = None
        self._handle_drag_offset = QPointF(0, 0)
        # Pick, drag and snap radii are in screen pixels; __px() converts them to image pixels.
        self._handle_radius = 2.5
        self._handle_pick_radius = self._handle_radius * 2.0
        self._segment_pick_radius = 28.0
        self._snap_to_image_enabled = True
        self._snap_to_image_radius = 2.0
        self._snap_image = None
        self._snap_edge_mask = None
        self._snap_edge_strength = None
        self._active_contour_drag = None
        self._contour_drag_influence_radius = 42.0
        self._contour_drag_falloff_sigma = 18.0
        self._edit_tool = "move"

        self._zoom = 1.0  # relative to fit-to-panel
        self._center: QPointF | None = None  # scene point at the panel centre; None = image centre
        self._actual_scale = 1.0  # screen pixels per image pixel
        self._drag: str | None = None
        self._drag_last = None
        self._press_pos = QPointF()
        self._wheel_acc = 0
        self._pixel_spacing_mm = (1.0, 1.0)
        self._contour_line_mm = 2.0
        self.crosshair_pos = None
        self.dragging_crosshair = None
        self._needs_fit = True
        self._fit_view = None
        self._frame = None  # (box in scene coordinates, cover): kept filling the panel until a manual zoom or pan
        self.image_2d_h = 0
        self.image_2d_w = 0

        self.__set_interface()
        self.__set_stylesheets()
        self.__set_panel_toggles()
        self.__apply_theme()
        theme.notifier.changed.connect(self.__apply_theme)
        bindings.track_held_keys()

    def __set_interface(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setEnabled(False)
        self.setFocusPolicy(Qt.ClickFocus)  # keys act on the clicked panel, never steal focus from a text field
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.image_item)

        self.vertical_line = self.scene.addLine(0, 0, 0, 0)
        self.horizontal_line = self.scene.addLine(0, 0, 0, 0)
        for line in (self.vertical_line, self.horizontal_line):
            line.setZValue(100)
            line.setVisible(False)

    def __apply_theme(self):
        pen = QPen(theme.color("accent"), 1)
        pen.setDashPattern([6, 6])  # equal dash and gap
        pen.setCosmetic(True)
        for line in (self.vertical_line, self.horizontal_line):
            line.setPen(pen)
        off, on = theme.color("text_muted").name(), theme.color("accent").name()
        for button, draw in self._toggle_draws:
            button.setIcon(_icon(draw, off, on))

    def __set_stylesheets(self):
        self.setProperty("role", "imageViewport")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__apply_view()
        self.__position_toggles()

    def __set_panel_toggles(self):
        buttons = []
        for action, tip, draw in (("link", "Link zoom, pan and scrolling with the other linked panels of this view (SAX, 4CH, ...)",
                                   _draw_link),
                                  ("own_frame", "Own time frame: the others play and scroll without this panel; select it "
                                   "to scrub it alone (e.g. ED next to ES)", _draw_clock),
                                  ("contours", "Show contours in this panel", _draw_contours)):
            button = QToolButton(self)
            button.setCheckable(True)
            button.setIconSize(QSize(16, 16))
            button.setFixedSize(20, 20)
            key = bindings.current().get(action)
            button.setToolTip(f"{tip} ({bindings.label(key)})" if key else tip)
            button.setProperty("role", "viewerOverlay")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, a=action: self.action_requested.emit(a, 0))
            buttons.append(button)
        self._link_button, self._time_button, self._contour_button = buttons
        self._toggle_draws = list(zip(buttons, (_draw_link, _draw_clock, _draw_contours)))
        for button in (self._link_button, self._time_button):  # visible, faded, while off the default
            button.setGraphicsEffect(QGraphicsOpacityEffect(button))
        self._has_frames = False
        self._contour_button.setChecked(True)
        self.__show_toggles(False)

    def set_panel_toggles(self, linked: bool, contours: bool, own_frame: bool | None = None):
        """own_frame None: the series has no frames, so no clock."""
        self._link_button.setChecked(linked)
        self._contour_button.setChecked(contours)
        self._has_frames = own_frame is not None
        self._time_button.setChecked(bool(own_frame))
        self.__show_toggles(self.underMouse())

    def __show_toggles(self, hover: bool):
        self._contour_button.setVisible(hover)
        self._link_button.setVisible(hover or not self._link_button.isChecked())
        self._time_button.setVisible(self._has_frames and (hover or self._time_button.isChecked()))
        for button in (self._link_button, self._time_button):
            button.graphicsEffect().setOpacity(1.0 if hover else 0.45)
        self.__position_toggles()

    def enterEvent(self, event):
        self.__show_toggles(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.__show_toggles(self.rect().contains(self.mapFromGlobal(QCursor.pos())))  # still inside, on a button
        super().leaveEvent(event)

    def __position_toggles(self):
        margin, spacing = 8, 4
        shown = [b for b in (self._link_button, self._time_button, self._contour_button) if not b.isHidden()]
        for i, button in enumerate(shown):  # right to left, no gap for a hidden one
            button.move(self.width() - margin - (i + 1) * (button.width() + spacing) + spacing,
                        self.height() - margin - button.height())
            button.raise_()

    # ── View transform (zoom and pan) ─────────────────────────────────────

    def __view_center(self) -> QPointF:
        return self._center if self._center is not None else QPointF(self.image_2d_w / 2, self.image_2d_h / 2)

    def __apply_view(self):
        if not self.image_2d_w or not self.image_2d_h:
            return
        view = self.viewport()
        fit = min(view.width() / self.image_2d_w, view.height() / self.image_2d_h)
        if self._frame is not None:
            (x0, y0, x1, y1), cover = self._frame
            scale = (max if cover else min)(view.width() / (x1 - x0), view.height() / (y1 - y0))
            self._zoom, self._center = scale / max(fit, 1e-6), QPointF((x0 + x1) / 2, (y0 + y1) / 2)
        self._actual_scale = max(fit, 1e-6) * self._zoom
        self.setTransform(QTransform.fromScale(self._actual_scale, self._actual_scale))
        self.centerOn(self.__view_center())
        self.__refresh_pen_widths()

    def __px(self) -> float:
        """Image pixels per screen pixel."""
        return 1.0 / max(self._actual_scale, 1e-6)

    def _zoom_by(self, factor: float, view_pos=None):
        """Zoom by factor, keeping the image point under view_pos (default: the centre) in place."""
        new_zoom = max(0.5, min(10.0, self._zoom * factor))
        if not self.image_2d_w or new_zoom == self._zoom:
            return
        self._frame = None
        center = self.__view_center()
        if view_pos is not None:
            anchor = self.mapToScene(view_pos.toPoint())
            center = anchor + (center - anchor) * (self._zoom / new_zoom)
        self._zoom, self._center = new_zoom, center
        self.__apply_view()
        self.__emit_view()

    def __emit_view(self):
        if self.view() is not None:
            self.view_changed.emit(*self.view())

    def view(self) -> tuple[float, float, float] | None:
        """(zoom relative to fit, centre x and y as offsets from the image centre in image widths), or None."""
        if not (self.image_2d_w and self.image_2d_h):
            return None
        center = self.__view_center()
        return self._zoom, center.x() / self.image_2d_w - 0.5, center.y() / self.image_2d_h - 0.5

    def set_view(self, view: tuple[float, float, float] | None):
        self._fit_view, self._frame = view, None
        if self.image_2d_w and not self._needs_fit:
            self.__show_fit_view()

    def __show_fit_view(self):
        if self._fit_view is None:
            self._zoom, self._center = 1.0, None
        else:
            zoom, fx, fy = self._fit_view
            self._zoom, self._center = zoom, QPointF((0.5 + fx) * self.image_2d_w, (0.5 + fy) * self.image_2d_h)
        self.__apply_view()

    def zoom_in(self):
        self._zoom_by(1.25)

    def zoom_out(self):
        self._zoom_by(0.8)

    def reset_view(self):
        self._fit_view = self._frame = None
        self.__show_fit_view()
        self.__emit_view()

    def frame(self, box: tuple[float, float, float, float] | None, cover: bool = False):
        """Keep box (row0, col0, row1, col1 in image pixels, inclusive; None = the whole image) inside
        the panel, or with cover the panel filled by it, also after a resize, until a manual zoom or pan."""
        r0, c0, r1, c1 = box if box is not None else (0, 0, self.image_2d_h - 1, self.image_2d_w - 1)
        self._frame = (c0, r0, c1 + 1, r1 + 1), cover
        self.__apply_view()

    def set_zoom(self, zoom: float):
        """Zoom relative to fit-to-panel, around the current centre."""
        self._frame, self._zoom = None, zoom
        self.__apply_view()

    def new_series(self, view: tuple[float, float, float] | None = None):
        self._needs_fit, self._fit_view, self._frame = True, view, None

    def __pan(self, delta):
        self._frame = None
        self._center = self.__view_center() - QPointF(delta.x(), delta.y()) * self.__px()
        self.centerOn(self._center)
        self.__emit_view()

    # ── Input ─────────────────────────────────────────────────────────────

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit()

    def mousePressEvent(self, event):
        self._press_pos = event.position()
        pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton and self.edit_mode and self.__start_edit(pos):
            return
        binding = bindings.drag_binding(event.button(), event.modifiers(), bindings.held)
        click = bindings.action_for(bindings.click_of(binding))
        if click:
            self.__trigger(click)
            return
        self._drag = bindings.action_for(binding)
        self._drag_last = event.position()
        if self._drag != "crosshair":
            return
        self.dragging_crosshair = None
        if self.crosshair_pos is not None:  # a press within 5 screen pixels of a line drags only that line
            near = 5 * self.__px()
            if abs(pos.x() - self.crosshair_pos[0]) <= near:
                self.dragging_crosshair = "vertical"
            elif abs(pos.y() - self.crosshair_pos[1]) <= near:
                self.dragging_crosshair = "horizontal"
        self.__move_crosshair_to(pos)

    def __start_edit(self, pos) -> bool:
        """Contour editing takes a left press on a handle or segment; returns whether it did."""
        if self._edit_tool == "add":
            if self.__add_point_at(self.__clamp_scene_to_image(pos)):
                return True
        elif self._edit_tool == "remove":
            if self.__remove_point_at(pos):
                return True
        handle_info = self.__pick_handle_at(pos)
        if handle_info:
            self._active_handle = handle_info
            self._handle_drag_offset = handle_info["item"].pos() - pos
            return True
        if self._edit_tool == "move":
            contour_drag = self.__pick_contour_segment_at(pos)
            if contour_drag is not None:
                self._active_contour_drag = contour_drag
                return True
        return False

    def __move_crosshair_to(self, pos):
        if self.dragging_crosshair == "vertical":
            pos = QPointF(pos.x(), self.crosshair_pos[1])
        elif self.dragging_crosshair == "horizontal":
            pos = QPointF(self.crosshair_pos[0], pos.y())
        self.updateCrosshairPosition(pos.x(), pos.y())
        volx, voly = self.__from_graphics_position_to_raw_volume_position(pos)
        self.coordinates_changed.emit(volx, voly)

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.dragging_crosshair = None
        self._active_handle = None
        self._active_contour_drag = None

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())

        if self.edit_mode and self._active_handle is not None:
            target = pos + self._handle_drag_offset
            clamped = self.__clamp_scene_to_image(target)
            if self._snap_to_image_enabled and (event.modifiers() & Qt.AltModifier):
                clamped = self.__snap_scene_point_to_image(clamped)
            self.__move_active_handle(clamped)
            return

        if self.edit_mode and self._active_contour_drag is not None:
            target = self.__clamp_scene_to_image(pos)
            if self._snap_to_image_enabled and (event.modifiers() & Qt.AltModifier):
                target = self.__snap_scene_point_to_image(target)
            self.__drag_active_contour(target)
            return

        if self._drag is None:
            return
        delta = event.position() - self._drag_last
        self._drag_last = event.position()
        if self._drag == "crosshair":
            self.__move_crosshair_to(pos)
        elif self._drag == "pan":
            self.__pan(delta)
        elif self._drag == "zoom_drag":
            self._zoom_by(1.01 ** delta.y(), self._press_pos)
        elif self._drag == "window":
            self.window_dragged.emit(delta.x(), delta.y())

    def mouseDoubleClickEvent(self, event):
        in_place = (event.position() - self._press_pos).manhattanLength() <= 4
        if event.button() == Qt.LeftButton and not self.edit_mode and in_place:
            self.__trigger("reset")
        else:
            self.mousePressEvent(event)

    def wheelEvent(self, event):
        action = bindings.action_for(bindings.wheel_binding(event.modifiers(), bindings.held))
        d = event.angleDelta().y() or event.angleDelta().x()  # Shift+wheel is horizontal on some systems
        if action == "zoom_wheel":
            self._zoom_by(1.0015 ** d, event.position())
        elif action in ("slice_scroll", "frame_scroll"):
            self._wheel_acc += d  # a trackpad sends many small steps; 120 is one notch
            steps = int(self._wheel_acc / 120)
            if steps:
                self._wheel_acc -= steps * 120
                self.action_requested.emit("slice" if action == "slice_scroll" else "frame", -steps)
        event.accept()

    def viewportEvent(self, event):
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            self._zoom_by(1.0 + event.value(), event.position())
            return True
        return super().viewportEvent(event)

    def keyPressEvent(self, event):
        action = bindings.action_for(bindings.key_binding(event))
        if event.isAutoRepeat() and action not in _REPEATING:
            return
        if not self.__trigger(action):
            super().keyPressEvent(event)

    def __trigger(self, action: str | None) -> bool:
        """Run a single (key or click) action; returns whether there was one."""
        if action in _KEY_ACTIONS:
            self.action_requested.emit(*_KEY_ACTIONS[action])
        elif action == "zoom_in":
            self.zoom_in()
        elif action == "zoom_out":
            self.zoom_out()
        elif action == "reset":
            self.reset_view()
            self.action_requested.emit("reset", 0)
        else:
            return False
        return True

    # ── Crosshair ─────────────────────────────────────────────────────────

    crosshair_shown = True  # View > Crosshair, shared by every panel

    def updateCrosshairPosition(self, x, y):
        """Crosshair at scene (image) point x, y, spanning the image."""
        w, h = self.image_2d_w, self.image_2d_h
        x, y = min(max(0.0, x), w), min(max(0.0, y), h)
        self.crosshair_pos = (x, y)
        self.vertical_line.setLine(x, 0, x, h)
        self.horizontal_line.setLine(0, y, w, y)
        self.show_crosshair(CustomQGraphicsView.crosshair_shown)

    def show_crosshair(self, shown: bool):
        """The position keeps following clicks while hidden, so it reappears in the right place."""
        for line in (self.vertical_line, self.horizontal_line):
            line.setVisible(shown and self.crosshair_pos is not None)

    def clear_display(self):
        self.image_2d_h = 0
        self.image_2d_w = 0
        self.crosshair_pos = None
        self._zoom, self._center, self._frame = 1.0, None, None
        self.image_item.setPixmap(QPixmap())
        self.vertical_line.setVisible(False)
        self.horizontal_line.setVisible(False)
        self.clear_contours()
        self.setEnabled(False)

    def __from_graphics_position_to_raw_volume_position(self, graphics_point):
        """(col, row) of the image pixel under a scene point, clamped to the image."""
        if not self.image_2d_w or not self.image_2d_h:
            return 0, 0
        row, col = self.__scene_point_to_image(graphics_point)
        return int(round(col)), int(round(row))

    def set_crosshair_from_volume_indices(self, row_index, col_index):
        if not self.image_2d_w or not self.image_2d_h:
            return
        row = max(0, min(int(row_index), self.image_2d_h - 1))
        col = max(0, min(int(col_index), self.image_2d_w - 1))
        point = self.__image_point_to_scene(col, row)
        self.updateCrosshairPosition(point.x(), point.y())

    # ── Rendering ─────────────────────────────────────────────────────────

    def update_slice_view(self, slice_2d):
        """Show a uint8 slice. The view transform is kept, unless the image size changed."""
        self.setEnabled(True)
        image_2d = np.ascontiguousarray(slice_2d)
        self._snap_image, self._snap_edge_mask = image_2d, None  # edge map is built on first snap
        h, w = image_2d.shape
        new_size = (h, w) != (self.image_2d_h, self.image_2d_w)
        self.image_2d_h, self.image_2d_w = h, w

        self.image_item.setPixmap(QPixmap.fromImage(QImage(image_2d.data, w, h, w, QImage.Format_Grayscale8)))

        if new_size:
            self.scene.setSceneRect(QRectF(-10 * w, -10 * h, 21 * w, 21 * h))  # room to pan
        if new_size or self._needs_fit:
            self._needs_fit = False
            self.__show_fit_view()
            self.updateCrosshairPosition(w / 2, h / 2)
        elif self.crosshair_pos is None:
            self.updateCrosshairPosition(w / 2, h / 2)

    def __refresh_pen_widths(self):
        width = self.__contour_pen_width()
        for entry in self._contour_items:
            pen = entry["path"].pen()
            pen.setWidthF(width)
            entry["path"].setPen(pen)

    def set_pixel_spacing(self, spacing):
        dr, dc = spacing
        if dr > 0 and dc > 0:
            self._pixel_spacing_mm = (float(dr), float(dc))

    def set_contour_line_mm(self, mm: float):
        self._contour_line_mm = float(mm)

    def set_edit_mode(self, enabled):
        self.edit_mode = bool(enabled)
        self._active_handle = None
        self._active_contour_drag = None
        if not self.edit_mode:
            self._edit_tool = "move"
        self.__update_handle_visibility()

    def set_edit_tool(self, tool):
        if tool not in {"move", "add", "remove"}:
            tool = "move"
        self._edit_tool = tool
        self._active_handle = None
        self._active_contour_drag = None

    def update_contour_paths(self, contour_defs):
        if contour_defs is None:
            self._contour_definitions = []
            self.__clear_contour_items()
            return
        sanitized = []
        for contour in contour_defs:
            points = contour.get("points", [])
            if not points:
                continue
            sanitized.append(
                {
                    "class_id": contour.get("class_id", 0),
                    "points": [(float(r), float(c)) for (r, c) in points],
                }
            )
        self._contour_definitions = sanitized
        self.__rebuild_contour_items()

    def clear_contours(self):
        self._contour_definitions = []
        self.__clear_contour_items()

    def __pick_handle_at(self, scene_pos):
        for handle in self._handle_items:
            item = handle["item"]
            if not item.isVisible():
                continue
            center = item.pos()
            dx = scene_pos.x() - center.x()
            dy = scene_pos.y() - center.y()
            if (dx * dx + dy * dy) <= (
                (self._handle_pick_radius * self.__px()) ** 2
            ):
                return handle
        return None

    def __pick_contour_segment_at(self, scene_pos):
        if not self._contour_definitions:
            return None

        best_match = None
        for contour_idx, contour in enumerate(self._contour_definitions):
            points = contour.get("points", [])
            num_points = len(points)
            if num_points < 2:
                continue

            for point_idx in range(num_points):
                start_row, start_col = points[point_idx]
                end_row, end_col = points[(point_idx + 1) % num_points]
                seg_start = self.__image_point_to_scene(start_col, start_row)
                seg_end = self.__image_point_to_scene(end_col, end_row)
                dist_sq = self.__distance_sq_to_segment(scene_pos, seg_start, seg_end)

                if best_match is None or dist_sq < best_match["dist_sq"]:
                    best_match = {
                        "dist_sq": dist_sq,
                        "contour_idx": contour_idx,
                    }

        if best_match is None:
            return None

        max_dist_sq = (self._segment_pick_radius * self.__px()) ** 2
        if best_match["dist_sq"] > max_dist_sq:
            return None

        return {
            "contour_idx": best_match["contour_idx"],
            "last_scene_pos": QPointF(scene_pos.x(), scene_pos.y()),
        }

    def __drag_active_contour(self, scene_pos):
        if self._active_contour_drag is None:
            return

        contour_idx = self._active_contour_drag.get("contour_idx")
        if (
            contour_idx is None
            or contour_idx < 0
            or contour_idx >= len(self._contour_definitions)
        ):
            self._active_contour_drag = None
            return

        previous_scene = self._active_contour_drag.get("last_scene_pos")
        if previous_scene is None:
            self._active_contour_drag["last_scene_pos"] = QPointF(
                scene_pos.x(), scene_pos.y()
            )
            return

        prev_row, prev_col = self.__scene_point_to_image(previous_scene)
        curr_row, curr_col = self.__scene_point_to_image(scene_pos)
        delta_row = curr_row - prev_row
        delta_col = curr_col - prev_col
        if abs(delta_row) < 1e-6 and abs(delta_col) < 1e-6:
            return

        contour = self._contour_definitions[contour_idx]
        points = contour.get("points", [])
        if not points:
            self._active_contour_drag["last_scene_pos"] = QPointF(
                scene_pos.x(), scene_pos.y()
            )
            return

        influence_radius = max(1.0, self._contour_drag_influence_radius) * self.__px()
        sigma = max(1.0, self._contour_drag_falloff_sigma) * self.__px()
        influence_sq = influence_radius * influence_radius

        max_row = max(0.0, float(getattr(self, "image_2d_h", 1) - 1))
        max_col = max(0.0, float(getattr(self, "image_2d_w", 1) - 1))

        updated_points = []
        for row, col in points:
            point_scene = self.__image_point_to_scene(col, row)
            dx = point_scene.x() - scene_pos.x()
            dy = point_scene.y() - scene_pos.y()
            dist_sq = dx * dx + dy * dy

            if dist_sq > influence_sq:
                weight = 0.0
            else:
                weight = float(np.exp(-dist_sq / (2.0 * sigma * sigma)))

            new_row = min(max(0.0, row + delta_row * weight), max_row)
            new_col = min(max(0.0, col + delta_col * weight), max_col)
            updated_points.append((new_row, new_col))

        contour["points"] = updated_points
        self.__update_contour_path(contour_idx)

        if contour_idx < len(self._contour_items):
            for handle in self._contour_items[contour_idx]["handles"]:
                point_idx = handle["point_idx"]
                row, col = contour["points"][point_idx]
                scene_point = self.__image_point_to_scene(col, row)
                handle["item"].setPos(scene_point)

        self.__emit_contours_updated()
        self._active_contour_drag["last_scene_pos"] = QPointF(
            scene_pos.x(), scene_pos.y()
        )

    def __move_active_handle(self, scene_point):
        if self._active_handle is None:
            return
        handle = self._active_handle
        handle["item"].setPos(scene_point)
        row, col = self.__scene_point_to_image(scene_point)
        contour = self._contour_definitions[handle["contour_idx"]]
        contour["points"][handle["point_idx"]] = (row, col)
        self.__update_contour_path(handle["contour_idx"])
        self.__emit_contours_updated()

    @staticmethod
    def __distance_sq_to_segment(point, seg_start, seg_end):
        seg_dx = seg_end.x() - seg_start.x()
        seg_dy = seg_end.y() - seg_start.y()
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-9:
            dx = point.x() - seg_start.x()
            dy = point.y() - seg_start.y()
            return dx * dx + dy * dy

        numerator = (point.x() - seg_start.x()) * seg_dx + (
            point.y() - seg_start.y()
        ) * seg_dy
        t = numerator / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = seg_start.x() + t * seg_dx
        proj_y = seg_start.y() + t * seg_dy
        dx = point.x() - proj_x
        dy = point.y() - proj_y
        return dx * dx + dy * dy

    @staticmethod
    def __project_point_to_segment(point, seg_start, seg_end):
        seg_dx = seg_end.x() - seg_start.x()
        seg_dy = seg_end.y() - seg_start.y()
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-9:
            return QPointF(seg_start.x(), seg_start.y())

        numerator = (point.x() - seg_start.x()) * seg_dx + (
            point.y() - seg_start.y()
        ) * seg_dy
        t = numerator / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = seg_start.x() + t * seg_dx
        proj_y = seg_start.y() + t * seg_dy
        return QPointF(proj_x, proj_y)

    def __update_image_snap_cache(self, image_2d):
        if image_2d is None or image_2d.size == 0:
            self._snap_edge_mask = None
            self._snap_edge_strength = None
            return

        image_float = np.asarray(image_2d, dtype=np.float32)
        grad_row, grad_col = np.gradient(image_float)
        edge_strength = np.hypot(grad_row, grad_col)

        valid = edge_strength[np.isfinite(edge_strength)]
        if valid.size == 0:
            self._snap_edge_mask = None
            self._snap_edge_strength = None
            return

        max_strength = float(valid.max())
        if max_strength <= 1e-6:
            self._snap_edge_mask = None
            self._snap_edge_strength = edge_strength
            return

        threshold = max(float(np.percentile(valid, 95.0)), max_strength * 0.5)
        mask = edge_strength >= threshold

        self._snap_edge_strength = edge_strength
        self._snap_edge_mask = mask if mask.any() else None

    def __snap_scene_point_to_image(self, scene_pos):
        if self._snap_image is not None:
            self.__update_image_snap_cache(self._snap_image)
            self._snap_image = None
        if self._snap_edge_mask is None or self._snap_edge_strength is None:
            return scene_pos

        center_row, center_col = self.__scene_point_to_image(scene_pos)
        height, width = self._snap_edge_mask.shape
        row_idx = min(max(0, int(round(center_row))), height - 1)
        col_idx = min(max(0, int(round(center_col))), width - 1)

        image_radius = max(1, round(self._snap_to_image_radius * self.__px()))
        row_min = max(0, row_idx - image_radius)
        row_max = min(height - 1, row_idx + image_radius)
        col_min = max(0, col_idx - image_radius)
        col_max = min(width - 1, col_idx + image_radius)

        local_mask = self._snap_edge_mask[
            row_min : row_max + 1,
            col_min : col_max + 1,
        ]
        if not local_mask.any():
            return scene_pos

        local_candidates = np.argwhere(local_mask)
        cand_rows = local_candidates[:, 0] + row_min
        cand_cols = local_candidates[:, 1] + col_min

        row_delta = cand_rows - row_idx
        col_delta = cand_cols - col_idx
        dist_sq = row_delta * row_delta + col_delta * col_delta

        best_local_idx = int(np.argmin(dist_sq))
        best_distance = dist_sq[best_local_idx]
        tied = np.where(dist_sq == best_distance)[0]

        if tied.size > 1:
            tied_rows = cand_rows[tied]
            tied_cols = cand_cols[tied]
            tie_strength = self._snap_edge_strength[tied_rows, tied_cols]
            best_tie_idx = tied[int(np.argmax(tie_strength))]
            best_row = int(cand_rows[best_tie_idx])
            best_col = int(cand_cols[best_tie_idx])
        else:
            best_row = int(cand_rows[best_local_idx])
            best_col = int(cand_cols[best_local_idx])

        return self.__image_point_to_scene(best_col, best_row)

    def __add_point_at(self, scene_pos):
        if not self._contour_definitions:
            return False

        best_match = None
        for contour_idx, contour in enumerate(self._contour_definitions):
            points = contour.get("points", [])
            num_points = len(points)
            if num_points < 2:
                continue

            for point_idx in range(num_points):
                start_row, start_col = points[point_idx]
                end_row, end_col = points[(point_idx + 1) % num_points]
                seg_start = self.__image_point_to_scene(start_col, start_row)
                seg_end = self.__image_point_to_scene(end_col, end_row)
                dist_sq = self.__distance_sq_to_segment(scene_pos, seg_start, seg_end)
                if best_match is None or dist_sq < best_match[0]:
                    projected = self.__project_point_to_segment(
                        scene_pos, seg_start, seg_end
                    )
                    best_match = (
                        dist_sq,
                        contour_idx,
                        point_idx + 1,
                        projected,
                    )

        if best_match is None:
            return False

        max_dist_sq = (self._segment_pick_radius * self.__px()) ** 2
        if best_match[0] > max_dist_sq:
            return False

        snapped_pos = best_match[3] if best_match[3] is not None else scene_pos
        if self._snap_to_image_enabled:
            snapped_pos = self.__snap_scene_point_to_image(snapped_pos)
        row, col = self.__scene_point_to_image(snapped_pos)
        _, contour_idx, insert_idx, _ = best_match
        contour = self._contour_definitions[contour_idx]
        contour["points"].insert(insert_idx, (row, col))

        self.__rebuild_contour_items()
        self.__emit_contours_updated()
        return True

    def __remove_point_at(self, scene_pos):
        handle_info = self.__pick_handle_at(scene_pos)
        if not handle_info:
            return False

        contour_idx = handle_info["contour_idx"]
        point_idx = handle_info["point_idx"]
        if contour_idx < 0 or contour_idx >= len(self._contour_definitions):
            return False

        contour = self._contour_definitions[contour_idx]
        points = contour.get("points", [])
        if len(points) <= 3:
            return False

        points.pop(point_idx)
        self.__rebuild_contour_items()
        self.__emit_contours_updated()
        return True

    def __clamp_scene_to_image(self, scene_point):
        row, col = self.__scene_point_to_image(scene_point)
        return self.__image_point_to_scene(col, row)

    def __scene_point_to_image(self, scene_point):
        row = min(max(0.0, scene_point.y() - 0.5), max(0, self.image_2d_h - 1))
        col = min(max(0.0, scene_point.x() - 0.5), max(0, self.image_2d_w - 1))
        return row, col

    @staticmethod
    def __image_point_to_scene(col, row):
        return QPointF(col + 0.5, row + 0.5)

    def __clear_contour_items(self):
        for entry in self._contour_items:
            path_item = entry.get("path")
            if path_item is not None:
                self.scene.removeItem(path_item)
        for handle in self._handle_items:
            self.scene.removeItem(handle["item"])
        self._contour_items = []
        self._handle_items = []
        self._active_handle = None

    def __rebuild_contour_items(self):
        self.__clear_contour_items()
        if not self._contour_definitions:
            return
        for idx, contour in enumerate(self._contour_definitions):
            path_item = QGraphicsPathItem()
            pen = QPen(theme.label_color(contour["class_id"]))
            pen.setWidthF(self.__contour_pen_width())
            pen.setCosmetic(True)  # width in screen pixels, updated on zoom
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            path_item.setPen(pen)
            path_item.setZValue(120)
            self.scene.addItem(path_item)
            handles = []
            for point_idx, _ in enumerate(contour["points"]):
                handle_item = self.__create_handle_item(pen.color())
                handle_info = {
                    "item": handle_item,
                    "contour_idx": idx,
                    "point_idx": point_idx,
                }
                handles.append(handle_info)
                self._handle_items.append(handle_info)
            self._contour_items.append({"path": path_item, "handles": handles})
        self.__refresh_contour_geometry()

    def __update_contour_path(self, contour_idx):
        if contour_idx >= len(self._contour_items):
            return
        contour = self._contour_definitions[contour_idx]
        path = self.__build_path(contour)
        self._contour_items[contour_idx]["path"].setPath(path)

    def __contour_pen_width(self) -> float:
        dr, dc = self._pixel_spacing_mm
        mm_per_native_pixel = (dr + dc) / 2.0
        if mm_per_native_pixel <= 0:
            return 2.0
        width = self._contour_line_mm * self._actual_scale / mm_per_native_pixel
        return max(1.0, min(width, 6.0))

    def __refresh_contour_geometry(self):
        if not self._contour_items:
            return
        for idx, contour in enumerate(self._contour_definitions):
            if idx >= len(self._contour_items):
                continue
            self._contour_items[idx]["path"].setPath(self.__build_path(contour))
            for handle in self._contour_items[idx]["handles"]:
                row, col = contour["points"][handle["point_idx"]]
                scene_point = self.__image_point_to_scene(col, row)
                handle["item"].setPos(scene_point)
        self.__update_handle_visibility()

    def __build_path(self, contour):
        points = contour.get("points", [])
        if not points:
            return QPainterPath()
        scene_points = [self.__image_point_to_scene(col, row) for (row, col) in points]

        painter_path = QPainterPath(scene_points[0])
        if len(scene_points) < 3:
            for point in scene_points[1:]:
                painter_path.lineTo(point)
            return painter_path

        def __distance(point_a, point_b):
            dx = point_b.x() - point_a.x()
            dy = point_b.y() - point_a.y()
            return (dx * dx + dy * dy) ** 0.5

        def __safe_div(value, denominator):
            if abs(denominator) < 1e-6:
                return 0.0
            return value / denominator

        num_points = len(scene_points)
        alpha = 0.5  # centripetal parameterization reduces loop/overshoot artifacts
        for idx in range(num_points):
            point0 = scene_points[(idx - 1) % num_points]
            point1 = scene_points[idx]
            point2 = scene_points[(idx + 1) % num_points]
            point3 = scene_points[(idx + 2) % num_points]

            t0 = 0.0
            t1 = t0 + (__distance(point0, point1) ** alpha)
            t2 = t1 + (__distance(point1, point2) ** alpha)
            t3 = t2 + (__distance(point2, point3) ** alpha)

            m1x = (t2 - t1) * (
                __safe_div(point1.x() - point0.x(), t1 - t0)
                - __safe_div(point2.x() - point0.x(), t2 - t0)
                + __safe_div(point2.x() - point1.x(), t2 - t1)
            )
            m1y = (t2 - t1) * (
                __safe_div(point1.y() - point0.y(), t1 - t0)
                - __safe_div(point2.y() - point0.y(), t2 - t0)
                + __safe_div(point2.y() - point1.y(), t2 - t1)
            )
            m2x = (t2 - t1) * (
                __safe_div(point2.x() - point1.x(), t2 - t1)
                - __safe_div(point3.x() - point1.x(), t3 - t1)
                + __safe_div(point3.x() - point2.x(), t3 - t2)
            )
            m2y = (t2 - t1) * (
                __safe_div(point2.y() - point1.y(), t2 - t1)
                - __safe_div(point3.y() - point1.y(), t3 - t1)
                + __safe_div(point3.y() - point2.y(), t3 - t2)
            )

            seg_len = __distance(point1, point2)

            def __clamp_tangent(mx, my, seg_len=seg_len):
                mag = (mx * mx + my * my) ** 0.5
                if mag > seg_len and mag > 1e-6:
                    scale = seg_len / mag
                    return mx * scale, my * scale
                return mx, my

            m1x, m1y = __clamp_tangent(m1x, m1y)
            m2x, m2y = __clamp_tangent(m2x, m2y)

            control1 = QPointF(point1.x() + (m1x / 3.0), point1.y() + (m1y / 3.0))
            control2 = QPointF(point2.x() - (m2x / 3.0), point2.y() - (m2y / 3.0))
            painter_path.cubicTo(control1, control2, point2)

        painter_path.closeSubpath()
        return painter_path

    def __create_handle_item(self, color):
        handle = QGraphicsEllipseItem(
            -self._handle_radius,
            -self._handle_radius,
            self._handle_radius * 2,
            self._handle_radius * 2,
        )
        handle.setBrush(color)
        handle.setPen(Qt.NoPen)
        handle.setFlag(QGraphicsItem.ItemIgnoresTransformations)  # same size at any zoom
        handle.setZValue(150)
        handle.setVisible(self.edit_mode)
        self.scene.addItem(handle)
        return handle

    def __update_handle_visibility(self):
        for handle in self._handle_items:
            handle["item"].setVisible(self.edit_mode)

    def __emit_contours_updated(self):
        if not self.edit_mode:
            return
        payload = []
        for contour in self._contour_definitions:
            points = contour.get("points", [])
            if len(points) < 3:
                continue
            payload.append(
                {
                    "class_id": int(contour.get("class_id", 0)),
                    "points": [(float(r), float(c)) for (r, c) in points],
                }
            )
        self.contours_updated.emit(payload)
