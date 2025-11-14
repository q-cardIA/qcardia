import logging
import os

import numpy as np

# from gui.UtilsWidgets.CustomQDialog.ImportDataQDialog import ImportDataQDialog
from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from view.utils import SoftwareConfigResources

# from scipy.ndimage import rotate  # Unused; keep disabled until rotation is required


class CustomQGraphicsView(QGraphicsView):
    """ """

    mri_volume_imported = Signal(
        str
    )  # From the drag/drop event to include more MRI volumes
    annotation_volume_imported = Signal(
        str
    )  # From the drag/drop event to include more annotations
    patient_imported = Signal(
        str
    )  # From the drag/drop event to include more patients to the project?
    coordinates_changed = Signal(
        int, int
    )  # From the mouse move event to change the point-of-view.
    annotation_display_state_changed = (
        Signal()
    )  # When pressing 's' to toggle on/off the annotations

    def __init__(self, view_type="axial", size=QSize(150, 150), parent=None):
        super(CustomQGraphicsView, self).__init__()
        self.parent = parent
        # self.setBaseSize(size)
        # self.setMinimumSize(size)
        # logging.debug("Setting CustomQGraphicsView dimensions to {}.\n".format(self.size()))
        self.view_type = view_type
        self.original_annotations = {}  # Placeholder for the raw annotation 2d slices
        self.display_annotations = (
            {}
        )  # Placeholder for the annotation 2d slices, after display transform
        self.overlaid_items = {}  # Placeholder for the QGraphicsPixmapItem
        # Placeholder for color and opacity for now: {color:QColor(255,255,255), opacity:0.5}
        self.overlaid_items_display_parameters = {}

        self.right_button_on_hold = False
        self.left_button_on_hold = False
        self.zoom_ratio = 1.0  # Zoom ratio in the range [0.5, 4.0]
        self.clicked_right_pos = None
        self.slice = None
        self.display_2d = None
        self.original_2d_point = None  # 2d point position in the original MRI volume
        self.adapted_2d_point = None  # 2d point position in the MRI volume adjusted for conventional neurological view
        self.graphical_2d_point = None  # 2d point position in the graphical view
        self.crosshair_pos = None
        self.horizontal_line = None  # Store horizontal crosshair line
        self.vertical_line = None  # Store vertical crosshair line
        self.dragging_crosshair = False  # Track if we're dragging the crosshair
        self._last_indices = None

        self.__set_interface()
        self.__set_stylesheets()

    def __set_interface(self):
        self.scene = QGraphicsScene(self)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setEnabled(
            False
        )  # @TODO. Should it be enabled, to allow drag and drop at first, just prevent clicking...

        image_2d = np.zeros((150, 150), dtype="uint8")
        h, w = image_2d.shape
        bytes_per_line = 3 * w
        color_image = np.stack([image_2d] * 3, axis=-1)
        qimage = QImage(color_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        self.pixmap = QPixmap.fromImage(qimage)
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)

        pen = QPen()
        pen.setColor(QColor(60, 255, 137))
        pen.setStyle(Qt.DashLine)
        # Remove persistent references to QGraphicsLineItem objects
        # Always add new lines after scene.clear()
        self.setScene(self.scene)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def __set_stylesheets(self):
        self.setStyleSheet("QGraphicsView{background-color:rgb(0,0,0);}")

        self.setStyleSheet(
            """
            QGraphicsView {
            border: none;
            background-color: black;
                }
        
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0px;
            }
        
            QScrollBar::handle:vertical {
                background: #666;
                border-radius: 4px;
                min-height: 20px;
            }
        
            QScrollBar::handle:vertical:hover {
                background: #aaa;
            }
        
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        
            QScrollBar:horizontal {
                background: transparent;
                height: 8px;
                margin: 0px;
            }
        
            QScrollBar::handle:horizontal {
                background: #666;
                border-radius: 4px;
                min-width: 20px;
            }
        
            QScrollBar::handle:horizontal:hover {
                background: #aaa;
            }
        
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
            }
        """
        )

    def resizeEvent(self, event):
        """
        Ensure the image fits the current viewport size before any
        crosshair interaction by recomputing scaling/centering on
        resize/show.
        """
        super().resizeEvent(event)
        if getattr(self, "display_2d", None) is not None:
            # Recompute fit/offset using the latest viewport size
            self.update_slice_view(self.display_2d)

    def keyPressEvent(self, event) -> None:
        """
        Shortcuts for displaying or hiding elements, only if there is an active patient.
        S key: display or hide the tumor annotation.
        """
        if event.key() == Qt.Key_S:
            if not SoftwareConfigResources.getInstance().is_patient_list_empty():
                logging.info("Changing annotations display state.")
                self.annotation_display_state_changed.emit()

    def mousePressEvent(self, event):
        """
        Handle mouse press events for crosshair interaction and zoom.
        """
        pos = self.mapToScene(QPoint(event.localPos().x(), event.localPos().y()))

        if event.button() == Qt.LeftButton:
            # Check if clicking on either crosshair line
            items = self.scene.items(pos)
            clicked_line = None
            for item in items:
                if item == self.vertical_line:
                    clicked_line = "vertical"
                    break
                elif item == self.horizontal_line:
                    clicked_line = "horizontal"
                    break

            if clicked_line:
                self.dragging_crosshair = clicked_line
                self.left_button_on_hold = True
                if clicked_line == "vertical":
                    self.updateCrosshairPosition(pos.x(), self.crosshair_pos[1])
                else:
                    self.updateCrosshairPosition(self.crosshair_pos[0], pos.y())

                volx, voly = self.__from_graphics_position_to_raw_volume_position(pos)
                self.coordinates_changed.emit(volx, voly)
            else:
                self.left_button_on_hold = True
                self.updateCrosshairPosition(pos.x(), pos.y())
                volx, voly = self.__from_graphics_position_to_raw_volume_position(pos)
                self.coordinates_changed.emit(volx, voly)

        if event.button() == Qt.RightButton:
            self.clicked_right_pos = event.globalPos()
            self.right_button_on_hold = True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_button_on_hold = False
            self.dragging_crosshair = None
        elif event.button() == Qt.RightButton:
            self.right_button_on_hold = False

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(QPoint(event.localPos().x(), event.localPos().y()))

        if self.left_button_on_hold:
            if self.dragging_crosshair:
                # Update only the line being dragged
                if self.dragging_crosshair == "vertical":
                    self.updateCrosshairPosition(pos.x(), self.crosshair_pos[1])
                else:
                    self.updateCrosshairPosition(self.crosshair_pos[0], pos.y())

                volx, voly = self.__from_graphics_position_to_raw_volume_position(pos)
                self.coordinates_changed.emit(volx, voly)
            else:
                # Normal left-click drag behavior
                self.updateCrosshairPosition(pos.x(), pos.y())
                volx, voly = self.__from_graphics_position_to_raw_volume_position(pos)
                self.coordinates_changed.emit(volx, voly)

        elif self.right_button_on_hold:
            if event.globalPos().y() > self.clicked_right_pos.y():
                self.zoom_ratio = min(4.0, self.zoom_ratio + 0.025)
            else:
                self.zoom_ratio = max(0.5, self.zoom_ratio - 0.025)
            self.__repaint_view()
            self.clicked_right_pos = event.globalPos()

    def updateCrosshairPosition(self, x, y):
        """
        Update both crosshair lines to a new position
        """
        if x < 0:
            x = 0
        elif x >= self.width():
            x = self.width() - 1
        if y < 0:
            y = 0
        elif y >= self.height():
            y = self.height() - 1

        # Store the position even if lines don't exist yet
        self.crosshair_pos = (x, y)

        # Adjust coordinates based on view type
        view_height = self.height()
        view_width = self.width()

        # Only update lines if they exist
        if self.vertical_line is not None:
            if self.view_type == "sagittal":
                # In sagittal view, vertical line moves up/down (y)
                self.vertical_line.setLine(x, 0, x, view_height)
            else:
                # In axial and coronal views, vertical line moves left/right (x)
                self.vertical_line.setLine(x, 0, x, view_height)

        if self.horizontal_line is not None:
            if self.view_type == "coronal":
                # In coronal view, horizontal line moves up/down (y)
                self.horizontal_line.setLine(0, y, view_width, y)
            else:
                # In axial and sagittal views, horizontal line moves up/down (y)
                self.horizontal_line.setLine(0, y, view_width, y)

        # If lines don't exist and we have a scene, create them
        if self.scene and (self.vertical_line is None or self.horizontal_line is None):
            pen = QPen(QColor(60, 255, 137))
            pen.setStyle(Qt.DashLine)
            pen.setWidth(2)

            if self.vertical_line is None:
                self.vertical_line = self.scene.addLine(x, 0, x, view_height, pen)
                self.vertical_line.setFlag(QGraphicsItem.ItemIsMovable)
                self.vertical_line.setFlag(QGraphicsItem.ItemIsSelectable)
                self.vertical_line.setData(0, "vertical")

            if self.horizontal_line is None:
                self.horizontal_line = self.scene.addLine(0, y, view_width, y, pen)
                self.horizontal_line.setFlag(QGraphicsItem.ItemIsMovable)
                self.horizontal_line.setFlag(QGraphicsItem.ItemIsSelectable)
                self.horizontal_line.setData(0, "horizontal")

    def dragEnterEvent(self, event):
        entered_content_raw = event.mimeData().urls()
        entered_content_list = [
            x.toLocalFile() for x in entered_content_raw
        ]  # entered_content_raw.strip().split("\n")
        eligibility_file_states = [
            ".".join(os.path.basename(x).split(".")[1:]).strip()
            in SoftwareConfigResources.getInstance().accepted_image_format
            for x in entered_content_list
        ]
        eligibility_dir_states = [
            os.path.isdir(x.strip()) for x in entered_content_list
        ]
        accept_state = True in (eligibility_file_states or eligibility_dir_states)

        if accept_state:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        pass

    def dragLeaveEvent(self, event):
        event.ignore()

    def dropEvent(self, event):
        """
        Enabling the possibility to open / add to the current patient, a new volume
        """
        # entered_content_raw = event.mimeData().text()
        # entered_content_list = entered_content_raw.strip().split("\n")
        entered_content_raw = event.mimeData().urls()
        entered_content_list = [x.toLocalFile() for x in entered_content_raw]
        entered_eligible_files = []
        entered_eligible_dir = []
        for elem in entered_content_list:
            if os.path.isdir(elem):
                entered_eligible_dir.append(elem)
            elif (
                ".".join(os.path.basename(elem).split(".")[1:]).strip()
                in SoftwareConfigResources.getInstance().accepted_image_format
            ):
                entered_eligible_files.append(elem.strip())

        # dialog = ImportDataQDialog(self)
        # dialog.mri_volume_imported.connect(self.mri_volume_imported)
        # dialog.annotation_volume_imported.connect(self.annotation_volume_imported)
        # dialog.patient_imported.connect(self.patient_imported)

        # dialog.setup_interface_from_files(entered_eligible_files)
        # code = dialog.exec()

        # if code == QDialog.Accepted:

    def update_overlay(self, overlay_image):
        """Store the overlay and trigger a redraw with compositing."""
        if overlay_image is None:
            self.overlay_data = None
        else:
            # Apply the same transform as the base image
            # (horizontal flip for display)
            if overlay_image.ndim == 3 and overlay_image.shape[2] in (3, 4):
                # Color/RGBA overlay
                self.overlay_data = overlay_image[:, ::-1, :].copy()
            else:
                # Unexpected shape; keep a safe copy
                self.overlay_data = overlay_image.copy()
        # Redraw the slice view to composite overlay
        if hasattr(self, "display_2d") and self.display_2d is not None:
            self.update_slice_view(self.display_2d)

    def update_annotation_view(self, annotation_uid, annotation_slice):
        """ """
        if annotation_uid not in self.overlaid_items.keys():
            self.original_annotations[annotation_uid] = None
            self.display_annotations[annotation_uid] = None
            self.overlaid_items[annotation_uid] = QGraphicsPixmapItem()
            self.scene.addItem(self.overlaid_items[annotation_uid])
        if annotation_uid not in self.overlaid_items_display_parameters.keys():
            annotation_color = (
                SoftwareConfigResources.getInstance()
                .get_active_patient()
                .get_annotation_by_uid(annotation_uid)
                .display_color
            )
            self.overlaid_items_display_parameters[annotation_uid] = {
                "color": QColor.fromRgb(
                    annotation_color[0],
                    annotation_color[1],
                    annotation_color[2],
                    annotation_color[3],
                ),
                "opacity": float(
                    SoftwareConfigResources.getInstance()
                    .get_active_patient()
                    .get_annotation_by_uid(annotation_uid)
                    .get_display_opacity()
                    / 100.0
                ),
            }

        self.original_annotations[annotation_uid] = annotation_slice
        # Apply consistent display transform for all views (flip + rotate +180)
        image_2d = annotation_slice[:, ::-1]
        # image_2d = rotate(image_2d, 180).astype("uint8")
        self.display_annotations[annotation_uid] = image_2d
        self.__repaint_overlay(annotation_uid)

    def reset_viewer(self):
        """
        Not used as of now, hasn't been tried either.
        """
        self.cleanse_annotations()
        self.update_slice_view(np.zeros((150, 150), dtype="uint8"))

    def cleanse_annotations(self):
        for k in list(self.overlaid_items):
            self.remove_annotation_view(k)
        self.overlaid_items_display_parameters.clear()

    def remove_annotation_view(self, annotation_uid):
        graphics_item = self.overlaid_items[annotation_uid]
        self.scene.removeItem(graphics_item)
        self.overlaid_items.pop(annotation_uid)
        self.original_annotations.pop(annotation_uid)
        self.display_annotations.pop(annotation_uid)
        # Should we keep the last parameters, so that it shows back as it was when it's toggled back, rather than
        # set to default values? Not much memory use for this.
        # self.overlaid_items_display_parameters.pop(annotation_uid)

    def update_annotation_opacity(self, annotation_uid, value):
        """
        Opacity value comes from a QSlider with a value in the range [0, 100]
        """
        self.overlaid_items_display_parameters[annotation_uid]["opacity"] = (
            value / 100.0
        )
        self.__repaint_overlay(annotation_uid)

    def update_annotation_color(self, annotation_uid, color):
        """
        Update to the display color for the current annotation, indicated by annotation_uid.
        """
        self.overlaid_items_display_parameters[annotation_uid]["color"] = color
        self.__repaint_overlay(annotation_uid)

    def __update_point_clicker_lines(self, posx, posy):
        """
        Updates the location of the crosshair lines.
        """
        self.updateCrosshairPosition(posx, posy)

    def __from_graphics_position_to_raw_volume_position(self, graphics_point):
        """
        Conversion between the graphical position in the QGraphicsScene referential and the voxel indices
        in the original 2D slice using the display orientation emitted to the other views.
        """
        if (
            self.image_item is None
            or self.image_item.pixmap().isNull()
            or self.image_2d_h == 0
            or self.image_2d_w == 0
        ):
            return 0, 0

        local_point = self.image_item.mapFromScene(graphics_point)
        pixmap = self.image_item.pixmap()
        offset = self.image_item.offset()

        pixmap_width = max(1, pixmap.width())
        pixmap_height = max(1, pixmap.height())

        local_x = local_point.x() - offset.x()
        local_y = local_point.y() - offset.y()

        local_x = max(0.0, min(pixmap_width - 1, local_x))
        local_y = max(0.0, min(pixmap_height - 1, local_y))

        col_ratio = local_x / (pixmap_width - 1) if pixmap_width > 1 else 0.0
        row_ratio = local_y / (pixmap_height - 1) if pixmap_height > 1 else 0.0

        col_index = (self.image_2d_w - 1) * col_ratio
        row_index = (self.image_2d_h - 1) * row_ratio

        col_index = max(0.0, min(self.image_2d_w - 1, col_index))
        row_index = max(0.0, min(self.image_2d_h - 1, row_index))

        self.adapted_2d_point = QPointF(local_x, local_y)
        row_int = int(round(row_index))
        col_int = int(round(col_index))
        self._last_indices = (row_int, col_int)

        return col_int, row_int

    def set_crosshair_from_volume_indices(self, row_index, col_index):
        if (
            self.image_item is None
            or self.image_item.pixmap().isNull()
            or self.image_2d_h == 0
            or self.image_2d_w == 0
        ):
            self._last_indices = (int(row_index), int(col_index))
            return

        clamped_row = max(0, min(int(row_index), self.image_2d_h - 1))
        clamped_col = max(0, min(int(col_index), self.image_2d_w - 1))

        pixmap = self.image_item.pixmap()
        pixmap_width = max(1, pixmap.width())
        pixmap_height = max(1, pixmap.height())

        if self.image_2d_h > 1:
            row_ratio = clamped_row / (self.image_2d_h - 1)
        else:
            row_ratio = 0.0

        if self.image_2d_w > 1:
            col_ratio = clamped_col / (self.image_2d_w - 1)
        else:
            col_ratio = 0.0

        local_x = col_ratio * (pixmap_width - 1)
        local_y = row_ratio * (pixmap_height - 1)
        offset = self.image_item.offset()
        scene_point = self.image_item.mapToScene(
            QPointF(local_x + offset.x(), local_y + offset.y())
        )

        self._last_indices = (clamped_row, clamped_col)
        self.adapted_2d_point = QPointF(local_x, local_y)
        self.updateCrosshairPosition(scene_point.x(), scene_point.y())

    def update_slice_view(self, slice_2d, x=100, y=100):
        self.setEnabled(True)
        import numpy as np

        # Apply consistent display transform to match expected view (flip + rotate +180)
        slice_2d = np.ascontiguousarray(slice_2d)
        image_2d = slice_2d  # [:, ::-1]
        # image_2d = rotate(image_2d, 180).astype("uint8")
        self.display_2d = image_2d
        h, w = image_2d.shape
        self.image_2d_h = h
        self.image_2d_w = w
        if self._last_indices is None:
            self.zoom_ratio = 1.0

        base_rgba = np.stack(
            [image_2d] * 3 + [np.full_like(image_2d, 255)], axis=-1
        ).astype(np.uint8)

        if hasattr(self, "overlay_data") and self.overlay_data is not None:
            overlay = self.overlay_data
            if overlay.shape[:2] != base_rgba.shape[:2]:
                from cv2 import resize

                overlay = resize(
                    overlay, (base_rgba.shape[1], base_rgba.shape[0]), interpolation=3
                )
            alpha_overlay = overlay[..., 3:4] / 255.0
            alpha_base = 1.0 - alpha_overlay
            base_rgba[..., :3] = (
                base_rgba[..., :3] * alpha_base + overlay[..., :3] * alpha_overlay
            ).astype(np.uint8)
            base_rgba[..., 3] = 255

        qimage = QImage(base_rgba.data, w, h, QImage.Format_RGBA8888)
        # Fit image to current viewport while preserving aspect ratio
        view_w = max(1, self.viewport().width())
        view_h = max(1, self.viewport().height())
        fit_scale_w = view_w / w if w else 1.0
        fit_scale_h = view_h / h if h else 1.0
        fit_scale = min(fit_scale_w, fit_scale_h)
        fit_scale = max(fit_scale, 1e-6)
        self.zoom_ratio = max(0.5, min(self.zoom_ratio, 4.0))
        actual_scale = fit_scale * self.zoom_ratio
        scaled_width = max(1, int(round(w * actual_scale)))
        scaled_height = max(1, int(round(h * actual_scale)))

        # Create pixmap for the base image
        qimage_scaled = qimage.scaled(
            scaled_width,
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        pixmap = QPixmap.fromImage(qimage_scaled)
        self.image_item.setPixmap(pixmap)

        # Center the image within the viewport
        offset_x = max(0, int((view_w - pixmap.width()) / 2))
        offset_y = max(0, int((view_h - pixmap.height()) / 2))
        self.image_item.setOffset(offset_x, offset_y)
        self.scene.setSceneRect(0, 0, view_w, view_h)

        # Repaint any overlays to match the new base pixmap size
        for overlay_uid in list(self.display_annotations.keys()):
            self.__repaint_overlay(overlay_uid)

        # Crosshair lines: create centered on first paint, refresh thereafter
        if self.crosshair_pos is None:
            center_x = int(offset_x + max(1, pixmap.width()) / 2)
            center_y = int(offset_y + max(1, pixmap.height()) / 2)
            self.updateCrosshairPosition(center_x, center_y)
        else:
            self.updateCrosshairPosition(self.crosshair_pos[0], self.crosshair_pos[1])

        # Atlas-related overlay helpers removed (feature not used)

    def __repaint_view(self):
        """
        Full repainting of the scene, including the MRI volume, point-position lines,
        and overlays.
        """
        if self.display_2d is None:
            return

        self.update_slice_view(self.display_2d)

    def __repaint_overlay(self, overlay_uid):
        """
        Recomputes the QPixmap, updates the QGraphicsPixmapItem, which repaints the scene accordingly for the
        specified annotation to overlay indicated by overlay_uid.
        """
        if overlay_uid not in self.display_annotations:
            return

        overlay_image = self.display_annotations[overlay_uid]
        if overlay_image is None:
            return

        if overlay_uid not in self.overlaid_items_display_parameters:
            return

        if self.image_item is None or self.image_item.pixmap().isNull():
            return

        h, w = overlay_image.shape
        bytes_per_line = 3 * w
        color_image = np.stack(
            [np.zeros(overlay_image.shape, dtype=np.uint8)] * 3, axis=-1
        )
        color = self.overlaid_items_display_parameters[overlay_uid]["color"]
        color_image[..., 0][overlay_image != 0] = color.red()
        color_image[..., 1][overlay_image != 0] = color.green()
        color_image[..., 2][overlay_image != 0] = color.blue()

        qimage = QImage(color_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)

        alpha_channel = np.zeros(overlay_image.shape, dtype=np.uint8)
        alpha_channel[overlay_image != 0] = 255
        qimage.setAlphaChannel(
            QImage(alpha_channel.data, w, h, w, QImage.Format_Alpha8)
        )

        pixmap = self.image_item.pixmap()
        target_width = max(1, pixmap.width())
        target_height = max(1, pixmap.height())
        qimage_scaled = qimage.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        graphics_item = self.overlaid_items[overlay_uid]
        graphics_item.setPixmap(QPixmap.fromImage(qimage_scaled))
        graphics_item.setOffset(self.image_item.offset())
        graphics_item.setZValue(1)

        opacity = QGraphicsOpacityEffect()
        opacity.setOpacity(
            self.overlaid_items_display_parameters[overlay_uid]["opacity"]
        )
        graphics_item.setGraphicsEffect(opacity)
