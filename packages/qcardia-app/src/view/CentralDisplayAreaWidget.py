import logging
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_dilation, binary_erosion

from view.CustomQGraphicsView import CustomQGraphicsView
from view.utils import SoftwareConfigResources

# add central area execution widget


class CentralDisplayAreaWidget(QWidget):
    # """ """

    import_data_triggered = Signal()
    # Unique id for the MRI volume, belonging to the active patient
    mri_volume_imported = Signal(str)
    # Unique id for the annotation volume, belonging to the active patient
    annotation_volume_imported = Signal(str)
    patient_imported = Signal(str)  # The str is the unique id for the patient
    annotation_display_state_changed = Signal()

    def __init__(self, parent=None):
        super(CentralDisplayAreaWidget, self).__init__()
        self.parent = parent
        # Previous dynamic min size logic removed for clarity.
        opt = SoftwareConfigResources.getInstance().get_optimal_dimensions()
        self.setBaseSize(
            QSize(
                (1325 / opt.width()) * self.parent.baseSize().width(),
                (950 / opt.height()) * self.parent.baseSize().height(),
            )
        )
        logging.debug("Setting CentralDisplayAreaWidget dimensions to %s.", self.size())
        self.__set_interface()
        # self.__set_layout_dimensions()
        self.__set_stylesheets()
        self.__set_connections()
        self.current_patient_parameters = None
        self.displayed_image = None
        self.displayed_image_uid = None
        self.point_clicker_position = [
            0,
            0,
            0,
        ]  # Knowing at all time the center of the cross-hair blue lines.
        self.overlaid_volumes = {}  # Annotation volumes to overlay on the image.
        # Segmentation: class ids per voxel, shape (Z, H, W)
        self.segmentation_volume = None
        self.primary_slice_positions = (
            None  # Physical locations along primary volume depth
        )
        self.lge_slice_positions = None  # Physical locations along LGE stack
        self.lge_segmentation_volume = None
        self.contour_edit_mode = False

    def resizeEvent(self, event):
        # Default resize behavior retained.
        super().resizeEvent(event)

    def __set_interface(self):
        self.setStyleSheet("QWidget{background-color:#1E1E1E;}")
        self.layout = QGridLayout(self)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        # LGE volume viewer container (top-right)
        self.lge_volume = None
        self.lge_slice_index = 0
        self.lge_container = QWidget()
        self.lge_container.setStyleSheet("QWidget{background-color:#1E1E1E;}")
        self.lge_container_layout = QVBoxLayout(self.lge_container)
        self.lge_container_layout.setContentsMargins(2, 2, 2, 2)
        self.lge_container_layout.setSpacing(2)
        self.lge_header = self.__make_viewer_header("LGE Plane")
        self.secondary_viewer = CustomQGraphicsView(
            view_type="lge", size=QSize(100, 100), parent=self
        )
        self.secondary_slider = self.__make_slice_slider()
        self.secondary_slider.valueChanged.connect(self.__on_secondary_slider_changed)
        self.lge_container_layout.addLayout(self.lge_header)
        self.lge_container_layout.addWidget(self.secondary_slider)
        self.lge_container_layout.addWidget(self.secondary_viewer)
        half_w = int(self.minimumWidth() / 2)
        half_h = int(self.minimumHeight() / 2)
        common_size = QSize(half_w, half_h)
        # Wrap each primary viewer in a container with header + slider
        (
            self.sagittal_container,
            self.sagittal_viewer,
            self.sagittal_slider,
            self.sagittal_header_labels,
        ) = self.__make_plane_view(
            "Sagittal Plane", "sagittal", common_size, self.__on_sagittal_slider_changed
        )
        (
            self.axial_container,
            self.axial_viewer,
            self.axial_slider,
            self.axial_header_labels,
        ) = self.__make_plane_view(
            "Transverse Plane", "axial", common_size, self.__on_axial_slider_changed
        )
        (
            self.coronal_container,
            self.coronal_viewer,
            self.coronal_slider,
            self.coronal_header_labels,
        ) = self.__make_plane_view(
            "Coronal Plane", "coronal", common_size, self.__on_coronal_slider_changed
        )

        self.horizontal_line = QLabel()
        self.horizontal_line.setFixedHeight(3)
        self.vertical_line = QLabel()
        self.vertical_line.setFixedWidth(3)
        # Place containers (keeping existing spatial semantics for now)
        self.layout.addWidget(self.sagittal_container, 0, 0)
        self.layout.addWidget(self.lge_container, 0, 2)
        self.layout.addWidget(self.horizontal_line, 1, 0, 1, 3)
        self.layout.addWidget(self.axial_container, 2, 0)
        self.layout.addWidget(self.coronal_container, 2, 2)
        self.layout.addWidget(self.vertical_line, 0, 1, 3, 1)

    # Removed unused __set_layout_dimensions (previously only comments).

    def __set_stylesheets(self):
        dark_view = (
            "QGraphicsView{background-color:rgb(0,0,0); border:1px solid #2F3440;}"
        )
        self.secondary_viewer.setStyleSheet(dark_view)
        self.axial_viewer.setStyleSheet(dark_view)
        self.coronal_viewer.setStyleSheet(dark_view)
        self.sagittal_viewer.setStyleSheet(dark_view)
        self.horizontal_line.setStyleSheet("""QLabel{background-color: #2F3440;}""")
        self.vertical_line.setStyleSheet("""QLabel{background-color: #2F3440;}""")

    def __set_connections(self):
        self.axial_viewer.coordinates_changed.connect(
            self.__on_axial_coordinates_changed
        )
        self.coronal_viewer.coordinates_changed.connect(
            self.__on_coronal_coordinates_changed
        )
        self.sagittal_viewer.coordinates_changed.connect(
            self.__on_sagittal_coordinates_changed
        )
        self.secondary_viewer.coordinates_changed.connect(
            self.__on_lge_coordinates_changed
        )

        self.axial_viewer.mri_volume_imported.connect(self.mri_volume_imported)
        self.axial_viewer.annotation_volume_imported.connect(
            self.annotation_volume_imported
        )
        self.axial_viewer.annotation_display_state_changed.connect(
            self.annotation_display_state_changed
        )
        # self.axial_viewer.patient_imported.connect(self.patient_imported)
        self.coronal_viewer.mri_volume_imported.connect(self.mri_volume_imported)
        self.coronal_viewer.annotation_volume_imported.connect(
            self.annotation_volume_imported
        )
        self.coronal_viewer.annotation_display_state_changed.connect(
            self.annotation_display_state_changed
        )
        # self.coronal_viewer.patient_imported.connect(self.patient_imported)
        self.sagittal_viewer.mri_volume_imported.connect(self.mri_volume_imported)
        self.sagittal_viewer.annotation_volume_imported.connect(
            self.annotation_volume_imported
        )
        self.sagittal_viewer.annotation_display_state_changed.connect(
            self.annotation_display_state_changed
        )
        self.sagittal_viewer.contours_updated.connect(
            self.__on_sagittal_contours_updated
        )
        # self.sagittal_viewer.patient_imported.connect(self.patient_imported)

    def reset_viewer(self):
        """ """
        self.reset_overlay()
        self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
        self.primary_slice_positions = None
        self.displayed_image_uid = None
        self.point_clicker_position = [
            int(self.displayed_image.shape[0] / 2),
            int(self.displayed_image.shape[1] / 2),
            int(self.displayed_image.shape[2] / 2),
        ]
        self.update_viewers_image()

    def reset_overlay(self):
        """
        When the active patient or volume changes, clear all overlays.
        """
        self.axial_viewer.cleanse_annotations()
        self.coronal_viewer.cleanse_annotations()
        self.sagittal_viewer.cleanse_annotations()
        self.sagittal_viewer.clear_contours()
        self.overlaid_volumes.clear()

    def on_import_data(self):
        """
        DEPRECATED: To Remove.
        """
        if self.displayed_image is None:
            self.current_patient_parameters = (
                SoftwareConfigResources.getInstance().patients_parameters[
                    SoftwareConfigResources.getInstance().active_patient_name
                ]
            )

            # If the active patient has MRI volumes, display the first one.
            if self.current_patient_parameters.get_patient_mri_volumes_number() != 0:
                self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
                    self.current_patient_parameters.get_all_mri_volumes_uids()[0]
                ).display_volume
                self.primary_slice_positions = None
                self.point_clicker_position = [
                    int(self.displayed_image.shape[0] / 2),
                    int(self.displayed_image.shape[1] / 2),
                    int(self.displayed_image.shape[2] / 2),
                ]
                self.displayed_image = self.displayed_image[:, ::-1, :]
                self.primary_slice_positions = None
                self.update_viewers_image()
            else:
                self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
                self.primary_slice_positions = None
                self.point_clicker_position = [
                    int(self.displayed_image.shape[0] / 2),
                    int(self.displayed_image.shape[1] / 2),
                    int(self.displayed_image.shape[2] / 2),
                ]
                self.update_viewers_image()
                # self.axial_viewer.setEnabled(False)
                # self.coronal_viewer.setEnabled(False)
                # self.sagittal_viewer.setEnabled(False)

    def on_patient_selected(self):
        self.current_patient_parameters = (
            SoftwareConfigResources.getInstance().patients_parameters[
                SoftwareConfigResources.getInstance().active_patient_name
            ]
        )

        self.reset_overlay()
        # If active patient has MRI volumes, display current or first.
        if self.current_patient_parameters.get_patient_mri_volumes_number() != 0:
            if (
                self.displayed_image_uid
                in self.current_patient_parameters.get_all_mri_volumes_uids()
            ):
                self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
                    self.displayed_image_uid
                ).display_volume
                self.primary_slice_positions = None
            else:  # New patient selected: use first volume
                first_uid = self.current_patient_parameters.get_all_mri_volumes_uids()[
                    0
                ]
                self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
                    first_uid
                ).display_volume
                self.primary_slice_positions = None
                self.displayed_image_uid = first_uid
            self.point_clicker_position = [
                int(self.displayed_image.shape[0] / 2),
                int(self.displayed_image.shape[1] / 2),
                int(self.displayed_image.shape[2] / 2),
            ]
        else:  # Empty patient fallback
            self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
            self.primary_slice_positions = None
            self.displayed_image_uid = None
            self.point_clicker_position = [
                int(self.displayed_image.shape[0] / 2),
                int(self.displayed_image.shape[1] / 2),
                int(self.displayed_image.shape[2] / 2),
            ]
        self.update_viewers_image()

    def on_volume_layer_toggled(self, volume_uid, state):
        """Toggle display of an MRI volume (state should normally be True)."""
        if not self.current_patient_parameters:
            self.current_patient_parameters = (
                SoftwareConfigResources.getInstance().get_active_patient()
            )

        try:
            if state:
                # Clear overlays until multi-volume co-registration is supported.
                self.reset_overlay()
                self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
                    volume_uid
                ).display_volume
                self.primary_slice_positions = None
                self.displayed_image_uid = volume_uid

                # Reset viewpoint to center.
                self.point_clicker_position = [
                    int(self.displayed_image.shape[0] / 2),
                    int(self.displayed_image.shape[1] / 2),
                    int(self.displayed_image.shape[2] / 2),
                ]
                self.update_viewers_image()
            else:  # If all images have been removed by the user, should display an empty view
                self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
                self.primary_slice_positions = None
                self.displayed_image_uid = None
                self.point_clicker_position = [
                    int(self.displayed_image.shape[0] / 2),
                    int(self.displayed_image.shape[1] / 2),
                    int(self.displayed_image.shape[2] / 2),
                ]
                self.update_viewers_image()
        except (
            Exception
        ) as e:  # Broad catch retained until specific exceptions are known.
            logging.error(
                "[Software error][CentralDisplayAreaWidget] Issue toggling view for volume: %s",
                volume_uid,
            )
            logging.info("Collected: %s\n %s", e, traceback.format_exc())

    def on_volume_contrast_changed(self, volume_uid):
        # @TODO. Should group the viewer calls into another function somewhere, since all three above methods use it.
        self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
            volume_uid
        ).display_volume
        self.primary_slice_positions = None
        self.displayed_image_uid = volume_uid
        self.update_viewers_image()

    def update_viewers_image(self):
        """
        Further send the current display image, at the selected point clicker position, to each of the 2D viewers
        that are enabled in this widget (i.e., axial, coronal, and sagittal currently)
        """
        if self.displayed_image is None:
            return

        self.__ensure_primary_slice_positions()

        self.axial_viewer.update_slice_view(
            self.displayed_image[:, :, self.point_clicker_position[2]],
            self.point_clicker_position[0],
            self.point_clicker_position[1],
        )
        self.coronal_viewer.update_slice_view(
            self.displayed_image[:, self.point_clicker_position[1], :],
            self.point_clicker_position[0],
            self.point_clicker_position[2],
        )
        self.sagittal_viewer.update_slice_view(
            self.displayed_image[self.point_clicker_position[0], :, :],
            self.point_clicker_position[1],
            self.point_clicker_position[2],
        )

        # for volume_uid, volume in self.overlaid_volumes.items():
        #     # Display segmentation only on sagittal (top-left). Use correct sagittal slice: volume[x, :, :]
        #     sagittal_slice = volume[self.point_clicker_position[0], :, :]

        #     self.sagittal_viewer.update_annotation_view(volume_uid, sagittal_slice)

        #     # Ensure other viewers do NOT retain stale overlays
        #     if volume_uid in self.axial_viewer.overlaid_items:
        #         self.axial_viewer.remove_annotation_view(volume_uid)
        #     if volume_uid in self.coronal_viewer.overlaid_items:
        #         self.coronal_viewer.remove_annotation_view(volume_uid)

        self.__synchronize_crosshairs()
        # After base slices updated, refresh dynamic segmentation overlay
        self.__update_segmentation_overlay()
        # Refresh LGE volume if present
        self.__sync_lge_with_primary()
        # Sync sliders positions
        self.__sync_primary_sliders()

    def set_lge_volume(self, volume, slice_positions=None):
        """Assign an LGE 3D volume (Z,H,W) plus optional physical slice positions."""
        if volume is None or volume.ndim != 3:
            return
        self.lge_volume = volume
        self.lge_segmentation_volume = None
        self.lge_slice_index = volume.shape[0] // 2
        if slice_positions is not None and len(slice_positions) == volume.shape[0]:
            self.lge_slice_positions = [float(val) for val in slice_positions]
        else:
            self.lge_slice_positions = [float(i) for i in range(volume.shape[0])]
        self.secondary_slider.setMaximum(max(0, volume.shape[0] - 1))
        self.secondary_slider.setValue(self.lge_slice_index)
        self.secondary_slider.setEnabled(volume.shape[0] > 1)
        self.secondary_viewer.update_overlay(None)
        self.__sync_lge_with_primary()

    def __update_lge_viewer(self):
        if self.lge_volume is None:
            return
        z = min(max(0, self.lge_slice_index), self.lge_volume.shape[0] - 1)
        slice_2d = self.lge_volume[z, :, :]
        self.secondary_viewer.update_slice_view(slice_2d)
        self.__update_lge_crosshair()
        self.__update_lge_segmentation_overlay()

    def __on_secondary_slider_changed(self, value):
        if self.lge_volume is None:
            return
        max_lge = self.lge_volume.shape[0] - 1
        clamped_value = min(max(0, value), max_lge)
        self.lge_slice_index = clamped_value

        if self.displayed_image is None:
            self.__update_lge_viewer()
            return

        self.__ensure_primary_slice_positions()
        ratio = self.__index_to_ratio(
            clamped_value, self.lge_slice_positions, self.lge_volume.shape[0]
        )
        primary_depth = self.displayed_image.shape[0]
        new_primary = self.__ratio_to_index(
            ratio, self.primary_slice_positions, primary_depth
        )
        new_primary = min(max(0, new_primary), primary_depth - 1)

        if self.point_clicker_position[0] != new_primary:
            self.point_clicker_position[0] = new_primary
            self.update_viewers_image()
        else:
            self.__update_lge_viewer()

    def __sync_lge_with_primary(self):
        if self.lge_volume is None or self.displayed_image is None:
            return

        self.__ensure_primary_slice_positions()
        primary_depth = self.displayed_image.shape[0]
        ratio = self.__index_to_ratio(
            self.point_clicker_position[0],
            self.primary_slice_positions,
            primary_depth,
        )
        target_index = self.__ratio_to_index(
            ratio, self.lge_slice_positions, self.lge_volume.shape[0]
        )

        if target_index != self.lge_slice_index:
            self.lge_slice_index = target_index

        block_state = self.secondary_slider.blockSignals(True)
        self.secondary_slider.setValue(self.lge_slice_index)
        self.secondary_slider.blockSignals(block_state)

        self.__update_lge_viewer()
        self.__update_lge_crosshair()

    def __ensure_primary_slice_positions(self):
        if self.displayed_image is None:
            self.primary_slice_positions = None
            return
        depth = self.displayed_image.shape[0]
        if (
            self.primary_slice_positions is None
            or len(self.primary_slice_positions) != depth
        ):
            self.primary_slice_positions = [float(i) for i in range(depth)]

    @staticmethod
    def __index_to_ratio(index, positions, length):
        if length <= 1:
            return 0.0
        if positions is not None and len(positions) == length:
            start = float(positions[0])
            end = float(positions[-1])
            if end == start:
                return 0.0
            value = float(positions[index])
            return (value - start) / (end - start)
        return index / float(length - 1)

    @staticmethod
    def __ratio_to_index(ratio, positions, length):
        if length <= 0:
            return 0
        ratio = max(0.0, min(1.0, float(ratio)))
        if positions is not None and len(positions) == length:
            start = float(positions[0])
            end = float(positions[-1])
            value = start + ratio * (end - start)
            closest = min(
                range(length),
                key=lambda idx: abs(float(positions[idx]) - value),
            )
            return closest
        if length == 1:
            return 0
        return int(round(ratio * (length - 1)))

    @staticmethod
    def __map_index_between_volumes(source_index, source_len, target_len):
        if target_len <= 0:
            return 0
        if source_len <= 1:
            return min(target_len - 1, max(0, int(source_index)))
        ratio = max(0.0, min(1.0, float(source_index) / float(source_len - 1)))
        return int(round(ratio * (target_len - 1)))

    def __update_lge_crosshair(self):
        if self.lge_volume is None:
            return

        height = self.lge_volume.shape[1]
        width = self.lge_volume.shape[2]

        if self.displayed_image is None:
            row_idx = height // 2
            col_idx = width // 2
        else:
            primary_height = self.displayed_image.shape[1]
            primary_width = self.displayed_image.shape[2]
            row_idx = self.__map_index_between_volumes(
                self.point_clicker_position[1], primary_height, height
            )
            col_idx = self.__map_index_between_volumes(
                self.point_clicker_position[2], primary_width, width
            )

        self.secondary_viewer.set_crosshair_from_volume_indices(row_idx, col_idx)

    # Primary plane slider callbacks
    def __on_axial_slider_changed(self, value):
        self.point_clicker_position[2] = value
        self.update_viewers_image()

    def __on_sagittal_slider_changed(self, value):
        self.point_clicker_position[0] = value
        self.update_viewers_image()

    def __on_coronal_slider_changed(self, value):
        self.point_clicker_position[1] = value
        self.update_viewers_image()

    def set_segmentation_volume(self, seg_volume):
        """Store a predicted segmentation volume and trigger overlay refresh."""

        self.segmentation_volume = seg_volume
        self.__update_segmentation_overlay()

    def set_lge_segmentation_volume(self, seg_volume):
        """Store an LGE segmentation stack for overlay on the LGE viewer."""
        if seg_volume is None or seg_volume.ndim != 3:
            self.lge_segmentation_volume = None
            self.secondary_viewer.update_overlay(None)
            return

        self.lge_segmentation_volume = seg_volume
        self.__update_lge_segmentation_overlay()

    def set_contour_edit_mode(self, enabled):
        self.contour_edit_mode = bool(enabled)
        if hasattr(self, "sagittal_viewer"):
            self.sagittal_viewer.set_edit_mode(self.contour_edit_mode)
            self.__update_segmentation_overlay()

    def __on_sagittal_contours_updated(self, contour_defs):
        if (
            not self.contour_edit_mode
            or self.segmentation_volume is None
            or contour_defs is None
        ):
            return
        try:
            depth = self.segmentation_volume.shape[0]
            x_idx = self.point_clicker_position[0]
            if x_idx < 0 or x_idx >= depth:
                return
            slice_idx = depth - x_idx - 1
            if slice_idx < 0 or slice_idx >= depth:
                return
            target_shape = self.segmentation_volume[slice_idx].shape
            updated_slice = self.__rasterize_contours(contour_defs, target_shape)
            if updated_slice is None:
                return
            self.segmentation_volume[slice_idx] = updated_slice
            self.sagittal_viewer.update_overlay(None)
        except Exception as exc:
            logging.error(
                "[CentralDisplayAreaWidget] Failed to persist contour edit: %s",
                exc,
            )

    def __update_segmentation_overlay(self):
        if self.segmentation_volume is None:
            self.sagittal_viewer.update_overlay(None)
            self.sagittal_viewer.clear_contours()
            return

        try:
            # Sagittal viewer uses an X-oriented slice: volume[x, :, :]
            x_idx = self.point_clicker_position[0]
            if x_idx < 0 or x_idx >= self.segmentation_volume.shape[0]:
                return
            seg_slice = self.segmentation_volume[-x_idx - 1, :, :]
            contours = self.__extract_segmentation_contours(seg_slice)
            self.sagittal_viewer.update_contour_paths(contours)
            self.sagittal_viewer.update_overlay(None)
        except Exception as e:
            logging.error(
                "[CentralDisplayAreaWidget] Failed to update segmentation overlay: %s",
                e,
            )

    def __update_lge_segmentation_overlay(self):
        if self.lge_segmentation_volume is None:
            self.secondary_viewer.update_overlay(None)
            return

        try:
            z = min(
                max(0, self.lge_slice_index), self.lge_segmentation_volume.shape[0] - 1
            )
            seg_slice = self.lge_segmentation_volume[z, :, :]
            overlay = self.__build_segmentation_overlay(seg_slice)
            self.secondary_viewer.update_overlay(overlay)
        except Exception as e:
            logging.error(
                "[CentralDisplayAreaWidget] Failed to update LGE segmentation overlay: %s",
                e,
            )
            self.secondary_viewer.update_overlay(None)

    @staticmethod
    def __build_segmentation_overlay(seg_slice):
        h, w = seg_slice.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        class_colors = {
            3: (0, 0, 255, 192),
            1: (255, 0, 0, 192),
        }
        for cls, color in class_colors.items():
            mask = seg_slice == cls
            if not mask.any():
                continue
            eroded = binary_erosion(mask)
            edges = mask & (~eroded)
            edges = binary_dilation(edges)
            overlay[edges] = color

        myo_mask = seg_slice == 2
        if myo_mask.any():
            lv_mask = seg_slice == 1
            lv_dilated = binary_dilation(lv_mask)
            myo_eroded = binary_erosion(myo_mask)
            myo_edges = myo_mask & (~myo_eroded)
            myo_outer_edges = myo_edges & (~lv_dilated)
            myo_outer_edges = binary_dilation(myo_outer_edges)
            overlay[myo_outer_edges] = (0, 255, 0, 192)

        return overlay

    @staticmethod
    def __extract_segmentation_contours(seg_slice):
        contours = []
        if seg_slice is None:
            return contours
        class_ids = (1, 2, 3)
        for class_id in class_ids:
            mask = (seg_slice == class_id).astype(np.uint8)
            if not mask.any():
                continue
            found, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in found:
                if len(contour) < 3:
                    continue
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(float(point[0][1]), float(point[0][0])) for point in approx]
                if len(points) >= 3:
                    contours.append({"class_id": class_id, "points": points})
        return contours

    def __rasterize_contours(self, contour_defs, target_shape):
        if (
            contour_defs is None
            or target_shape is None
            or self.segmentation_volume is None
        ):
            return None
        height, width = target_shape
        if height <= 0 or width <= 0:
            return None
        canvas = np.zeros(target_shape, dtype=self.segmentation_volume.dtype)
        class_order = (2, 1, 3)
        for class_id in class_order:
            polygons = []
            for contour in contour_defs:
                if contour.get("class_id") != class_id:
                    continue
                poly = self.__contour_to_polygon(
                    contour.get("points", []), height, width
                )
                if poly is not None:
                    polygons.append(poly)
            if polygons:
                cv2.fillPoly(canvas, polygons, int(class_id))
        return canvas

    @staticmethod
    def __contour_to_polygon(points, height, width):
        if not points or height <= 0 or width <= 0:
            return None
        converted = []
        for row, col in points:
            r = int(round(row))
            c = int(round(col))
            r = min(max(0, r), height - 1)
            c = min(max(0, c), width - 1)
            converted.append([c, r])
        if len(converted) < 3:
            return None
        return np.asarray(converted, dtype=np.int32).reshape((-1, 1, 2))

    def on_annotation_layer_toggled(self, volume_uid, state):
        if state:
            self.current_patient_parameters = (
                SoftwareConfigResources.getInstance().patients_parameters[
                    SoftwareConfigResources.getInstance().active_patient_name
                ]
            )
            self.overlaid_volumes[volume_uid] = (
                self.current_patient_parameters.get_annotation_by_uid(
                    volume_uid
                ).display_volume
            )
            # Only show contour on sagittal viewer (top left) using sagittal slice
            sagittal_slice = self.overlaid_volumes[volume_uid][
                self.point_clicker_position[0],
                :,
                :,
            ]
            self.sagittal_viewer.update_annotation_view(volume_uid, sagittal_slice)

            # Ensure other viewers do NOT retain stale overlays
            if volume_uid in self.axial_viewer.overlaid_items:
                self.axial_viewer.remove_annotation_view(volume_uid)
            if volume_uid in self.coronal_viewer.overlaid_items:
                self.coronal_viewer.remove_annotation_view(volume_uid)

        elif volume_uid in list(self.overlaid_volumes.keys()):
            self.overlaid_volumes.pop(
                volume_uid, None
            )  # None should not be necessary as the key should be in the dict
            self.sagittal_viewer.remove_annotation_view(volume_uid)
            # Also ensure purge from other viewers if any stale overlays exist
            if volume_uid in self.axial_viewer.overlaid_items:
                self.axial_viewer.remove_annotation_view(volume_uid)
            if volume_uid in self.coronal_viewer.overlaid_items:
                self.coronal_viewer.remove_annotation_view(volume_uid)

    def on_annotation_opacity_changed(self, volume_uid, value):
        self.sagittal_viewer.update_annotation_opacity(volume_uid, value)

    def on_annotation_color_changed(self, volume_uid, color):
        self.sagittal_viewer.update_annotation_color(volume_uid, color)

    def __on_axial_coordinates_changed(self, x, y):
        """
        When a new location is clicked in the axial plane, the same location is set in focus in the coronal and
        sagittal plane, for the main MRI volume and all overlaid annotations.
        """
        if self.displayed_image is not None:
            self.point_clicker_position[0] = min(
                max(0, y), self.displayed_image.shape[0] - 1
            )
            self.point_clicker_position[1] = min(
                max(0, x), self.displayed_image.shape[1] - 1
            )
            self.update_viewers_image()
        else:
            self.point_clicker_position[0] = 0
            self.point_clicker_position[1] = 0

    def __on_coronal_coordinates_changed(self, x, y):
        if self.displayed_image is not None:
            self.point_clicker_position[0] = min(
                max(0, y), self.displayed_image.shape[0] - 1
            )
            self.point_clicker_position[2] = min(
                max(0, x), self.displayed_image.shape[2] - 1
            )
            self.update_viewers_image()
        else:
            self.point_clicker_position[0] = 0
            self.point_clicker_position[2] = 0

    def __on_sagittal_coordinates_changed(self, x, y):
        if self.displayed_image is not None:
            self.point_clicker_position[1] = min(
                max(0, y), self.displayed_image.shape[1] - 1
            )
            self.point_clicker_position[2] = min(
                max(0, x), self.displayed_image.shape[2] - 1
            )
            self.update_viewers_image()
        else:
            self.point_clicker_position[1] = 0
            self.point_clicker_position[2] = 0

    def __on_lge_coordinates_changed(self, x, y):
        if self.lge_volume is None or self.displayed_image is None:
            return

        height = self.lge_volume.shape[1]
        width = self.lge_volume.shape[2]
        if height <= 0 or width <= 0:
            return

        row_idx = max(0, min(height - 1, int(y)))
        col_idx = max(0, min(width - 1, int(x)))

        primary_height = self.displayed_image.shape[1]
        primary_width = self.displayed_image.shape[2]
        mapped_y = self.__map_index_between_volumes(row_idx, height, primary_height)
        mapped_x = self.__map_index_between_volumes(col_idx, width, primary_width)

        updated = False
        if self.point_clicker_position[1] != mapped_y:
            self.point_clicker_position[1] = mapped_y
            updated = True
        if self.point_clicker_position[2] != mapped_x:
            self.point_clicker_position[2] = mapped_x
            updated = True

        if updated:
            self.update_viewers_image()

    def __synchronize_crosshairs(self):
        if self.displayed_image is None:
            return

        self.axial_viewer.set_crosshair_from_volume_indices(
            self.point_clicker_position[0],
            self.point_clicker_position[1],
        )
        self.coronal_viewer.set_crosshair_from_volume_indices(
            self.point_clicker_position[0],
            self.point_clicker_position[2],
        )
        self.sagittal_viewer.set_crosshair_from_volume_indices(
            self.point_clicker_position[1],
            self.point_clicker_position[2],
        )
        # Update headers slice labels
        self.__update_header_slice_labels()

    # Helpers for viewer containers
    def __make_viewer_header(self, title):
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color:#FFFFFF; font-weight:500; font-size:13px;")
        lbl_slice = QLabel("")  # dynamic slice info
        lbl_slice.setStyleSheet("color:#A0A0A0; font-size:11px;")
        layout.addWidget(lbl_title)
        layout.addStretch(1)
        layout.addWidget(lbl_slice)
        return layout

    def __make_slice_slider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setEnabled(False)
        slider.setMinimumHeight(20)
        slider.setStyleSheet(
            "QSlider {background: transparent;}"
            "QSlider::groove:horizontal {background:#2F3440; height:6px; border-radius:3px;}"
            "QSlider::handle:horizontal {background:#85aaff; width:14px; height:14px; margin:-4px 0; border-radius:7px;}"
            "QSlider::handle:horizontal:hover {background:#a0c0ff;}"
            "QSlider::handle:horizontal:disabled {background:#555555;}"
        )
        return slider

    def __make_plane_view(self, title, view_type, size, cb):
        container = QWidget()
        container.setStyleSheet("QWidget{background-color:#1E1E1E;}")
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(2, 2, 2, 2)
        vlayout.setSpacing(2)
        header_layout = self.__make_viewer_header(title)
        # Header layout last widget is slice label; keep reference
        slice_label = header_layout.itemAt(header_layout.count() - 1).widget()
        slider = self.__make_slice_slider()
        slider.valueChanged.connect(cb)
        viewer = CustomQGraphicsView(view_type=view_type, size=size, parent=self)
        vlayout.addLayout(header_layout)
        vlayout.addWidget(slider)
        vlayout.addWidget(viewer)
        return container, viewer, slider, {"slice_label": slice_label, "title": title}

    def __sync_primary_sliders(self):
        if self.displayed_image is None:
            return
        # Set ranges
        self.sagittal_slider.setMaximum(self.displayed_image.shape[0] - 1)
        self.axial_slider.setMaximum(self.displayed_image.shape[2] - 1)
        self.coronal_slider.setMaximum(self.displayed_image.shape[1] - 1)
        for sl in (self.sagittal_slider, self.axial_slider, self.coronal_slider):
            sl.setEnabled(sl.maximum() > 0)
        # Set positions
        self.sagittal_slider.setValue(self.point_clicker_position[0])
        self.coronal_slider.setValue(self.point_clicker_position[1])
        self.axial_slider.setValue(self.point_clicker_position[2])

    def __update_header_slice_labels(self):
        if self.displayed_image is None:
            return
        self.sagittal_header_labels["slice_label"].setText(
            f"Slice: {self.point_clicker_position[0]}/{self.displayed_image.shape[0]-1}"
        )
        self.coronal_header_labels["slice_label"].setText(
            f"Slice: {self.point_clicker_position[1]}/{self.displayed_image.shape[1]-1}"
        )
        self.axial_header_labels["slice_label"].setText(
            f"Slice: {self.point_clicker_position[2]}/{self.displayed_image.shape[2]-1}"
        )
