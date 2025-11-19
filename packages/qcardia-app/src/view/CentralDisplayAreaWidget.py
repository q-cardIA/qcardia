import logging
import traceback

import numpy as np
from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget
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

    def resizeEvent(self, event):
        # Default resize behavior retained.
        super().resizeEvent(event)

    def __set_interface(self):
        self.layout = QGridLayout(self)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.empty_label = QLabel()
        half_w = int(self.minimumWidth() / 2)
        half_h = int(self.minimumHeight() / 2)
        common_size = QSize(half_w, half_h)
        self.axial_viewer = CustomQGraphicsView(
            view_type="axial", size=common_size, parent=self
        )
        self.sagittal_viewer = CustomQGraphicsView(
            view_type="sagittal", size=common_size, parent=self
        )
        self.coronal_viewer = CustomQGraphicsView(
            view_type="coronal", size=common_size, parent=self
        )

        self.horizontal_line = QLabel()
        self.horizontal_line.setFixedHeight(3)
        self.vertical_line = QLabel()
        self.vertical_line.setFixedWidth(3)
        # Top-left should be sagittal so segmentation contour appears there
        self.layout.addWidget(self.sagittal_viewer, 0, 0)
        self.layout.addWidget(self.empty_label, 0, 2)
        self.layout.addWidget(self.horizontal_line, 1, 0, 1, 3)
        self.layout.addWidget(self.axial_viewer, 2, 0)
        self.layout.addWidget(self.coronal_viewer, 2, 2)
        self.layout.addWidget(self.vertical_line, 0, 1, 3, 1)

    # Removed unused __set_layout_dimensions (previously only comments).

    def __set_stylesheets(self):
        self.empty_label.setStyleSheet("QLabel{background-color:rgb(0, 0, 0);}")
        self.horizontal_line.setStyleSheet(
            """QLabel{background-color: rgba(214, 214, 214, 1);}"""
        )
        self.vertical_line.setStyleSheet(
            """QLabel{background-color: rgba(214, 214, 214, 1);}"""
        )

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
        # self.sagittal_viewer.patient_imported.connect(self.patient_imported)

    def reset_viewer(self):
        """ """
        self.reset_overlay()
        self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
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
                self.point_clicker_position = [
                    int(self.displayed_image.shape[0] / 2),
                    int(self.displayed_image.shape[1] / 2),
                    int(self.displayed_image.shape[2] / 2),
                ]
                self.displayed_image = self.displayed_image[:, ::-1, :]
                self.update_viewers_image()
            else:
                self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
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
            else:  # New patient selected: use first volume
                first_uid = self.current_patient_parameters.get_all_mri_volumes_uids()[
                    0
                ]
                self.displayed_image = self.current_patient_parameters.get_mri_by_uid(
                    first_uid
                ).display_volume
                self.displayed_image_uid = first_uid
            self.point_clicker_position = [
                int(self.displayed_image.shape[0] / 2),
                int(self.displayed_image.shape[1] / 2),
                int(self.displayed_image.shape[2] / 2),
            ]
        else:  # Empty patient fallback
            self.displayed_image = np.zeros(shape=(150, 150, 150), dtype="uint8")
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
        self.displayed_image_uid = volume_uid
        self.update_viewers_image()

    def update_viewers_image(self):
        """
        Further send the current display image, at the selected point clicker position, to each of the 2D viewers
        that are enabled in this widget (i.e., axial, coronal, and sagittal currently)
        """
        if self.displayed_image is None:
            return

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

    def set_segmentation_volume(self, seg_volume):
        """Store a predicted segmentation volume and trigger overlay refresh.

        seg_volume: np.ndarray with shape matching displayed_image orientation, class ids per voxel.
        """
        self.segmentation_volume = seg_volume
        self.__update_segmentation_overlay()

    def __update_segmentation_overlay(self):
        """Regenerate the segmentation overlay for the current slice index.

        Keeps behavior of showing segmentation in the top-left (sagittal) viewer.
        Uses class ids (e.g. 1,2,3) to build an RGBA overlay.
        """
        if self.segmentation_volume is None:
            return
        try:
            # Sagittal viewer uses an X-oriented slice: volume[x, :, :]
            x_idx = self.point_clicker_position[0]
            if x_idx < 0 or x_idx >= self.segmentation_volume.shape[0]:
                return
            seg_slice = self.segmentation_volume[x_idx, :, :]
            # Build RGBA overlay: initialize transparent canvas
            h, w = seg_slice.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            # Colors for contours (semi-transparent)
            class_colors = {
                3: (0, 0, 255, 192),  # RV - blue
                1: (255, 0, 0, 192),  # LV - red (endocardium)
            }
            # Generate 1-2 px thick contour for each class via morphological gradient
            for cls, color in class_colors.items():
                mask = seg_slice == cls
                if not mask.any():
                    continue
                eroded = binary_erosion(mask)
                edges = mask & (~eroded)
                # Optional: thicken contour slightly
                edges = binary_dilation(edges)
                overlay[edges] = color

            # Special handling for myocardium (class 2): only show epicardium (outer edge)
            # Exclude inner edge where it borders LV (class 1)
            myo_mask = seg_slice == 2
            if myo_mask.any():
                lv_mask = seg_slice == 1
                # Dilate LV to find myocardium's inner boundary
                lv_dilated = binary_dilation(lv_mask)
                # Erode myocardium to get its edges
                myo_eroded = binary_erosion(myo_mask)
                myo_edges = myo_mask & (~myo_eroded)
                # Remove edges that touch the LV (inner boundary)
                myo_outer_edges = myo_edges & (~lv_dilated)
                # Thicken contour slightly
                myo_outer_edges = binary_dilation(myo_outer_edges)
                overlay[myo_outer_edges] = (0, 255, 0, 192)  # Green
            # Send overlay to sagittal viewer (top-left)
            self.sagittal_viewer.update_overlay(overlay)
        except Exception as e:
            logging.error(
                "[CentralDisplayAreaWidget] Failed to update segmentation overlay: %s",
                e,
            )

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
