import logging

from PySide6.QtWidgets import QMessageBox

from view.utils import SoftwareConfigResources


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0

        # Connect signals
        self.view.load_button_signal.connect(self.load_dicom_files)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)

    def load_dicom_files(self, folder_path):
        """Load DICOM files through model and display."""
        try:
            self.model.load_dicom_files(folder_path)
            temporal_volumes, num_frames = self.model.process_temporal_volumes()

            # Display first frame
            self.current_time_index = 0
            self.view.update_display(temporal_volumes[self.current_time_index])

            # Update time slider
            if hasattr(self.view, "update_time_slider_range"):
                self.view.update_time_slider_range(num_frames)
        except Exception as e:
            logging.error(f"Failed to load DICOM files: {e}")
            self._show_error_dialog(
                "Load Error", f"Failed to load DICOM files:\n{str(e)}"
            )

    def run_segmentation(self):
        """Run segmentation prediction through model."""
        try:
            temporal_segmentations = self.model.predict_segmentation()

            # Display segmentation for current time frame
            if temporal_segmentations:
                idx = min(self.current_time_index, len(temporal_segmentations) - 1)
                self.view.display_area_widget.set_segmentation_volume(
                    temporal_segmentations[idx]
                )
        except Exception as e:
            logging.error(f"Segmentation failed: {e}")
            self._show_error_dialog(
                "Segmentation Error", f"Failed to run segmentation:\n{str(e)}"
            )

    def set_time_frame(self, time_index):
        """Update the displayed volume to the specified time frame."""
        temporal_volumes = self.model.temporal_volumes
        temporal_segmentations = self.model.temporal_segmentations

        if 0 <= time_index < len(temporal_volumes):
            self.current_time_index = time_index
            self.view.update_display_with_position(
                temporal_volumes[time_index], preserve_position=True
            )

            # Update segmentation overlay if available
            if temporal_segmentations and time_index < len(temporal_segmentations):
                self.view.display_area_widget.set_segmentation_volume(
                    temporal_segmentations[time_index]
                )

    def save_segmentation(self):
        """Save the segmentation results to default results folder."""
        try:
            output_path = self.model.save_segmentation()
            logging.info(f"Segmentation saved to {output_path}")
            self._show_info_dialog(
                "Save Successful",
                f"Segmentation saved successfully to:\n{output_path}",
            )
        except ValueError as e:
            self._show_warning_dialog("No Segmentation", str(e))
        except Exception as e:
            logging.error(f'Failed to save segmentation: {e}"')
            self._show_error_dialog(
                "Save Failed", f"Failed to save segmentation:\n{str(e)}"
            )

    def _get_dialog_style(self):
        """Get dark theme styling for dialogs."""
        colors = SoftwareConfigResources.getInstance().stylesheet_components
        return f"""
            QMessageBox {{
                background-color: {colors["Color2"]};
                color: {colors["Color7"]};
            }}
            QMessageBox QPushButton {{
                background-color: {colors["Color3"]};
                color: {colors["Color7"]};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {colors["Color4"]};
            }}
            QMessageBox QLabel {{
                color: {colors["Color7"]};
            }}
        """

    def _show_warning_dialog(self, title, message):
        """Show a warning dialog with dark theme."""
        msg = QMessageBox(self.view)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet(self._get_dialog_style())
        msg.exec()

    def _show_info_dialog(self, title, message):
        """Show an information dialog with dark theme."""
        msg = QMessageBox(self.view)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet(self._get_dialog_style())
        msg.exec()

    def _show_error_dialog(self, title, message):
        """Show an error dialog with dark theme."""
        msg = QMessageBox(self.view)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet(self._get_dialog_style())
        msg.exec()
