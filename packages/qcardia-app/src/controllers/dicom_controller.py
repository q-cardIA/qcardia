import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from qcardia.series import CineSeries
from qcardia_models.models import UNet2d

WANDB_RUN_PATH = Path.cwd() / "wandb"

config_path = WANDB_RUN_PATH / "files" / "config-copy.yaml"
config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

the_model = UNet2d(
    nr_input_channels=config["unet"]["nr_image_channels"],
    channels_list=config["unet"]["channels_list"],
    nr_output_classes=config["unet"]["nr_output_classes"],
    nr_output_scales=config["unet"]["nr_output_scales"],
).to("cpu")
model_weights = torch.load(WANDB_RUN_PATH / "files" / "last_model.pt")
the_model.load_state_dict(model_weights)


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.load_button_signal.connect(self.load_dicom_files)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)

        # Store all temporal frames
        self.temporal_volumes = []  # List of 3D volumes, one per time frame
        self.current_time_index = 0
        self.num_time_frames = 0
        # Optional temporal segmentations aligned with temporal_volumes
        self.temporal_segmentations = []  # List of 3D label maps per time frame

    def load_dicom_files(self, folder_path):
        self.folder_path = Path(folder_path)
        self.model.load_dicom_files(folder_path)
        self.cine_seq = CineSeries(Path(folder_path), batch_size=200)

        # dicom_files = self.model.dicom_files
        self.display_dicom_images()

    def apply_window(self, image, center, width):
        min_val = center - width / 2
        max_val = center + width / 2
        windowed = np.clip(image, min_val, max_val)
        # Normalize to 0-255 for display
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return windowed

    def display_dicom_images(self):
        # Load all temporal frames for the cine sequence
        from scipy.ndimage import zoom

        # Get window settings from first slice
        center = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowCenter)
        width = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowWidth)

        # Get number of temporal frames from first slice
        first_key = sorted(self.cine_seq.slice_data.keys())[0]
        self.num_time_frames = len(self.cine_seq.slice_data[first_key]["pixel_array"])

        # Build a list of 3D volumes, one for each time frame
        self.temporal_volumes = []
        zoom_factors = [1.0, 1.0, 1.0]

        for time_idx in range(self.num_time_frames):
            # Build 3D stack for this time frame
            slices = [
                self.cine_seq.slice_data[key]["pixel_array"][time_idx]
                for key in sorted(self.cine_seq.slice_data)
            ]
            image_stack = np.stack(slices, axis=0)  # shape: (num_slices, height, width)
            stack_resampled = self.apply_window(
                zoom(image_stack, zoom_factors, order=1), center, width
            )
            self.temporal_volumes.append(stack_resampled)

        # Display the first time frame initially
        self.current_time_index = 0
        self.view.update_display(self.temporal_volumes[self.current_time_index])

        # Update the time slider range in the view
        if hasattr(self.view, "update_time_slider_range"):
            self.view.update_time_slider_range(self.num_time_frames)

    def run_segmentation(self):
        """Predict segmentation for the cine sequence and prepare per-time overlays.

        Attempts to infer axes order from the result and align with the
        current temporal dimension (num_time_frames) and slice count.
        """
        cine_segmentation = self.cine_seq.predict_segmentation(WANDB_RUN_PATH)

        arr = np.asarray(cine_segmentation)
        dims = arr.shape

        # Identify axes
        nslices = len(sorted(self.cine_seq.slice_data.keys()))
        tframes = self.num_time_frames if self.num_time_frames else None

        # Find optional scale axis (match configured nr_output_scales if possible)
        scale_axis = None
        try:
            n_scales = int(config["unet"]["nr_output_scales"])  # from loaded config
            if n_scales in dims:
                scale_axis = dims.index(n_scales)
        except Exception:
            scale_axis = None

        # Helper to get a 3D (Z,H,W) label volume for a given time index
        def extract_ZHW_at_time(time_idx: int) -> np.ndarray:
            x = arr
            # If scale axis exists, pick scale=1 if available else 0
            if scale_axis is not None and 0 <= scale_axis < x.ndim:
                sel = 1 if dims[scale_axis] > 1 else 0
                x = np.take(x, sel, axis=scale_axis)

            # After removing scale, recompute dims
            d = x.shape

            # Determine time and slice axes by matching sizes
            time_axis = None
            slice_axis = None
            if tframes is not None and tframes in d:
                time_axis = d.index(tframes)
            if nslices in d:
                # If nslices equals tframes and axes clash, prefer a different index
                candidates = [i for i, s in enumerate(d) if s == nslices]
                if len(candidates) == 1:
                    slice_axis = candidates[0]
                else:
                    # Heuristic: choose the first candidate not equal to time_axis
                    slice_axis = next(
                        (i for i in candidates if i != time_axis), candidates[0]
                    )

            # If we couldn't infer, try simple common layouts
            if time_axis is None and d[0] == tframes:
                time_axis = 0
            if slice_axis is None:
                if d[0] == nslices:
                    slice_axis = 0
                elif d[1] == nslices:
                    slice_axis = 1
                else:
                    # Last resort: assume first axis is slices
                    slice_axis = 0

            # Select this time slice if time axis exists
            if time_axis is not None and 0 <= time_axis < x.ndim:
                x = np.take(x, time_idx, axis=time_axis)

            # Now x should be 3D with one axis == nslices
            # Reorder to (Z, H, W)
            d2 = x.shape
            # Find which axis is Z (slices)
            try:
                z_axis = [i for i, s in enumerate(d2) if s == nslices][0]
            except Exception:
                # Fallback: assume axis 0 is Z
                z_axis = 0
            axes = list(range(x.ndim))
            # Move z_axis to front
            axes.remove(z_axis)
            new_order = [z_axis] + axes
            zhw = np.transpose(x, new_order)

            # Flip horizontally to match image orientation used elsewhere
            zhw = zhw[:, :, ::-1]
            return zhw.astype(np.uint8, copy=False)

        # Build temporal list
        if tframes is None:
            # If time frames not known yet, assume first axis is time
            tframes = dims[0]

        self.temporal_segmentations = []
        for t in range(int(tframes)):
            try:
                vol = extract_ZHW_at_time(t)
                self.temporal_segmentations.append(vol)
            except Exception:
                # If extraction fails, break and fallback to single volume
                break

        if not self.temporal_segmentations:
            # Fallback: try to squeeze to 3D and use as a static overlay
            try:
                seg_volume = np.squeeze(arr)
                if seg_volume.ndim == 3:
                    seg_volume = seg_volume[:, :, ::-1]
                    self.temporal_segmentations = [seg_volume]
            except Exception:
                self.temporal_segmentations = []

        # Apply overlay for the current time frame if available
        if self.temporal_segmentations:
            idx = min(self.current_time_index, len(self.temporal_segmentations) - 1)
            self.view.display_area_widget.set_segmentation_volume(
                self.temporal_segmentations[idx]
            )

    # Legacy per-slice display removed; segmentation overlays update
    def set_time_frame(self, time_index):
        """Update the displayed volume to the specified time frame."""
        if 0 <= time_index < self.num_time_frames and self.temporal_volumes:
            self.current_time_index = time_index
            # Preserve current crosshair position while switching time frames
            vol = self.temporal_volumes[time_index]
            da = self.view.display_area_widget
            # Current [x, y, z] indices
            pos = getattr(da, "point_clicker_position", [0, 0, 0])
            # Clamp to new volume bounds (Z, H, W) with indices in [x, y, z]
            x = min(max(0, pos[0]), vol.shape[0] - 1)
            y = min(max(0, pos[1]), vol.shape[1] - 1)
            z = min(max(0, pos[2]), vol.shape[2] - 1)
            # Swap the displayed volume and refresh views without recentering
            da.displayed_image = vol
            da.point_clicker_position = [x, y, z]
            da.update_viewers_image()
            # If we have per-time segmentations, update overlay as well
            if self.temporal_segmentations:
                idx = min(time_index, len(self.temporal_segmentations) - 1)
                self.view.display_area_widget.set_segmentation_volume(
                    self.temporal_segmentations[idx]
                )

    def save_segmentation(self):
        """Save the segmentation results to default results folder."""
        from PySide6.QtWidgets import QMessageBox

        from view.utils import SoftwareConfigResources

        # Get dark theme colors
        colors = SoftwareConfigResources.getInstance().stylesheet_components
        dialog_style = f"""
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

        if not hasattr(self.cine_seq, "_segmentation_prediction"):
            msg = QMessageBox(self.view)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("No Segmentation")
            msg.setText("No segmentation to save. Please run segmentation first.")
            msg.setStyleSheet(dialog_style)
            msg.exec()
            return False

        # Default to results folder next to the DICOM folder
        output_path = self.folder_path.parent / "results" / self.folder_path.name

        try:
            self.cine_seq.save_predictions(Path(output_path))
            logging.info(f"Segmentation saved to {output_path}")
            msg = QMessageBox(self.view)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Save Successful")
            msg.setText(f"Segmentation saved successfully to:\n{output_path}")
            msg.setStyleSheet(dialog_style)
            msg.exec()
            return True
        except Exception as e:
            logging.error(f"Failed to save segmentation: {e}")
            msg = QMessageBox(self.view)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Save Failed")
            msg.setText(f"Failed to save segmentation:\n{str(e)}")
            msg.setStyleSheet(dialog_style)
            msg.exec()
            return False
