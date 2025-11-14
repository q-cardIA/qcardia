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

    def load_dicom_files(self, folder_path):
        self.model.load_dicom_files(folder_path)
        self.cine_seq = CineSeries(Path(folder_path))

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
        # Assuming self.cine_seq.slice_data is a dict of slices
        # Build a 3D stack from your slices
        slices = [
            self.cine_seq.slice_data[key]["pixel_array"][1]
            for key in sorted(self.cine_seq.slice_data)
        ]

        center = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowCenter)
        width = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowWidth)

        from scipy.ndimage import zoom

        image_stack = np.stack(slices, axis=0)  # shape: (num_slices, height, width)

        zoom_factors = [
            # self.cine_seq.slice_data["slice01"]["meta_data"][0].SliceThickness,
            1.0,
            1.0,
            1.0,
        ]  # example values
        stack_resampled = self.apply_window(
            zoom(image_stack, zoom_factors, order=1), center, width
        )
        self.view.update_display(stack_resampled)

    # def display_dicom_images(self):

    #     dicom_data = self.cine_seq.slice_data["slice06"]["pixel_array"][1]

    #     int_min = 0
    #     int_max = 1000
    #     # int_min = dicom_data.WindowCenter - 0.5 * dicom_data.WindowWidth
    #     # int_max = dicom_data.WindowCenter + 0.5 * dicom_data.WindowWidth
    #     normalised_array = np.clip(dicom_data, int_min, int_max)
    #     normalised_array = (
    #         (normalised_array - int_min) / (int_max - int_min) * 255
    #     ).astype(np.uint8)

    #     dicom_scene = QGraphicsScene()
    #     pixmap = QPixmap.fromImage(
    #         QImage(
    #             normalised_array,
    #             normalised_array.shape[0],
    #             normalised_array.shape[0],
    #             QImage.Format_Grayscale8,
    #         )
    #     )
    #     self.normalised_array = normalised_array
    #     self.view.update_display(normalised_array)
    #     # dicom_scene.addPixmap(pixmap)
    #     # self.view.ui.graphicsView.setScene(dicom_scene)

    def run_segmentation(self):
        cine_segmentation = self.cine_seq.predict_segmentation(WANDB_RUN_PATH)
        # Optional: volume curve currently unused; compute only if needed.
        # Store segmentation volume (choose scale/index 1 as before) and
        # trigger dynamic overlays
        try:
            # Assuming cine_segmentation shape: (slices, scales, H, W)
            # with class ids in selected scale
            seg_volume = cine_segmentation[:, 1, :, ::-1]
        except Exception:
            # Fallback: if shape unexpected, attempt to squeeze
            seg_volume = np.squeeze(cine_segmentation)
        self.view.display_area_widget.set_segmentation_volume(seg_volume)

    # Legacy per-slice display removed; segmentation overlays update
    # dynamically as the slice index changes in the view.
    def display_segmentation(self, cine_segmentation):
        """Deprecated: dynamic contour overlay handled by CentralDisplayAreaWidget."""
        return
