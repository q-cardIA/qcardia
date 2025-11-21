import logging
import os
from pathlib import Path

import numpy as np
import pydicom
import torch
import yaml
from qcardia.series import CineSeries
from qcardia_models.models import UNet2d
from scipy.ndimage import zoom


class DICOMModel:
    def __init__(self):
        self.dicom_files = []
        self.cine_seq = None
        self.folder_path = None
        self.temporal_volumes = []
        self.temporal_segmentations = []
        self.num_time_frames = 0

        # Load segmentation model
        self._load_segmentation_model()

    def _load_segmentation_model(self):
        """Load the UNet2d segmentation model."""
        wandb_path = Path.cwd() / "wandb"
        config_path = wandb_path / "files" / "config-copy.yaml"

        self.config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)
        self.wandb_run_path = wandb_path

        self.seg_model = UNet2d(
            nr_input_channels=self.config["unet"]["nr_image_channels"],
            channels_list=self.config["unet"]["channels_list"],
            nr_output_classes=self.config["unet"]["nr_output_classes"],
            nr_output_scales=self.config["unet"]["nr_output_scales"],
        ).to("cpu")

        model_weights = torch.load(wandb_path / "files" / "last_model.pt")
        self.seg_model.load_state_dict(model_weights)

    def load_dicom_files(self, folder_path):
        """Load DICOM files and create CineSeries."""
        self.folder_path = Path(folder_path)
        self.dicom_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    file_path = os.path.join(root, file)
                    try:
                        dicom_data = pydicom.read_file(file_path)
                        self.dicom_files.append(dicom_data)
                    except Exception as e:
                        print(f"Error reading DICOM file '{file_path}': {e}")

        # Create CineSeries for processing
        self.cine_seq = CineSeries(Path(folder_path), batch_size=200)

    def apply_window(self, image, center, width):
        """Apply window/level to image for display."""
        min_val = center - width / 2
        max_val = center + width / 2
        windowed = np.clip(image, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return windowed

    def process_temporal_volumes(self):
        """Load and process all temporal frames from the cine sequence."""
        if self.cine_seq is None:
            raise ValueError("No DICOM data loaded")

        # Get window settings from first slice
        center = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowCenter)
        width = float(self.cine_seq.slice_data["slice01"]["meta_data"][0].WindowWidth)

        # Get number of temporal frames
        first_key = sorted(self.cine_seq.slice_data.keys())[0]
        self.num_time_frames = len(self.cine_seq.slice_data[first_key]["pixel_array"])

        # Build temporal volumes
        self.temporal_volumes = []
        zoom_factors = [1.0, 1.0, 1.0]

        for time_idx in range(self.num_time_frames):
            slices = [
                self.cine_seq.slice_data[key]["pixel_array"][time_idx]
                for key in sorted(self.cine_seq.slice_data)
            ]
            image_stack = np.stack(slices, axis=0)
            stack_resampled = self.apply_window(
                zoom(image_stack, zoom_factors, order=1), center, width
            )
            self.temporal_volumes.append(stack_resampled)

        return self.temporal_volumes, self.num_time_frames

    def predict_segmentation(self):
        """Run segmentation prediction and organize results by time frame."""
        if self.cine_seq is None:
            raise ValueError("No DICOM data loaded")

        cine_segmentation = self.cine_seq.predict_segmentation(self.wandb_run_path)
        arr = np.asarray(cine_segmentation)
        dims = arr.shape

        # Identify axes
        nslices = len(sorted(self.cine_seq.slice_data.keys()))
        tframes = self.num_time_frames if self.num_time_frames else None

        # Find optional scale axis
        scale_axis = None
        try:
            n_scales = int(self.config["unet"]["nr_output_scales"])
            if n_scales in dims:
                scale_axis = dims.index(n_scales)
        except Exception:
            scale_axis = None

        # Helper to extract 3D volume for a time index
        def extract_ZHW_at_time(time_idx: int) -> np.ndarray:
            x = arr
            if scale_axis is not None and 0 <= scale_axis < x.ndim:
                sel = 1 if dims[scale_axis] > 1 else 0
                x = np.take(x, sel, axis=scale_axis)

            d = x.shape
            time_axis = None
            slice_axis = None

            if tframes is not None and tframes in d:
                time_axis = d.index(tframes)
            if nslices in d:
                candidates = [i for i, s in enumerate(d) if s == nslices]
                if len(candidates) == 1:
                    slice_axis = candidates[0]
                else:
                    slice_axis = next(
                        (i for i in candidates if i != time_axis), candidates[0]
                    )

            if time_axis is None and d[0] == tframes:
                time_axis = 0
            if slice_axis is None:
                if d[0] == nslices:
                    slice_axis = 0
                elif d[1] == nslices:
                    slice_axis = 1

            if time_axis is not None and 0 <= time_idx < d[time_axis]:
                x = np.take(x, time_idx, axis=time_axis)

            d = x.shape
            if slice_axis is not None and slice_axis < x.ndim:
                if slice_axis != 0:
                    axes = list(range(x.ndim))
                    axes[0], axes[slice_axis] = axes[slice_axis], axes[0]
                    x = np.transpose(x, axes)

            if x.ndim == 3:
                return x
            elif x.ndim == 2:
                return x[np.newaxis, :, :]
            else:
                return x.reshape(nslices, -1, x.shape[-1])

        # Extract per-time segmentations
        self.temporal_segmentations = []
        if tframes and tframes > 1:
            for t in range(tframes):
                try:
                    vol = extract_ZHW_at_time(t)
                    self.temporal_segmentations.append(vol)
                except Exception as e:
                    logging.warning(f"Failed to extract segmentation for time {t}: {e}")

        if not self.temporal_segmentations:
            try:
                seg_volume = extract_ZHW_at_time(0)
                if self.num_time_frames and self.num_time_frames > 1:
                    self.temporal_segmentations = [seg_volume] * self.num_time_frames
                else:
                    self.temporal_segmentations = [seg_volume]
            except Exception as e:
                logging.error(f"Failed to extract any segmentation: {e}")
                self.temporal_segmentations = []

        return self.temporal_segmentations

    def save_segmentation(self, output_path=None):
        """Save segmentation predictions to disk."""
        if self.cine_seq is None:
            raise ValueError("No DICOM data loaded")

        if not hasattr(self.cine_seq, "_segmentation_prediction"):
            raise ValueError("No segmentation to save")

        if output_path is None:
            output_path = self.folder_path.parent / "results" / self.folder_path.name

        self.cine_seq.save_predictions(Path(output_path))
        return output_path
