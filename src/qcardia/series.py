"""
This module contains the BaseSeries class which is used for handling series of
DICOM images.

The BaseSeries class provides methods for loading DICOM data from a folder,
preprocessing the data, running a model on the data, and postprocessing the model
output. The model is specified by a path to a Weights & Biases run.

Classes:
    BaseSeries: A base class for handling sequences of DICOM images.
"""

from copy import deepcopy
from pathlib import Path
from typing import List

import cv2
import nibabel as nib
import numpy as np
import pydicom
import torch
import yaml
from natsort import natsorted
from qcardia_models.models import UNet2d
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import find_contours
from torch.nn import functional as F

import qcardia.utils as utils


class BaseSeries:
    """
    A base class for handling series of DICOM images.

    This class provides methods for loading DICOM data from a folder, preprocessing the data,
    running a model on the data, and postprocessing the model output. The model is specified
    by a path to a Weights & Biases run.

    Attributes:
        folder (Path): The folder where the DICOM files are located.
        batch_size (int): The batch size to use when running the model.
        slice_data (dict): A dictionary where the keys are the slice numbers and the values
            are another dictionary containing the pixel array, slice position, and
            meta data for each slice.
        number_of_slices (int): The number of slices in the DICOM data.
        inference_dict (dict): A dictionary used to store various parameters and data needed
            for inference.
    """

    def __init__(self, folder: Path, batch_size: int = 50):
        self.folder = folder
        (
            self.slice_data,
            self.number_of_slices,
            self.number_of_temporal_positions,
            self.rows,
            self.columns,
        ) = self._load_data()
        self.inference_dict = {}
        self.batch_size = batch_size
        self.base_slice_num = 1
        self.mid_slice_num = 2
        self.apex_slice_num = 3
        self.rv_insertion_points = [[0, 0], [self.rows, 0]]
        self.lv_center_point = [[self.rows // 2, self.columns // 2]]

    def predict_segmentation(self, wandb_run_path: Path) -> np.ndarray:
        """
        Predict the segmentation for the DICOM data using the specified model.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.

        Returns:
            np.ndarray: The predicted segmentation for the DICOM data.
        """
        self._run_model(wandb_run_path)
        return self._segmentation_prediction

    def save_predictions(self, output_path: Path) -> None:
        """
        Saves the segmentation predictions to the specified path.

        Args:
            output_path (Path): The path to save the segmentation predictions.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        seg_prediction = np.transpose(self._segmentation_prediction).astype(np.uint8)[
            ..., ::-1
        ]
        if len(seg_prediction.shape) == 4:
            for i in range(seg_prediction.shape[-2]):
                seg_nib = nib.Nifti1Image(seg_prediction[..., i, :], np.eye(4))
                nib.save(seg_nib, output_path / f"segmentation_{i+1}.nii")
        else:
            seg_nib = nib.Nifti1Image(seg_prediction, np.eye(4))
            nib.save(seg_nib, output_path / "segmentation.nii")

    def _run_model(self, wandb_run_path: Path, image_type: str = "pixel") -> None:
        """
        Runs the model inference on preprocessed slices.
        
        Now supports both simple UNet models and context-aware Transformer models.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.
            image_type (str): Type of image array to use ('pixel' or other)
        """
        # Import helper modules
        try:
            from qcardia.inference import InferenceConfig, load_model_from_config, InferencePredictor
            USE_NEW_INFERENCE = True
        except ImportError:
            USE_NEW_INFERENCE = False
            print("Warning: New inference modules not available. Using legacy inference.")
        
        # Preprocessing (unchanged)
        preprocessed_slices = self._preproccess_slices(
            self._get_array(image_type=image_type)
        )
        
        # Load raw config
        raw_config = self._get_config(wandb_run_path)
        
        if USE_NEW_INFERENCE:
            # NEW: Use helper modules for context-aware inference
            try:
                # Parse config with new handler
                config = InferenceConfig(raw_config)
                
                # Store inference parameters
                self.inference_dict["target_pixdim"] = torch.tensor(config.target_pixdim)
                self.inference_dict["target_size"] = torch.tensor(config.target_size)
                self.inference_dict["grid_sample_modes"] = [config.image_grid_sample_mode]
                self.inference_dict["nr_output_classes"] = config.nr_classes
                
                # Load model dynamically (supports both UNet and Transformer)
                # Auto-detect device (prefer GPU if available)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"  Using device: {device}")
                the_model = load_model_from_config(raw_config, wandb_run_path, device=device)
                
                # Rescale and standardize (unchanged)
                self.inference_dict["dimension_scale_factor"], rescaled_tensor = (
                    self._rescale_tensor(preprocessed_slices)
                )
                standardised_tensor = utils.standardise(rescaled_tensor)
                
                # NEW: Reshape for context-aware models if needed
                if config.needs_context():
                    # Context-aware models expect (B, C, H, W, Z, T) format
                    # Current shape is (Z*T, C, H, W), need to reshape
                    n_slices = self.number_of_slices
                    n_frames = self.number_of_temporal_positions
                    _, C, H, W = standardised_tensor.shape
                    
                    # Reshape: (Z*T, C, H, W) -> (1, C, H, W, Z, T)
                    standardised_tensor = standardised_tensor.view(
                        n_slices, n_frames, C, H, W
                    ).permute(2, 3, 4, 0, 1).unsqueeze(0)
                    print(f"    Reshaped for context-aware: {standardised_tensor.shape}")
                
                # NEW: Use predictor for inference (handles context automatically)
                predictor = InferencePredictor(the_model, config, device=device, batch_size=self.batch_size)
                model_output = predictor.predict(standardised_tensor)
                
                # Reshape output back for downstream processing
                if config.needs_context() and len(model_output.shape) == 6:
                    # Output shape: (B, num_classes, H, W, Z, T)
                    # Reshape to: (Z*T, num_classes, H, W)
                    print(f"    Pre-reshape output shape: {model_output.shape}")
                    B, num_classes, H, W, Z, T = model_output.shape
                    # First remove batch dimension: (num_classes, H, W, Z, T)
                    output_5d = model_output[0]
                    print(f"    After removing batch: {output_5d.shape}")
                    # Permute to: (Z, T, num_classes, H, W)
                    output_permuted = output_5d.permute(3, 4, 0, 1, 2)
                    print(f"    After permute: {output_permuted.shape}")
                    # Reshape to: (Z*T, num_classes, H, W)
                    model_output = output_permuted.reshape(Z*T, num_classes, H, W)
                    print(f"    Final reshaped output: {model_output.shape}")
                
            except Exception as e:
                print(f"Warning: New inference failed ({e}), falling back to legacy")
                USE_NEW_INFERENCE = False
        
        if not USE_NEW_INFERENCE:
            # LEGACY: Original hardcoded UNet inference
            config = raw_config
            self.inference_dict["target_pixdim"] = torch.tensor(
                config["data"]["target_pixdim"]
            )
            self.inference_dict["target_size"] = torch.tensor(config["data"]["target_size"])
            self.inference_dict["grid_sample_modes"] = [
                config["data"]["image_grid_sample_mode"]
            ]
            self.inference_dict["nr_output_classes"] = config["unet"]["nr_output_classes"]
            
            the_model = UNet2d(
                nr_input_channels=config["unet"]["nr_image_channels"],
                channels_list=config["unet"]["channels_list"],
                nr_output_classes=config["unet"]["nr_output_classes"],
                nr_output_scales=config["unet"]["nr_output_scales"],
            ).to("cpu")
            
            # Try flat layout first, then WandB files/ subdirectory
            weights_path = wandb_run_path / "last_model.pt"
            if not weights_path.exists():
                weights_path = wandb_run_path / "best_model.pt"
            if not weights_path.exists():
                weights_path = wandb_run_path / "files" / "last_model.pt"
            if not weights_path.exists():
                weights_path = wandb_run_path / "files" / "best_model.pt"
            
            model_weights = torch.load(weights_path)
            the_model.load_state_dict(model_weights)

            self.inference_dict["dimension_scale_factor"], rescaled_tensor = (
                self._rescale_tensor(preprocessed_slices)
            )
            standardised_tensor = utils.standardise(rescaled_tensor)
            model_output = self._forward_model(the_model, standardised_tensor)

        rescale_model_output = self._invert_rescale_tensor(model_output)
        model_prediction = torch.argmax(
            rescale_model_output, dim=1, keepdim=True
        ).float()

        self._segmentation_prediction = self._postprocess_output(model_prediction)

    def _load_data(self):
        """
        Load DICOM data from a folder and extract relevant information.

        Returns:
            slices_dict (dict): A dictionary containing information for each slice.
                Each key represents a slice number, and the corresponding value is a dictionary
                with the following keys:
                    - "pixel_array": A list of pixel arrays for each image in the slice.
                    - "slice_position": The position of the slice.
                    - "meta_data": A list of meta data objects for each image in the slice.
            number_of_slices (int): The total number of slices.

        """

        # Use natsorted to sort the files in the folder in natural order
        files = natsorted(
            [
                # Loop through each file in the folder
                f
                for f in self.folder.iterdir()
                if (
                    # Check if the path is a file
                    f.is_file()
                    # Ignore hidden files that start with "."
                    and not f.stem.startswith(".")
                    # Ignore files with "dicomdir" in their name
                    and "dicomdir" not in str(f).lower()
                )
            ]
        )

        all_dicom_data = []
        slice_position = []
        slice_orientation = []
        temporal_positions = []

        # Read DICOM files and extract relevant information
        for file_idx, file in enumerate(files):
            the_ds = pydicom.dcmread(file)

            # Skip files that contain no image data (e.g. metadata-only DICOM files)
            if 'PixelData' not in the_ds:
                continue

            all_dicom_data.append(the_ds)

            # Spatial position/orientation — fall back for non-standard DICOM
            if hasattr(the_ds, 'ImagePositionPatient') and hasattr(the_ds, 'ImageOrientationPatient'):
                slice_position.append(the_ds.ImagePositionPatient)
                slice_orientation.append(the_ds.ImageOrientationPatient)
            elif hasattr(the_ds, 'SliceLocation'):
                # Construct a position along the z-axis using the scalar SliceLocation tag
                slice_position.append([0.0, 0.0, float(the_ds.SliceLocation)])
                slice_orientation.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            else:
                # Last resort: use file order as a proxy for slice position
                slice_position.append([0.0, 0.0, float(file_idx)])
                slice_orientation.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

            # Temporal position — fall back to InstanceNumber
            n_temporal = int(getattr(the_ds, 'NumberOfTemporalPositions', 0))
            if n_temporal == 1:
                temporal_positions.append(int(the_ds.InstanceNumber))
            elif hasattr(the_ds, 'TemporalPositionIdentifier'):
                temporal_positions.append(int(the_ds.TemporalPositionIdentifier))
            else:
                temporal_positions.append(int(the_ds.InstanceNumber))

        # Filter to the dominant image size — DICOM folders sometimes contain
        # scout/localizer images or mixed acquisitions with different H×W.
        if all_dicom_data:
            from collections import Counter
            all_shapes = [(ds.Rows, ds.Columns) for ds in all_dicom_data]
            dominant_shape = Counter(all_shapes).most_common(1)[0][0]
            if len(set(all_shapes)) > 1:
                print(f"    Warning: mixed image sizes found; keeping {dominant_shape[0]}x{dominant_shape[1]} only "
                      f"(skipping {sum(1 for s in all_shapes if s != dominant_shape)} file(s)).")
                keep = [s == dominant_shape for s in all_shapes]
                all_dicom_data   = [d for d, k in zip(all_dicom_data,   keep) if k]
                slice_position   = [p for p, k in zip(slice_position,   keep) if k]
                slice_orientation = [o for o, k in zip(slice_orientation, keep) if k]
                temporal_positions = [t for t, k in zip(temporal_positions, keep) if k]

        # Gets unique positions from given positions and orientations.
        # assigns a slice index to each image based on its position.
        indices_of_slices, the_slice_positions = self._get_slices_from_positions(
            slice_position, slice_orientation
        )

        slices_dict = {}
        number_of_slices = len(the_slice_positions)
        number_of_temporal_positions = len(temporal_positions) // number_of_slices
        # for each slice store the pixel array, slice position, and meta data
        # order based on temporal position
        for i in range(number_of_slices):
            image_array = []
            list_of_meta_data = []
            slice_tmp_position = []
            for j, the_ds in enumerate(all_dicom_data):
                if indices_of_slices[j] == i:
                    image_array.append(the_ds.pixel_array)
                    list_of_meta_data.append(the_ds)
                    slice_tmp_position.append(temporal_positions[j])
            sorted_list_of_meta_data = [
                x
                for _, x in sorted(
                    zip(slice_tmp_position, list_of_meta_data), key=lambda pair: pair[0]
                )
            ]
            sorted_image_array = [
                x
                for _, x in sorted(
                    zip(slice_tmp_position, image_array), key=lambda pair: pair[0]
                )
            ]

            # store the list of meta data for each slice in a dict with the slice number as
            # the key, along with the pixel array and slice position
            slices_dict[f"slice{i+1:02}"] = {
                "pixel_array": sorted_image_array,
                "slice_position": the_slice_positions[i],
                "meta_data": sorted_list_of_meta_data,
            }

        return (
            slices_dict,
            number_of_slices,
            number_of_temporal_positions,
            all_dicom_data[0].Rows,
            all_dicom_data[0].Columns,
        )

    def _get_slices_from_positions(
        self, positions: List[List[float]], orientations: List[List[float]]
    ):
        """
        Calculate slice index and unique positions based on given positions and orientations.

        Args:
            positions (list): List of positions.
            orientations (list): List of orientations.

        Returns:
            tuple: A tuple containing slice index and unique positions (both ndarray).
        """
        true_positions = []
        for i in range(len(positions)):
            true_positions.append(
                np.dot(
                    positions[i], np.cross(orientations[i][0:3], orientations[i][3:6])
                )
            )

        # find unique positions, these are the slices
        unique_positions = np.unique(true_positions)
        unique_positions = np.sort(unique_positions)[::-1]  # sort in descending order
        slice_index = np.zeros(len(true_positions))
        # give a slice index to each image based on its position
        for i in range(len(unique_positions)):
            slice_index[np.where(true_positions == unique_positions[i])] = i

        return slice_index, unique_positions

    def _preproccess_slices(self, pixel_array, motion_track=False):
        """
        Preprocesses the slices by performing various operations such as reshaping,
        border stripping, normalization, and conversion to tensors.

        Returns:
            torch.Tensor: The preprocessed pixel array without borders.
        """
        self.inference_dict["original_shape"] = torch.tensor(pixel_array.shape)
        if not motion_track:
            reshaped_pixel_array = self._reshape_array(pixel_array)
        else:
            reshaped_pixel_array = deepcopy(pixel_array)

        start0, stop0, start1, stop1, borderless_pixel_array = self._strip_borders(
            reshaped_pixel_array
        )
        self.inference_dict["border_indices"] = [
            start0,
            stop0,
            start1,
            stop1,
        ]
        borderless_pixel_array -= np.amin(borderless_pixel_array)
        borderless_pixel_tensor = torch.tensor(
            borderless_pixel_array.astype(np.float32)
        )
        self.inference_dict["source_shape"] = torch.tensor(
            borderless_pixel_array.shape[-2:], dtype=torch.float32
        )
        self.inference_dict["number_of_slices"] = reshaped_pixel_array.shape[0]
        self.inference_dict["pixdims"] = self._get_pixel_spacing()

        return borderless_pixel_tensor

    def _get_array(self, image_type="pixel"):
        """
        Get the pixel array for all slice/times.

        Shape depends on the number of slices/frames and the size of the pixel arrays.

        Returns:
            ndarray: The pixel array for all slices/times.
        """

        from collections import Counter
        arrays = [
            self.slice_data[f"slice{i+1:02}"][f"{image_type}_array"]
            for i in range(self.number_of_slices)
        ]
        # Trim to the most common temporal length if slices have inconsistent frame counts
        frame_counts = [len(a) for a in arrays]
        if len(set(frame_counts)) > 1:
            dominant_n = Counter(frame_counts).most_common(1)[0][0]
            print(f"    Warning: inconsistent frame counts across slices {set(frame_counts)}; "
                  f"trimming all slices to {dominant_n} frames.")
            arrays = [a[:dominant_n] for a in arrays if len(a) >= dominant_n]
        return np.asarray(arrays)

    def _reshape_array(self, pa: np.ndarray):
        """
        Reshape the pixel array to have all the times/slices in the
        batch dimension. With one channel.

        The pixel array is reshaped to have the following dimensions:
        (number_of_slices, 1, height, width).

        Returns:
            ndarray: The reshaped pixel array.
        """

        return pa.reshape(
            -1,
            1,
            pa.shape[-2],
            pa.shape[-1],
        )

    def _strip_borders(self, reshape_pa: np.ndarray):
        """
        Strip the borders from the pixel array.

        Returns:
            ndarray: The borderless pixel array.
        """
        summed_pixel_array = np.sum(reshape_pa, axis=(0, 1))
        borderless_idxs_0 = np.nonzero(np.any(summed_pixel_array, axis=1))[0]
        borderless_idxs_1 = np.nonzero(np.any(summed_pixel_array, axis=0))[0]

        start_idx_0, stop_idx_0 = borderless_idxs_0[0], borderless_idxs_0[-1] + 1
        start_idx_1, stop_idx_1 = borderless_idxs_1[0], borderless_idxs_1[-1] + 1

        return (
            start_idx_0,
            stop_idx_0,
            start_idx_1,
            stop_idx_1,
            reshape_pa[..., start_idx_0:stop_idx_0, start_idx_1:stop_idx_1],
        )

    def _get_config(self, wandb_run_path: Path):
        """
        Get the config file from the specified WandB run path.

        Args:
            wandb_run_path (Path): The path to the WandB run.

        Returns:
            dict: The configuration loaded from the specified path.
        """

        # Support both flat layout (config-copy.yaml next to weights) and
        # WandB run layout (files/config-copy.yaml).
        config_path = wandb_run_path / "config-copy.yaml"
        if not config_path.exists():
            config_path = wandb_run_path / "files" / "config-copy.yaml"
        return yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

    def _get_pixel_spacing(self):
        """
        Get the pixel spacing of the slice data.

        Returns:
            torch.Tensor: A tensor containing the pixel spacing values.
        """
        return torch.tensor(
            [
                float(self.slice_data["slice01"]["meta_data"][0].PixelSpacing[0]),
                float(self.slice_data["slice01"]["meta_data"][0].PixelSpacing[1]),
                float(self.slice_data["slice01"]["meta_data"][0].SliceThickness),
            ],
            dtype=torch.float32,
        )

    def _rescale_tensor(self, pixel_tensor: torch.Tensor):
        """
        Rescales the input pixel tensor based on the inference dictionary.

        Args:
            pixel_tensor (torch.Tensor): The input pixel tensor.

        Returns:
            Tuple[float, torch.Tensor]: A tuple containing the dimension scale factor and the rescaled pixel tensor.
        """
        real_source_size = (
            self.inference_dict["pixdims"][:2] * self.inference_dict["source_shape"]
        )
        real_target_size = (
            self.inference_dict["target_pixdim"] * self.inference_dict["target_size"]
        )

        dimension_scale_factor = real_target_size / real_source_size
        scale_t = utils.t_2d_scale(dimension_scale_factor)

        grid_size = [
            self.inference_dict["number_of_slices"],
            1,
            int(self.inference_dict["target_size"][0]),
            int(self.inference_dict["target_size"][1]),
        ]
        grid = F.affine_grid(
            theta=torch.repeat_interleave(
                scale_t[:-1, :].unsqueeze(0),
                self.inference_dict["number_of_slices"],
                dim=0,
            ),
            size=grid_size,
            align_corners=False,
        )
        return dimension_scale_factor, F.grid_sample(
            pixel_tensor,
            grid,
            align_corners=False,
            mode=self.inference_dict["grid_sample_modes"][0],
            padding_mode="zeros",
        )

    def _forward_model(self, model: torch.nn.Module, tensor: torch.Tensor):
        """
        Forward the pixel array through the model.

        Args:
            model (torch.nn.Module): The model to use for inference.
            tensor (torch.Tensor): The pixel array to forward through the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        from tqdm import tqdm
        
        model_output = torch.zeros(
            self.inference_dict["number_of_slices"],
            self.inference_dict["nr_output_classes"],
            tensor.shape[-2],
            tensor.shape[-1],
        )

        model.eval()
        with torch.no_grad():
            total_batches = self.inference_dict["number_of_slices"] // self.batch_size + 1
            pbar = tqdm(range(0, self.inference_dict["number_of_slices"] // self.batch_size + 1),
                       desc="Inference (legacy)",
                       total=total_batches,
                       unit="batch")
            
            for i in pbar:
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                if start_idx < self.inference_dict["number_of_slices"]:
                    model_output[start_idx:end_idx] = model(
                        tensor[start_idx:end_idx]
                    )[0]
                    pbar.set_postfix({'images': f'{min(end_idx, self.inference_dict["number_of_slices"])}/{self.inference_dict["number_of_slices"]}'})

        return model_output

    def _invert_rescale_tensor(self, model_output: torch.Tensor, mode: str = "bicubic"):
        """
        Inverts the rescaling operation applied to the model output tensor.

        Args:
            model_output (torch.Tensor): The model output tensor.

        Returns:
            torch.Tensor: The inverted rescaled tensor.
        """
        inv_scale_t = utils.t_2d_scale(
            1 / self.inference_dict["dimension_scale_factor"]
        )

        inv_grid_size = [
            self.inference_dict["number_of_slices"],
            self.inference_dict["nr_output_classes"],
            int(self.inference_dict["source_shape"][0]),
            int(self.inference_dict["source_shape"][1]),
        ]
        grid = F.affine_grid(
            theta=torch.repeat_interleave(
                inv_scale_t[:-1, :].unsqueeze(0),
                self.inference_dict["number_of_slices"],
                dim=0,
            ),
            size=inv_grid_size,
            align_corners=False,
        )
        return F.grid_sample(
            model_output,
            grid,
            align_corners=False,
            mode=mode,
            padding_mode="border",
        )

    def _postprocess_output(self, tensor: torch.Tensor, motion_track=False):
        """
        Postprocesses the output tensor and returns the segmentation in the
        original shape.

        Args:
            tensor (torch.Tensor): The output tensor from the model.

        Returns:
            np.ndarray: The segmentation in the original shape.
        """

        if not motion_track:
            the_type = np.uint8
            original_shape_segmentation = np.zeros(
                (
                    self.inference_dict["number_of_slices"],
                    self.inference_dict["original_shape"][-2],
                    self.inference_dict["original_shape"][-1],
                ),
                dtype=the_type,
            )
        else:
            the_type = np.float32
            original_shape_segmentation = np.zeros(
                (
                    self.inference_dict["number_of_slices"],
                    2,
                    self.inference_dict["original_shape"][-2],
                    self.inference_dict["original_shape"][-1],
                ),
                dtype=the_type,
            )
        original_shape_segmentation[
            ...,
            self.inference_dict["border_indices"][0] : self.inference_dict[
                "border_indices"
            ][1],
            self.inference_dict["border_indices"][2] : self.inference_dict[
                "border_indices"
            ][3],
        ] = (
            tensor.squeeze().numpy().astype(the_type)
        )

        return original_shape_segmentation.reshape(
            self.inference_dict["original_shape"].tolist()
        )


class CineSeries(BaseSeries):

    def __init__(self, folder: Path, batch_size: int = 50):
        super().__init__(folder, batch_size)

    def predict_segmentation(self, wandb_run_path: Path) -> np.ndarray:
        """
        Predict the segmentation for the DICOM data using the specified model.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.

        Returns:
            np.ndarray: The predicted segmentation for the DICOM data.
        """
        self._run_model(wandb_run_path)
        self._lv = 1.0 * (self._segmentation_prediction == 1)
        self._myo = 1.0 * (self._segmentation_prediction == 2)
        self._rv = 1.0 * (self._segmentation_prediction == 3)
        return self._segmentation_prediction

    def _compute_primary_slices(self):
        self.base_slice_num = 4
        self.mid_slice_num = 6
        self.apex_slice_num = 10

    def _forward_motion(self, model: torch.nn.Module, tensor: torch.Tensor):
        """
        Forward the pixel array through the model.

        Args:
            model (torch.nn.Module): The model to use for inference.
            tensor (torch.Tensor): The pixel array to forward through the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        model_output = torch.zeros(
            self.inference_dict["number_of_slices"],
            self.inference_dict["nr_output_classes"],
            self.inference_dict["target_size"][0],
            self.inference_dict["target_size"][1],
        )

        model.eval()
        with torch.no_grad():
            for i in range(
                0, self.inference_dict["number_of_slices"] // self.batch_size + 1
            ):
                model_output[i * self.batch_size : (i + 1) * self.batch_size] = model(
                    tensor[i * self.batch_size : (i + 1) * self.batch_size]
                )[0]

        return model_output

    def _get_motion_array(self):
        """
        ...
        """

        # return np.asarray(
        #     [
        #         np.stack(
        #             (
        #                 self.slice_data[f"slice{i+1:02}"]["pixel_array"][0],
        #                 self.slice_data[f"slice{i+1:02}"]["pixel_array"][self.es_time],
        #             ),
        #             axis=0,
        #         )
        #         for i in [self.base_slice_num, self.mid_slice_num, self.apex_slice_num]
        #     ]
        # )
        tmp_array = np.asarray(
            [
                [
                    np.stack(
                        (
                            self.slice_data[f"slice{i+1:02}"]["pixel_array"][0],
                            self.slice_data[f"slice{i+1:02}"]["pixel_array"][t],
                        ),
                        axis=0,
                    )
                    for t in range(1, self.number_of_temporal_positions)
                ]
                for i in [self.base_slice_num, self.mid_slice_num, self.apex_slice_num]
            ]
        )

        return tmp_array.reshape(-1, 2, self.rows, self.columns)

    def _run_motion(self, wandb_run_path):
        preprocessed_slices = self._preproccess_slices(
            self._get_motion_array(), motion_track=True
        )
        config = self._get_config(wandb_run_path)
        self.inference_dict["target_pixdim"] = torch.tensor(
            config["data"]["target_pixdim"]
        )
        self.inference_dict["target_size"] = torch.tensor(config["data"]["target_size"])
        self.inference_dict["grid_sample_modes"] = [
            config["data"]["image_grid_sample_mode"]
        ]
        self.inference_dict["nr_output_classes"] = config["unet"]["nr_output_classes"]
        the_model = UNet2d(
            nr_input_channels=config["unet"]["nr_image_channels"],
            channels_list=config["unet"]["channels_list"],
            nr_output_classes=config["unet"]["nr_output_classes"],
            nr_output_scales=config["unet"]["nr_output_scales"],
        ).to("cpu")
        model_weights = torch.load(wandb_run_path / "files" / "last_model.pt")
        the_model.load_state_dict(model_weights)

        self.inference_dict["dimension_scale_factor"], rescaled_tensor = (
            self._rescale_tensor(preprocessed_slices)
        )
        standardised_tensor = utils.standardise(rescaled_tensor)

        model_output = self._forward_motion(the_model, standardised_tensor)
        rescale_model_output = self._invert_rescale_tensor(model_output)

        self._deformations = self._postprocess_output(
            rescale_model_output, motion_track=True
        )

    def motion_track(self, wandb_run_path: Path) -> np.ndarray:
        """
        Predict the segmentation for the DICOM data using the specified model.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.

        Returns:
            np.ndarray: ...
        """
        self._compute_primary_slices()
        self._compute_marker_points()

        self._run_motion(wandb_run_path)
        return self._deformations

    def compute_volume_curve(self, structure="lv"):

        pixdim = self._get_pixel_spacing()
        scale_factor = 0.001 * pixdim[0] * pixdim[1] * pixdim[2]
        if structure == "lv":
            return np.sum(self._lv, axis=(0, 2, 3)) * scale_factor.numpy()
        elif structure == "rv":
            return np.sum(self._rv, axis=(0, 2, 3)) * scale_factor.numpy()
        elif structure == "myo":
            return np.sum(self._myo, axis=(0, 2, 3)) * scale_factor.numpy()
        else:
            raise ValueError(f"Unknown structure: {structure}, should be lv, rv or myo")

    def compute_ejection_fraction(self, curve):
        ed_vol = curve[0]
        self.es_time = np.argmin(curve)
        es_vol = curve[self.es_time]
        return (ed_vol - es_vol) / ed_vol

    def _compute_marker_points(self):
        self._lv_center_points = []
        self._rv_center_points = []
        self._rv_insertion_points = []
        for slice_num in [self.base_slice_num, self.mid_slice_num, self.apex_slice_num]:
            tmp_lv_center_pts = []
            tmp_rv_center_pts = []
            tmp_rv_insertion_pts = []
            for t in range(self._segmentation_prediction.shape[1]):
                the_myo = self._myo[slice_num, t]
                the_rv = self._rv[slice_num, t]
                the_lv = self._lv[slice_num, t]

                try:
                    tmp_lv_center_pts.append(
                        [
                            int(np.mean(np.where(the_lv > 0)[0])),
                            int(np.mean(np.where(the_lv > 0)[1])),
                        ]
                    )
                    tmp_rv_center_pts.append(
                        [
                            int(np.mean(np.where(the_rv > 0)[0])),
                            int(np.mean(np.where(the_rv > 0)[1])),
                        ]
                    )
                except:
                    tmp_lv_center_pts.append([0, 0])
                    tmp_rv_center_pts.append([0, 0])

                self._lv_center_points.append(tmp_lv_center_pts)
                self._rv_center_points.append(tmp_rv_center_pts)

                epi = the_lv + the_myo
                # Extract epicardial contour
                contours, _ = cv2.findContours(
                    cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                try:
                    epi_contour = contours[0][:, 0, :]
                    septum = []
                    dilate_iter = 1
                    while len(septum) == 0 and dilate_iter < 6:
                        # Dilate the RV till it intersects with LV epicardium.
                        # Normally, this is fulfilled after just one iteration.
                        rv_dilate = cv2.dilate(
                            the_rv,
                            np.ones((3, 3), dtype=np.uint8),
                            iterations=dilate_iter,
                        )
                        dilate_iter += 1
                        for y, x in epi_contour:
                            if rv_dilate[x, y] == 1:
                                septum += [[x, y]]
                        tmp_rv_insertion_pts.append(
                            [
                                [septum[0][1], septum[0][0]],
                                [septum[-1][1], septum[-1][0]],
                            ]
                        )
                except:
                    tmp_rv_insertion_pts.append(
                        [
                            [0, 0],
                            [0, 0],
                        ]
                    )
            self._rv_insertion_points.append(tmp_rv_insertion_pts)

    def get_lv_center_points(self):
        """
        Find the left ventricular center points.
        Returns:
            List[int]: The x and y coordinates of the left ventricular center points.
        """

        return self._lv_center_points

    def get_rv_center_points(self):
        """
        Find the right ventricular center points.
        Returns:
            List[int]: The x and y coordinates of the right ventricular center points.
        """

        return self._rv_center_points

    def get_rv_insertion_points(self):
        """
        Find the right ventricular insertion point.

        Returns:
            List[int]: The x and y coordinates of the right ventricular insertion point.
        """

        return self._rv_insertion_points


class LGESeries(BaseSeries):

    def __init__(self, folder: Path, batch_size: int = 50):
        super().__init__(folder, batch_size)
        self._extract_psir()

    def _compute_primary_slices(self):
        self.base_slice_num = 4
        self.mid_slice_num = 6
        self.apex_slice_num = 10

    def _extract_psir(self):
        for i in range(self.number_of_slices):
            for j in range(len(self.slice_data[f"slice{i+1:02}"]["pixel_array"])):
                if (
                    "m"
                    not in self.slice_data[f"slice{i+1:02}"]["meta_data"][j]
                    .ImageType[2][0]
                    .lower()
                ):
                    self.slice_data[f"slice{i+1:02}"]["psir_array"] = self.slice_data[
                        f"slice{i+1:02}"
                    ]["pixel_array"][j]

    def _rescale_image(self, image, dimension_scale_factor):

        scale_t = utils.t_2d_scale(dimension_scale_factor)
        grid_size = [
            1,
            1,
            int(self.inference_dict["target_size"][0]),
            int(self.inference_dict["target_size"][1]),
        ]

        grid = F.affine_grid(
            theta=torch.repeat_interleave(
                scale_t[:-1, :].unsqueeze(0),
                1,
                dim=0,
            ),
            size=grid_size,
            align_corners=False,
        )
        return F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode="nearest",
            padding_mode="border",
        )[0, 0, ...]

    def _find_landmark(self, quadrant_image):

        pred_r = (quadrant_image == 1) + (quadrant_image == 2)
        pred_r = binary_fill_holes(pred_r)
        distance_r = distance_transform_edt(pred_r)
        distance_r[distance_r != 1] = 0

        pred_l = (quadrant_image == 2) + (quadrant_image == 3)
        pred_l = binary_fill_holes(pred_l)
        distance_l = distance_transform_edt(pred_l)
        distance_l[distance_l != 1] = 0

        # Find the intersection of the two lines to find the center of the label
        list_l = np.where(distance_l == 1)
        list_r = np.where(distance_r == 1)

        skip_l = int(len(list_l[0]) / 5)
        skip_r = int(len(list_r[0]) / 5)

        b_l = 1
        b_r = -1
        a_l = np.mean(list_l[0][skip_l:-skip_l]) - b_l * np.mean(
            list_l[1][skip_l:-skip_l]
        )
        a_r = np.mean(list_r[0][skip_r:-skip_r]) - b_r * np.mean(
            list_r[1][skip_r:-skip_r]
        )

        inter_r = int((a_l - a_r) / (b_r - b_l))
        inter_c = int(a_l + b_l * inter_r)

        return [inter_r, inter_c]

    def _run_crop_model(
        self, wandb_run_path: Path, center_image: np.ndarray, image_type: str = "pixel"
    ) -> None:
        """
        Runs the model inference on preprocessed slices.

        This method loads the model weights, preprocesses the input data, runs the model and
        postprocesses the output.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.
        """
        preprocessed_slices = self._preproccess_slices(
            self._get_array(image_type=image_type)
        )

        preprocessed_center_image = center_image[
            self.inference_dict["border_indices"][0] : self.inference_dict[
                "border_indices"
            ][1],
            self.inference_dict["border_indices"][2] : self.inference_dict[
                "border_indices"
            ][3],
        ]
        config = self._get_config(wandb_run_path)

        # Load the model weights
        self.inference_dict["target_pixdim"] = torch.tensor(
            config["data"]["target_pixdim"]
        )
        self.inference_dict["target_size"] = torch.tensor(config["data"]["target_size"])
        self.inference_dict["original_target_size"] = torch.tensor(
            config["data"]["target_size"]
        )
        self.inference_dict["grid_sample_modes"] = [
            config["data"]["image_grid_sample_mode"]
        ]
        self.inference_dict["nr_output_classes"] = config["unet"]["nr_output_classes"]
        the_model = UNet2d(
            nr_input_channels=config["unet"]["nr_image_channels"],
            channels_list=config["unet"]["channels_list"],
            nr_output_classes=config["unet"]["nr_output_classes"],
            nr_output_scales=config["unet"]["nr_output_scales"],
        ).to("cpu")
        model_weights = torch.load(wandb_run_path / "files" / "last_model.pt")
        the_model.load_state_dict(model_weights)

        self.inference_dict["target_size"] = torch.tensor([480, 480]).to(torch.int32)

        # Preprocess the input data
        self.inference_dict["dimension_scale_factor"], rescaled_tensor = (
            self._rescale_tensor(preprocessed_slices)
        )

        rescale_center_image = self._rescale_image(
            torch.Tensor(preprocessed_center_image[np.newaxis, np.newaxis, ...]),
            self.inference_dict["dimension_scale_factor"],
        )
        center_point = self._find_landmark(rescale_center_image)

        standardised_tensor = utils.standardise(rescaled_tensor)
        # Crop the input data
        standardised_tensor = standardised_tensor[
            :,
            :,
            center_point[1]
            - self.inference_dict["original_target_size"][0] // 2 : center_point[1]
            + self.inference_dict["original_target_size"][0] // 2,
            center_point[0]
            - self.inference_dict["original_target_size"][1] // 2 : center_point[0]
            + self.inference_dict["original_target_size"][1] // 2,
        ]

        # Run the model
        model_output = self._forward_model(the_model, standardised_tensor)

        model_prediction = torch.argmax(model_output, dim=1, keepdim=True).float()

        full_size_prediction = torch.zeros(
            self.inference_dict["number_of_slices"],
            1,
            self.inference_dict["target_size"][0],
            self.inference_dict["target_size"][1],
        )

        full_size_prediction[
            :,
            :,
            center_point[1]
            - self.inference_dict["original_target_size"][0] // 2 : center_point[1]
            + self.inference_dict["original_target_size"][0] // 2,
            center_point[0]
            - self.inference_dict["original_target_size"][1] // 2 : center_point[0]
            + self.inference_dict["original_target_size"][1] // 2,
        ] = model_prediction

        # Postprocess the output
        rescale_model_prediction = self._invert_rescale_tensor(
            full_size_prediction, mode="nearest"
        )

        self._segmentation_prediction = self._postprocess_output(
            rescale_model_prediction
        )

    def predict_segmentation(
        self, wandb_run_path: Path, center_image: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict the segmentation for the DICOM data using the specified model.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.

        Returns:
            np.ndarray: The predicted segmentation for the DICOM data.
        """
        if center_image is not None:
            self._run_crop_model(wandb_run_path, center_image, image_type="psir")
        else:
            self._run_model(wandb_run_path, image_type="psir")


class PCFlowSeries:
    """Phase-contrast flow series.

    Loads all DICOMs in a PC_AORTA / PC_MPA folder and splits them into
    magnitude and phase components.

    Phase detection: any DICOM whose ImageType contains the standalone element
    'P' is phase (velocity-encoded).  This covers both Siemens style
    ('ORIGINAL','PRIMARY','P',...) and Philips style ('ORIGINAL','PRIMARY',
    'PHASE CONTRAST M','P','PCA').

    When multiple magnitude sub-types are present, priority is:
      M > M_FFE > M_PCA > MAG > (most-common fallback)
    M_FFE (standard FFE magnitude) gives the clearest anatomy for PC display.

    Attributes:
        magnitude_frames (list[np.ndarray]): per-frame 2-D magnitude arrays.
        phase_frames     (list[np.ndarray]): per-frame 2-D phase arrays.
        magnitude_meta   (list[pydicom.Dataset]): DICOM headers for magnitude frames.
        phase_meta       (list[pydicom.Dataset]): DICOM headers for phase frames.
        rows, columns    (int): image dimensions.
        n_mag, n_phase   (int): number of temporal frames per component.
    """

    _MAG_PRIORITY = ["M", "M_FFE", "M_PCA", "MAG"]

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        (
            self.magnitude_frames,
            self.phase_frames,
            self.magnitude_meta,
            self.phase_meta,
            self.rows,
            self.columns,
        ) = self._load_and_split()
        self.n_mag   = len(self.magnitude_frames)
        self.n_phase = len(self.phase_frames)

    def _load_and_split(self):
        from natsort import natsorted

        files = natsorted([
            f for f in self.folder.iterdir()
            if f.is_file() and not f.stem.startswith(".")
        ])

        mag_by_type: dict = {}   # type_str → list of (instance_num, ds)
        phase_list:  list = []   # list of (instance_num, ds)

        for f in files:
            try:
                ds = pydicom.dcmread(f)
                if "PixelData" not in ds:
                    continue
                img_type = list(getattr(ds, "ImageType", []))
                inst     = int(getattr(ds, "InstanceNumber", 0))
                # Phase detection: look for standalone 'P' anywhere in ImageType.
                if "P" in img_type:
                    phase_list.append((inst, ds))
                else:
                    component = img_type[2] if len(img_type) > 2 else "M"
                    mag_by_type.setdefault(component, []).append((inst, ds))
            except Exception:
                continue

        # Pick best magnitude sub-type
        best_type = None
        for t in self._MAG_PRIORITY:
            if t in mag_by_type:
                best_type = t
                break
        if best_type is None and mag_by_type:
            best_type = max(mag_by_type, key=lambda t: len(mag_by_type[t]))

        mag_list = mag_by_type.get(best_type, []) if best_type else []

        def _sort_and_split(pairs):
            pairs = sorted(pairs, key=lambda p: p[0])
            frames = [ds.pixel_array for _, ds in pairs]
            metas  = [ds               for _, ds in pairs]
            return frames, metas

        mag_frames, mag_metas     = _sort_and_split(mag_list)
        phase_frames, phase_metas = _sort_and_split(phase_list)

        all_ds = (mag_metas or phase_metas)
        rows    = all_ds[0].Rows    if all_ds else 64
        columns = all_ds[0].Columns if all_ds else 64

        return mag_frames, phase_frames, mag_metas, phase_metas, rows, columns


class PerfusionSeries(BaseSeries):
    """Perfusion (PERF_REST / PERF_STRESS) series.

    The acquisition protocol yields 4 short-axis slices where slice 3 (1-indexed)
    is a test/calibration slice and must be excluded.  The filtering is applied
    automatically when exactly 4 slices are present; other slice counts are left
    unchanged so the class is safe for non-standard acquisitions.
    """

    # 0-based index of the test slice within a 4-slice acquisition
    _TEST_SLICE_IDX: int = 1
    _TEST_SLICE_N:   int = 4

    def __init__(self, folder: Path, batch_size: int = 50):
        super().__init__(folder, batch_size)
        print(f"[PerfusionSeries] loaded {self.number_of_slices} slices from {folder}")
        if self.number_of_slices == self._TEST_SLICE_N:
            self._drop_test_slice()
            print(f"[PerfusionSeries] test slice (idx {self._TEST_SLICE_IDX}) dropped → {self.number_of_slices} slices remain")

    def _drop_test_slice(self) -> None:
        """Remove the test/calibration slice and renumber remaining slices."""
        old_keys = sorted(self.slice_data.keys())  # ["slice01", ..., "slice04"]
        new_data = {}
        new_idx = 1
        for i, key in enumerate(old_keys):
            if i == self._TEST_SLICE_IDX:
                continue
            new_data[f"slice{new_idx:02}"] = self.slice_data[key]
            new_idx += 1
        self.slice_data = new_data
        self.number_of_slices = len(new_data)
