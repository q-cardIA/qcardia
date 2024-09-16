"""
This module contains the BaseSeries class which is used for handling series of 
DICOM images.

The BaseSeries class provides methods for loading DICOM data from a folder, 
preprocessing the data, running a model on the data, and postprocessing the model 
output. The model is specified by a path to a Weights & Biases run.

Classes:
    BaseSeries: A base class for handling sequences of DICOM images.
"""

from pathlib import Path
from typing import List

import numpy as np
import pydicom
import torch
import yaml
from natsort import natsorted
from qcardia_models.models import UNet2d
from skimage.measure import find_contours
from torch.nn import functional as F

import src.qcardia.utils as utils


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
        self.slice_data, self.number_of_slices, self.rows, self.columns = (
            self._load_data()
        )
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
        self._lv = 1.0 * (self._segmentation_prediction == 1)
        self._myo = 1.0 * (self._segmentation_prediction == 2)
        self._rv = 1.0 * (self._segmentation_prediction == 3)
        return self._segmentation_prediction

    def _run_model(self, wandb_run_path: Path, image_type: str = "pixel") -> None:
        """
        Runs the model inference on preprocessed slices.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.
        """
        preprocessed_slices = self._preproccess_slices(image_type)
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
        for file in files:
            the_ds = pydicom.read_file(file)
            all_dicom_data.append(the_ds)
            slice_position.append(the_ds.ImagePositionPatient)
            slice_orientation.append(the_ds.ImageOrientationPatient)
            if int(the_ds.NumberOfTemporalPositions) == 1:
                temporal_positions.append(int(the_ds.InstanceNumber))
            else:
                temporal_positions.append(int(the_ds.TemporalPositionIdentifier))

        # Gets unique positions from given positions and orientations.
        # assigns a slice index to each image based on its position.
        indices_of_slices, the_slice_positions = self._get_slices_from_positions(
            slice_position, slice_orientation
        )

        slices_dict = {}
        number_of_slices = len(the_slice_positions)

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

    def _preproccess_slices(self, image_type: str = "pixel"):
        """
        Preprocesses the slices by performing various operations such as reshaping,
        border stripping, normalization, and conversion to tensors.

        Returns:
            torch.Tensor: The preprocessed pixel array without borders.
        """
        pixel_array = self._get_array(image_type)
        self.inference_dict["original_shape"] = torch.tensor(pixel_array.shape)
        reshaped_pixel_array = self._reshape_array(pixel_array)

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

        return np.asarray(
            [
                self.slice_data[f"slice{i+1:02}"][f"{image_type}_array"]
                for i in range(self.number_of_slices)
            ]
        )

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
            self.inference_dict["target_size"][0],
            self.inference_dict["target_size"][1],
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

    def _invert_rescale_tensor(self, model_output: torch.Tensor):
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
            mode="bicubic",
            padding_mode="border",
        )

    def _postprocess_output(self, tensor: torch.Tensor):
        """
        Postprocesses the output tensor and returns the segmentation in the
        original shape.

        Args:
            tensor (torch.Tensor): The output tensor from the model.

        Returns:
            np.ndarray: The segmentation in the original shape.
        """

        original_shape_segmentation = np.zeros(
            (
                self.inference_dict["number_of_slices"],
                self.inference_dict["original_shape"][-2],
                self.inference_dict["original_shape"][-1],
            ),
            dtype=np.uint8,
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
            tensor.squeeze().numpy().astype(np.uint8)
        )

        return original_shape_segmentation.reshape(
            self.inference_dict["original_shape"].tolist()
        )


class CineSeries(BaseSeries):

    def __init__(self, folder: Path, batch_size: int = 50):
        super().__init__(folder, batch_size)

    def _compute_primary_slices(self):
        self.base_slice_num = 4
        self.mid_slice_num = 6
        self.apex_slice_num = 10

    def _compute_rv_insertion_points(self):
        self.lv_center_point = []
        for slice_num in [self.base_slice_num, self.mid_slice_num, self.apex_slice_num]:
            tmp_center_pts = []
            for t in range(self._segmentation_prediction.shape[1]):
                the_myo = self._myo[slice_num, t]
                the_rv = self._rv[slice_num, t]
                the_lv = self._lv[slice_num, t]
                find_contours(the_myo + the_lv)
                tmp_center_pts.append(
                    [
                        int(np.mean(np.where(the_lv > 0)[2])),
                        int(np.mean(np.where(the_lv > 0)[3])),
                    ]
                )
            self.lv_center_point.append(tmp_center_pts)
        self._rv_insertion_points = [[0, 0], [1, 1]]

    def get_rv_insertion_points(self):
        """
        Find the right ventricular insertion point.

        Returns:
            List[int]: The x and y coordinates of the right ventricular insertion point.
        """

        self._compute_rv_insertion_points()
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

    def predict_segmentation(self, wandb_run_path: Path) -> np.ndarray:
        """
        Predict the segmentation for the DICOM data using the specified model.

        Args:
            wandb_run_path (Path): The path to the WandB run directory.

        Returns:
            np.ndarray: The predicted segmentation for the DICOM data.
        """
        self._run_model(wandb_run_path, image_type="psir")
        self._lv = 1.0 * (self._segmentation_prediction == 1)
        self._myo = 1.0 * (self._segmentation_prediction == 2)
        self._rv = 1.0 * (self._segmentation_prediction == 3)
        return self._segmentation_prediction
