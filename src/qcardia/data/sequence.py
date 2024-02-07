from copy import deepcopy

import numpy as np
import pydicom
from natsort import natsorted


class BaseSequence:
    """
    The base class used to represent the DICOM data of a CMR sequence acquisition.

    This class provides methods to load DICOM data from a specified folder and
    organize it into a dictionary. The dictionary keys are the slice numbers and
    the values are another dictionary containing the pixel array, slice position,
    and meta data for each slice.

    Attributes:
        folder (Path): The folder path where the DICOM files are located.
        slice_data (dict): A dictionary where the keys are the slice numbers and
        the values are another dictionary containing the pixel array, slice
        position, and meta data for each slice.
        number_of_slices (int): The total number of slices in the DICOM data.
    Methods:
        _load_data: Load DICOM data from a specified folder.
        _get_slices_from_positions: Get slice indices and unique positions from
        given positions and orientations.
    """

    def __init__(self, folder):
        self.folder = folder
        self.slice_data, self.number_of_slices = self._load_data()
        self.pixel_array = self._get_array()
        reshaped_pixel_array = self._reshape_array()
        self.start_idx_0, self.stop_idx_0, self.start_idx_1, self.stop_idx_1 = (
            self._get_borders(reshaped_pixel_array)
        )
        self.preprocessed_pixel_array = self._strip_borders(reshaped_pixel_array)

    def _load_data(self):
        """
        Load DICOM data from a specified folder.

        This function reads DICOM files from the given folder, extracts relevant
        information such as slice position, slice orientation, and temporal
        positions, and organizes the data into a dictionary. The dictionary keys
        are the slice numbers and the values are another dictionary containing the
        pixel array, slice position, and meta data for each slice.

        Args:
            folder (Path): The folder path where the DICOM files are located.

        Returns:
            dict: A dictionary where the keys are the slice numbers and the values
            are another dictionary containing the pixel array, slice position, and
            meta data for each slice.

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

        return slices_dict, number_of_slices

    def _get_slices_from_positions(self, positions, orientations):
        """
        Get slice indices and unique positions from given positions and orientations.

        This function calculates the true positions by taking the dot product of each position
        and the cross product of the orientation vectors. It then finds the unique positions,
        sorts them in descending order, and assigns a slice index to each position.

        Args:
            positions (list): A list of positions.
            orientations (list): A list of orientations.

        Returns:
            tuple: A tuple containing a list of slice indices and a list of unique positions.

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

    def _get_array(self):
        """
        Get the pixel array for all slice/times.

        Shape depends on the number of slices/frames and the size of the pixel arrays.

        Returns:
            ndarray: The pixel array for all slices/times.
        """

        return np.asarray(
            [
                self.slice_data[f"slice{i+1:02}"]["pixel_array"]
                for i in range(self.number_of_slices)
            ]
        )

    def _reshape_array(self):
        """
        Reshape the pixel array to have all the times/slices in the
        batch dimension. With one channel.

        The pixel array is reshaped to have the following dimensions:
        (number_of_slices, 1, height, width).

        Returns:
            ndarray: The reshaped pixel array.
        """

        return self.pixel_array.reshape(
            -1,
            1,
            self.pixel_array.shape[-2],
            self.pixel_array.shape[-1],
        )

    def _get_borders(self, reshaped_pixel_array):
        summed_pixel_array = np.sum(reshaped_pixel_array, axis=(0, 1))
        borderless_idxs_0 = np.nonzero(np.any(summed_pixel_array, axis=1))[0]
        borderless_idxs_1 = np.nonzero(np.any(summed_pixel_array, axis=0))[0]

        start_idx_0, stop_idx_0 = borderless_idxs_0[0], borderless_idxs_0[-1] + 1
        start_idx_1, stop_idx_1 = borderless_idxs_1[0], borderless_idxs_1[-1] + 1

        return start_idx_0, stop_idx_0, start_idx_1, stop_idx_1

    def _strip_borders(self, reshaped_pixel_array):
        """
        Strip the borders from the pixel array.

        Returns:
            ndarray: The borderless pixel array.
        """

        return reshaped_pixel_array[
            ..., self.start_idx_0 : self.stop_idx_0, self.start_idx_1 : self.stop_idx_1
        ]
