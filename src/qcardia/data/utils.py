import numpy as np
import pydicom
from natsort import natsorted

def load_data(folder):
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
            for f in folder.iterdir()
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

    indices_of_slices, the_slice_positions = get_slices_from_positions(
        slice_position, slice_orientation
    )

    slices_dict = {}
    number_of_slices = len(the_slice_positions)
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
        slices_dict[f"slice{i+1:02}"] = {
            "pixel_array": np.transpose(sorted_image_array, (1, 2, 0)),
            "slice_position": the_slice_positions[i],
            "meta_data": sorted_list_of_meta_data,
        }

    return slices_dict


def get_slices_from_positions(positions, orientations):
    """_summary_

    Args:
        positions: _description_
        orientations: _description_

    Returns:
        _description_
    """

    true_positions = []
    for i in range(len(positions)):
        true_positions.append(
            np.dot(positions[i], np.cross(orientations[i][0:3], orientations[i][3:6]))
        )

    unique_positions = np.unique(true_positions)
    unique_positions = np.sort(unique_positions)[::-1]
    slice_index = np.zeros(len(true_positions))
    for i in range(len(unique_positions)):
        slice_index[np.where(true_positions == unique_positions[i])] = i

    return slice_index, unique_positions
