from warnings import warn

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from monai.data.utils import affine_to_spacing
from natsort import natsorted


def set_nifti_zooms(nifti: nib.Nifti1Image) -> nib.Nifti1Image:
    """Set the zooms of a nifti image to the spacing of the affine matrix."""
    norm = affine_to_spacing(nifti.affine, r=nifti.header["dim"][0])
    nifti.header.set_zooms(norm)
    return nifti


def reformat_sa_cine_scans(
    cine: nib.Nifti1Image,
    ed_gt: nib.Nifti1Image,
    es_gt: nib.Nifti1Image,
    process_orientations_func: callable = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Reformat the short axis cine scans and ground truth masks to have a consistent
    orientations.

    Args:
        cine (.nifti1.Nifti1Image): 4D short axis cine scan
        ed_gt (nib.Nifti1Image): 3D ground truth mask for end-diastole timeframe
        es_gt (nib.Nifti1Image): 3D ground truth mask for end-systole timeframe
        process_orientations_func (callable, optional): callable function to further
            process the image to get the correct orientation, can be used for specific
            cases. Defaults to None.

    Returns:
        nib.Nifti1Image, nib.Nifti1Image: reoriented cine and ground truth nifti images
    """
    ed_gt.header["dim"][0] = 4  # dim[0] = number of dimensions
    ed_gt.header["dim"][4] = 2
    ed_gt.header["pixdim"][4] = 1.0
    dataobj = np.stack([ed_gt.get_fdata(), es_gt.get_fdata()], axis=-1)
    gt = nib.Nifti1Image(
        dataobj=dataobj, affine=ed_gt.affine, header=ed_gt.header, dtype=np.int64
    )

    cine_img_summed = np.sum(cine.get_fdata(), axis=(-1, -2))
    borderless_idxs_0 = np.nonzero(np.any(cine_img_summed, axis=1))[0]
    borderless_idxs_1 = np.nonzero(np.any(cine_img_summed, axis=0))[0]

    start_idx_0, stop_idx_0 = borderless_idxs_0[0], borderless_idxs_0[-1] + 1
    start_idx_1, stop_idx_1 = borderless_idxs_1[0], borderless_idxs_1[-1] + 1

    cine_borderless = cine.slicer[start_idx_0:stop_idx_0, start_idx_1:stop_idx_1, ...]
    gt_borderless = gt.slicer[start_idx_0:stop_idx_0, start_idx_1:stop_idx_1, ...]

    orientations = nib.orientations.io_orientation(cine_borderless.affine)

    # force third dimension to stay there
    if orientations[2, 0] == 0.0:
        orientations[:2, 0] -= 1
        orientations[2, 0] = 2.0
    elif orientations[2, 0] == 1.0:
        orientations[orientations[:, 0] == 2.0, 0] = 1.0
        orientations[2, 0] = 2.0

    orientations[:2, 0] = orientations[:2, 0][::-1]  # swap first and second axes
    if process_orientations_func is not None:
        orientations = process_orientations_func(orientations, gt_borderless)

    cine_processed = set_nifti_zooms(cine_borderless.as_reoriented(orientations))
    gt_processed = set_nifti_zooms(gt_borderless.as_reoriented(orientations))

    return cine_processed, gt_processed


def get_frame_index(cine: np.ndarray, frame: np.ndarray) -> int:
    """Get the index of a 3D time frame in a 4D cine scan.

    Args:
        cine (np.ndarray): 4D cine scan with the timeframe dimension as the last
        frame (np.ndarray): 3D time frame to find in the cine scan

    Returns:
        int: timeframe index of the frame in the cine scan
    """
    for frame_nr in range(cine.shape[-1]):
        if np.abs(np.sum(cine[..., frame_nr] - frame)) <= 0.0:
            return frame_nr


def get_ed_es_idxs(cine_gt: np.ndarray) -> tuple[int, int]:
    """Get the annotated ED and ES timeframe indexes from a 4D cine ground truth mask.

    Args:
        cine_gt (np.ndarray): 4D ground truth mask with labeled ED and ES timeframes,
            with the timeframe dimension as the last

    Returns:
        tuple[int, int]: ED and ES timeframe indexes respectively
    """
    idx = np.nonzero(np.sum(cine_gt, axis=(0, 1, 2)))[0]  # two annotated frame indexes

    # get volumes of the two annotated frames
    vols = np.array(
        [
            np.sum(cine_gt[..., idx[0]] > 0),
            np.sum(cine_gt[..., idx[1]] > 0),
        ]
    )

    ed_est = idx[np.argmax(vols)]  # frame with the largest volume is ED
    es_est = idx[np.argmin(vols)]  # frame with the smallest volume is ES
    return ed_est, es_est


def reorder_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """reorder the columns of a dataframe to have the SubjectID and Pathology columns
    first.

    Args:
        df (pd.DataFrame): dataframe with SubjectID and Pathology columns, and other
            columns

    Returns:
         pd.DataFrame: dataframe with SubjectID and Pathology columns first
    """
    reordered_cols = ["SubjectID", "Pathology"]
    reordered_cols += [col for col in df.columns.to_list() if col not in reordered_cols]
    return df[reordered_cols]


def load_dicom_data(folder):
    """Helper function to load dicom data into a dictionary that can easily be converted to Nifti format
    Args:
        folder: folder containing the dicom data
    Returns:
        Dict of dicts with pixel data, slice position and meta data of the dicom folder.
    """
    files = natsorted(
        [
            f
            for f in folder.iterdir()
            if (
                f.is_file()
                and not f.stem.startswith(".")
                and "dicomdir" not in str(f).lower()
            )
        ]
    )
    all_dicom_data = []
    slice_position = []
    slice_orientation = []
    temporal_positions = []
    for file in files:
        the_ds = pydicom.read_file(file)

        assert (
            "SeriesInstanceUID" in the_ds
        ), f"Invalid Dicom file: SeriesInstanceUID not found in {file}"
        assert (
            "InstanceNumber" in the_ds
        ), f"Invalid Dicom file: InstanceNumer not found in {file}"
        assert (
            "ImageOrientationPatient" in the_ds
        ), f"Invalid Dicom file: ImageOrientationPatient not found in {file}"
        assert (
            "ImagePositionPatient" in the_ds
        ), f"Invalid Dicom file: ImagePositionPatient not found in {file}"

        all_dicom_data.append(the_ds)
        slice_position.append(the_ds.ImagePositionPatient)
        slice_orientation.append(the_ds.ImageOrientationPatient)
        if "NumberOfTemporalPositions" not in the_ds:
            if the_ds.Manufacturer != "SIEMENS":
                warn(f"NumberOfTemporalPositions not found in {file}, assuming 1")
            the_ds.NumberOfTemporalPositions = 1
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


def get_affine_from_dicom(dicom_dict, index=0):
    """Helper function to get the affine matrix from dicom meta data, see
    https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine
    and
    https://github.com/icometrix/dicom2nifti/blob/main/dicom2nifti/common.py#L667

    Args:
        dicom_dict: Sorted dictionary of dicom files
        index: Index of the data dict in the list to use for the affine matrix, None if there is no list

    Returns:
        Affine matrix that can be used for Nifti conversion
    """
    keys = list(dicom_dict.keys())

    if index is not None:
        dicom_meta_data = dicom_dict[keys[0]]["meta_data"][index]
        last_image_position = np.array(
            dicom_dict[keys[-1]]["meta_data"][index].ImagePositionPatient
        )
    else:
        dicom_meta_data = dicom_dict[keys[0]]["meta_data"]
        last_image_position = np.array(
            dicom_dict[keys[-1]]["meta_data"].ImagePositionPatient
        )

    image_position = np.array(dicom_meta_data.ImagePositionPatient)
    image_orientation = np.array(dicom_meta_data.ImageOrientationPatient)
    pixel_spacing = np.array(dicom_meta_data.PixelSpacing)

    if len(dicom_dict) == 1:
        slice_thickness = np.array(dicom_meta_data.SliceThickness)
        step = np.cross(image_orientation[:3], image_orientation[3:]) * slice_thickness
    else:
        step = (last_image_position - image_position) / (len(dicom_dict) - 1)

    affine = np.array(
        [
            [
                -image_orientation[3] * pixel_spacing[1],
                -image_orientation[0] * pixel_spacing[0],
                -step[0],
                -image_position[0],
            ],
            [
                -image_orientation[4] * pixel_spacing[1],
                -image_orientation[1] * pixel_spacing[0],
                -step[1],
                -image_position[1],
            ],
            [
                image_orientation[5] * pixel_spacing[1],
                image_orientation[2] * pixel_spacing[0],
                step[2],
                image_position[2],
            ],
            [0, 0, 0, 1],
        ]
    )
    return affine
