from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from qcardia_data.setup.reformat.utils import (
    get_ed_es_idxs,
    reformat_sa_cine_scans,
    reorder_df_columns,
)


def fix_orientation_mm1(orientations, example_data: nib.Nifti1Image):
    orientations[:, 1] *= -1  # mirror all axes
    temp_data = example_data.as_reoriented(orientations)

    if temp_data.header["pixdim"][0] < 0:  # correct so base is at the top
        orientations[2, 1] *= -1.0
        temp_data = example_data.as_reoriented(orientations)

    if "".join(nib.aff2axcodes(temp_data.affine)) == "PLI":
        orientations[[0, 1], :] = orientations[[1, 0], :]
        orientations[0, 1] *= -1.0

    return orientations


def reformat_mm1(
    original_data_folder_path: Path,
    original_csv_file_path: Path,
    reformatted_data_folder_path: Path,
):
    reformatted_dataset_folder_path = reformatted_data_folder_path / "mm1"
    reformatted_dataset_csv_path = reformatted_data_folder_path / "mm1.csv"

    reformatted_dataset_folder_path.mkdir(parents=True, exist_ok=True)
    csv_df = pd.read_csv(original_csv_file_path)

    for idx, subject_name in enumerate(tqdm(csv_df["External code"])):
        subject_folder_path = original_data_folder_path / subject_name
        if not subject_folder_path.exists():
            raise FileNotFoundError(f"subject folder not found: {subject_folder_path}")
        cine_path = subject_folder_path / f"{subject_name}_sa.nii.gz"
        gt_path = subject_folder_path / f"{subject_name}_sa_gt.nii.gz"

        cine = nib.load(cine_path)
        gt = nib.load(gt_path)

        ed_idx, es_idx = get_ed_es_idxs(gt.get_fdata())
        if ed_idx != csv_df.iloc[idx]["ED"] or es_idx != csv_df.iloc[idx]["ES"]:
            tqdm.write(
                f"Recalculated time frames for subject {subject_name}:"
                + f' ED: {csv_df.iloc[idx]["ED"]} -> {ed_idx}'
                + f', ES: {csv_df.iloc[idx]["ES"]} -> {es_idx}'
            )
            csv_df.loc[csv_df["External code"] == subject_name, "ED"] = ed_idx
            csv_df.loc[csv_df["External code"] == subject_name, "ES"] = es_idx

        ed_gt = gt.slicer[..., ed_idx]
        es_gt = gt.slicer[..., es_idx]

        cine_processed, gt_processed = reformat_sa_cine_scans(
            cine, ed_gt, es_gt, process_orientations_func=fix_orientation_mm1
        )
        subject_path = reformatted_dataset_folder_path / subject_name
        subject_path.mkdir(parents=False, exist_ok=True)

        nib.save(cine_processed, subject_path / f"{subject_name}_sa_cine.nii.gz")
        nib.save(gt_processed, subject_path / f"{subject_name}_sa_cine_gt.nii.gz")

    csv_df = csv_df.rename(columns={"External code": "SubjectID"})
    csv_df = csv_df.drop(columns=["Unnamed: 0"])
    csv_df = reorder_df_columns(csv_df)
    csv_df.to_csv(reformatted_dataset_csv_path)

    print(
        "Finished reformatting M&Ms dataset,",
        f"meta data saved at `{reformatted_dataset_csv_path.resolve()}`\n",
    )
