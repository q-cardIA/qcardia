from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from qcardia_data.setup.reformat.utils import (
    get_frame_index,
    reformat_sa_cine_scans,
    reorder_df_columns,
    set_nifti_zooms,
)


def fix_orientation_mm2(orientations, example_data: nib.Nifti1Image):
    orientations[orientations[:, 0] != 1.0, 1] *= -1  # no axis 1 mirror
    temp_data = example_data.as_reoriented(orientations)

    if (
        temp_data.header["pixdim"][0] < 0
        and "".join(nib.aff2axcodes(temp_data.affine)) != "PRI"
    ):  # correct so base is at the top
        orientations[2, 1] *= -1.0
        temp_data = example_data.as_reoriented(orientations)

    if "".join(nib.aff2axcodes(temp_data.affine)) == "PRI":
        orientations[[0, 1], :] = orientations[[1, 0], :]
        if orientations[0, 0] == 0.0:
            orientations[[0, 1], 1] = -orientations[[0, 1], 1]
        temp_data = example_data.as_reoriented(orientations)
    return orientations


def reformat_mm2(
    original_data_folder_path: Path,
    original_csv_file_path: Path,
    reformatted_data_folder_path: Path,
):
    reformatted_dataset_folder_path = reformatted_data_folder_path / "mm2"
    reformatted_dataset_csv_path = reformatted_data_folder_path / "mm2.csv"

    reformatted_dataset_folder_path.mkdir(parents=True)
    csv_df = pd.read_csv(original_csv_file_path)
    es_idxs, ed_idxs = [], []

    for subject_name in tqdm(csv_df["SUBJECT_CODE"]):
        subject_name = f"{int(subject_name):03}"
        subject_folder_path = original_data_folder_path / subject_name
        if not subject_folder_path.exists():
            raise FileNotFoundError(f"subject folder not found: {subject_folder_path}")
        cine_path = subject_folder_path / f"{subject_name}_SA_CINE.nii.gz"
        ed_gt_path = subject_folder_path / f"{subject_name}_SA_ED_gt.nii.gz"
        es_gt_path = subject_folder_path / f"{subject_name}_SA_ES_gt.nii.gz"
        ed_path = subject_folder_path / f"{subject_name}_SA_ED.nii.gz"
        es_path = subject_folder_path / f"{subject_name}_SA_ES.nii.gz"

        cine = set_nifti_zooms(nib.load(cine_path))
        ed_gt = nib.load(ed_gt_path)
        es_gt = nib.load(es_gt_path)
        ed = nib.load(ed_path)
        es = nib.load(es_path)

        ed_idxs.append(get_frame_index(cine.get_fdata(), ed.get_fdata()))
        es_idxs.append(get_frame_index(cine.get_fdata(), es.get_fdata()))

        cine_processed, gt_processed = reformat_sa_cine_scans(
            cine, ed_gt, es_gt, process_orientations_func=fix_orientation_mm2
        )
        subject_path = reformatted_dataset_folder_path / subject_name
        subject_path.mkdir(parents=False, exist_ok=True)

        nib.save(cine_processed, subject_path / f"{subject_name}_sa_cine.nii.gz")
        nib.save(gt_processed, subject_path / f"{subject_name}_sa_cine_gt.nii.gz")
        print("=", end="", flush=True)

    csv_df = csv_df.rename(
        columns={
            "SUBJECT_CODE": "SubjectID",
            "DISEASE": "Pathology",
            "VENDOR": "VendorName",
            "SCANNER": "Scanner",
            "FIELD": "Field",
        }
    )
    csv_df["SubjectID"] = [
        f"{int(subject_id):03}" for subject_id in csv_df["SubjectID"]
    ]
    csv_df["VendorName"] = (
        csv_df["VendorName"]
        .str.replace("SIEMENS", "Siemens")
        .str.replace("Philips Medical Systems", "Philips")
        .str.replace("GE MEDICAL SYSTEMS", "GE")
    )
    csv_df["Vendor"] = (
        csv_df["VendorName"]
        .str.replace("Siemens", "A")
        .str.replace("Philips", "B")
        .str.replace("GE", "C")
    )
    csv_df["ED"] = ed_idxs
    csv_df["ES"] = es_idxs
    csv_df = reorder_df_columns(csv_df)
    csv_df.to_csv(reformatted_dataset_csv_path)

    print(
        "Finished reformatting M&Ms-2 dataset,",
        f"meta data saved at `{reformatted_dataset_csv_path.resolve()}`\n",
    )
