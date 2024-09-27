from pathlib import Path

from qcardia_data.utils import sample_from_csv_by_group


def test_subjects_mm1(data_path: Path) -> list[str]:
    """Generate test subjects for M&Ms dataset.

    Selects 15 subjects for each of the four vendors from the M&Ms dataset, for a total
    of 60 test subjects.

    Args:
        data_path (Path): Path to the data folder

    Returns:
        list[str]: list of M&Ms test subject IDs
    """
    return sample_from_csv_by_group(
        data_path / "reformatted_data" / "mm1.csv",
        sample_nr=15,
        group_key="Vendor",
        output_key="SubjectID",
        seed=1234,
    )


def test_subjects_mm2() -> list[str]:
    """Generate test subjects for M&Ms-2 dataset.

    Based on the original challenge test set, but excluding subjects with possible
    overlap with the original M&Ms challenge dataset.

    Returns:
        list[str]: list of M&Ms-2 test subject IDs
    """
    return [f"{subject_id:03}" for subject_id in range(281, 361)]


def test_subjects_kcl(data_path: Path):
    dirs = [f for f in (data_path / "reformatted_data").iterdir() if f.is_dir()]
    print(dirs)
    subject_dict = {}
    for dir in dirs:
        if "kcl_" in dir.stem:
            print(f"Generating split for {dir.name}")
            sample_list = sample_from_csv_by_group(
                f"{dir}.csv",
                sample_nr=5,
                group_key="Pathology",
                output_key="SubjectID",
                seed=1234,
            )
            subject_dict[dir.name] = sample_list
    return subject_dict


def test_subjects_emidec():
    return [f"{subject_id:03}" for subject_id in range(0, 12)]


def test_subjects_myops():
    return [f"{i:03d}" for i in range(101, 105)]
