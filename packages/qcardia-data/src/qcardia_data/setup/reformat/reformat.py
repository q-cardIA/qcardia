from pathlib import Path

from qcardia_data.setup.reformat.mm1 import _reformat_mm1
from qcardia_data.setup.reformat.mm2 import _reformat_mm2


def remove_folder(path: Path) -> None:
    """Remove a file or folder recursively

    Args:
        path (Path): path of file or folder to remove
    """
    if path.is_file():
        path.unlink()
    else:
        for child in path.iterdir():
            remove_folder(child)
        path.rmdir()


def reformat_check(
    original_data_path: Path,
    reformatted_data_path: Path,
    dataset_name: str,
    overwrite: bool,
) -> int:
    """Check if the dataset can and should be reformatted.

    Assumes no manual changes have been made to already reformatted data. If manual
    changes have been made that result in errors, removing the .csv will trigger a
    re-reformatted.

    Args:
        original_data_path (Path): path to the original data folder
        reformatted_data_path (Path): path to the reformatted data folder
        dataset_name (str): name of the dataset
        overwrite (bool): whether to force overwrite the reformatted data

    Returns:
        int: state of the dataset, either -1, 0, or 1:
            -1: unuseable (original dataset unavailable, and isn't yet reformatted)
             0: directly useable (reformatted dataset is already present)
             1: requires reformatting (reformatted dataset unavailable, invalid, or
                should be overwritten)

    """
    print(f"Checking dataset '{dataset_name}'", end=" | ")

    reformatted_dataset_folder_path = reformatted_data_path / dataset_name
    reformatted_dataset_csv_path = reformatted_data_path / f"{dataset_name}.csv"

    if reformatted_dataset_csv_path.exists():
        print(f"csv file found, overwrite = {overwrite}", end=" | ")
        if overwrite:
            print(
                f"Removing '{reformatted_dataset_folder_path}'"
                + f" and '{reformatted_dataset_csv_path}'"
            )
            remove_folder(reformatted_dataset_folder_path)
            reformatted_dataset_csv_path.unlink()
        else:
            print(f"Skipping {dataset_name}")
            return 0  # directly useable

    if reformatted_dataset_folder_path.exists():
        print(
            "Found data folder, but no .csv file"
            + f" | Removing '{reformatted_dataset_folder_path}'"
        )
        remove_folder(reformatted_dataset_folder_path)

    if not original_data_path.exists():
        print(f"original data folder not found: '{original_data_path}'")
        return -1  # unuseable

    print(f"Reformatting dataset `{dataset_name}` into `{reformatted_data_path}`:")
    return 1  # useable after reformatting


def reformat_mm1(data_path: Path, overwrite: bool = False) -> bool:
    """Reformat the M&Ms dataset from the original format to the reformatted format

    Args:
        data_path (Path): path to the data folder
        overwrite (bool, optional): whether to force overwrite the reformatted version.
            Defaults to False.

    Returns:
        bool: whether the M&Ms dataset was reformatted succesfully
    """
    original_data_path = data_path / "original_data" / "MnM"
    reformatted_data_path = data_path / "reformatted_data"

    reformat_state = reformat_check(
        original_data_path, reformatted_data_path, "mm1", overwrite
    )

    if reformat_state < 0:
        return False

    if reformat_state > 0:
        _reformat_mm1(
            original_data_path / "dataset",
            original_data_path
            / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
            reformatted_data_path,
        )
    return True


def reformat_mm2(data_path: Path, overwrite: bool = False) -> bool:
    """Reformat the M&Ms-2 dataset from the original format to the reformatted format

    Args:
        data_path (Path): path to the data folder
        overwrite (bool, optional): whether to force overwrite the reformatted version.
            Defaults to False.

    Returns:
        bool: whether the M&Ms-2 dataset was reformatted succesfully
    """
    original_data_path = data_path / "original_data" / "MnM2"
    reformatted_data_path = data_path / "reformatted_data"

    reformat_state = reformat_check(
        original_data_path, reformatted_data_path, "mm2", overwrite
    )

    if reformat_state < 0:
        return False

    if reformat_state > 0:
        _reformat_mm2(
            original_data_path / "dataset",
            original_data_path / "dataset_information.csv",
            reformatted_data_path,
        )
    return True
