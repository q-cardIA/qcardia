from pathlib import Path

from qcardia_data.setup.reformat.mm1 import reformat_mm1 as reformat_mm1_base
from qcardia_data.setup.reformat.mm2 import reformat_mm2 as reformat_mm2_base


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


def reformat_dataset_bool(
    reformatted_data_path: Path, dataset_name: str, overwrite: bool
) -> bool:
    """Check if the dataset should be reformatted

    Args:
        reformatted_data_path (Path): path to the reformatted data folder
        dataset_name (str): name of the dataset
        overwrite (bool): whether to force overwrite the reformatted data

    Returns:
        bool: whether the dataset should be reformatted
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
            return False
    if reformatted_dataset_folder_path.exists():
        print(
            "Found data folder, but no .csv file"
            + f" | Removing '{reformatted_dataset_folder_path}'"
        )
        remove_folder(reformatted_dataset_folder_path)
    print(f"Reformatting dataset `{dataset_name}` into `{reformatted_data_path}`:")
    return True


def reformat_mm1(data_path: Path, overwrite: bool = False) -> None:
    """Reformat the M&Ms dataset from the original format to the reformatted format

    Args:
        data_path (Path): path to the data folder
        overwrite (bool, optional): whether to force overwrite the reformatted version.
            Defaults to False.
    """
    original_data_path = data_path / "original_data"
    reformatted_data_path = data_path / "reformatted_data"

    if reformat_dataset_bool(reformatted_data_path, "mm1", overwrite):
        reformat_mm1_base(
            original_data_path / "MnM" / "dataset",
            original_data_path / "MnM" / "dataset_information.csv",
        )


def reformat_mm2(data_path: Path, overwrite: bool = False) -> None:
    """Reformat the M&Ms-2 dataset from the original format to the reformatted format

    Args:
        data_path (Path): path to the data folder
        overwrite (bool, optional): whether to force overwrite the reformatted version.
            Defaults to False.
    """
    original_data_path = data_path / "original_data"
    reformatted_data_path = data_path / "reformatted_data"

    if reformat_dataset_bool(reformatted_data_path, "mm2", overwrite):
        reformat_mm2_base(
            original_data_path / "MnMs2" / "dataset",
            original_data_path / "MnMs2" / "dataset_information.csv",
            reformatted_data_path,
        )
