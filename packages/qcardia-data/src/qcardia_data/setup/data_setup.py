from pathlib import Path

import yaml

from qcardia_data.setup.reformat import reformat_mm1, reformat_mm2
from qcardia_data.setup.test_splits import test_subjects_mm1, test_subjects_mm2
from qcardia_data.utils import data_to_file


def setup_cine(data_path: Path, overwrite: bool = False) -> None:
    """Reformat and save default test subject split file for M&Ms and M&Ms-2 datasets.

    Args:
        data_path (Path): path to data folder
        overwrite (bool, optional): Force reformat all datasets, overwriting any
            previous reformatted versions. Defaults to False.
    """

    mm1_available = reformat_mm1(data_path, overwrite=overwrite)
    mm2_available = reformat_mm2(data_path, overwrite=overwrite)

    split_path = data_path / "subject_splits" / "default-cine-test-split.yaml"
    split_path.parent.mkdir(exist_ok=True, parents=True)
    test_split = {"test": {}}
    if mm1_available:
        test_split["test"]["mm1"] = test_subjects_mm1(data_path)
    if mm2_available:
        test_split["test"]["mm2"] = test_subjects_mm2()
    data_to_file(test_split, split_path)

    print(f"Split file(s) written to '{split_path}'")
