from pathlib import Path

import numpy as np

from qcardia_data.utils import (
    dict_to_subject_list,
    load_file,
    read_dataset_csv,
    subject_list_to_dict,
)


def split_data_from_config(config: dict):
    """Split the data into train, valid, and test sets based on a config file, refering
        to a split file. Requires a split fiel to specify the subjects in the test set.
        The file can also specify the subjects in the train and valid sets, but if not,
        the development data will be split into train and valid sets based on a seeded
        shuffle.

    Args:
        config (dict): configuration dictionary, specifying paths to the data and a
            split file.

    Raises:
        FileNotFoundError: split file not found.

    Returns:
        dict, dict: nested dictionaries of train, valid, and test subjects, categorized
            by datasubset, and a similar dictionary for the development data (train and
            validation data combined).
    """
    split_file = Path(config["dataset"]["split_file"])
    data_path = Path(config["paths"]["data"])
    reformatted_data_path = data_path / "reformatted_data"
    if not split_file.exists():
        new_split_file = data_path / "subject_splits" / split_file
        if not new_split_file.exists():
            raise FileNotFoundError(f"Cannot find `{split_file}` or `{new_split_file}`")
        split_file = new_split_file

    file_split = load_file(split_file)

    if "train" in file_split and "valid" in file_split:
        print("splitting development data into train/valid based on file")
        development_dict = subject_list_to_dict(
            dict_to_subject_list(file_split["train"])
            + dict_to_subject_list(file_split["valid"])
        )
        return file_split, development_dict

    print("splitting development data into train/valid based on seeded shuffle")
    development_dict, test_dict = {}, {}
    dataset_names = sorted(config["dataset"]["subsets"])
    for dataset_name in dataset_names:
        dataset_subjects = read_dataset_csv(
            reformatted_data_path / f"{dataset_name}.csv"
        )["SubjectID"]

        test_subjects = set(file_split["test"][dataset_name])

        development_subjects = set(dataset_subjects) - set(test_subjects)

        # account for possible overlap between M&Ms and M&Ms-2 -> limit M&Ms-2 subjects
        if dataset_name == "mm2" and "mm1" in dataset_names:
            overlap_subjects = set(get_mm2_overlap_mm1(reformatted_data_path))
            development_subjects = development_subjects - overlap_subjects
            test_subjects = test_subjects - overlap_subjects

        if len(test_subjects) > 0 or len(development_subjects) > 0:
            test_dict[dataset_name] = sorted(test_subjects)
            development_dict[dataset_name] = sorted(development_subjects)

    subjects = sorted(dict_to_subject_list(development_dict))
    np.random.seed(config["dataset"]["valid_split_seed"])
    np.random.shuffle(subjects)
    nr_valid_subjects = round(len(subjects) * config["dataset"]["valid_partition"])
    train_subjects = sorted(subjects[nr_valid_subjects:])
    valid_subjects = sorted(subjects[:nr_valid_subjects])

    split_dict = {
        "train": subject_list_to_dict(train_subjects),
        "valid": subject_list_to_dict(valid_subjects),
        "test": test_dict,
    }

    return split_dict, development_dict


def get_mm2_overlap_mm1(reformatted_data_path: Path):
    """Get the subjects of M&Ms-2 that could also be included in M&Ms.

    Args:
        reformatted_data_path (Path): path to the reformatted data, containing a csv
        file with pathology information for each subject.

    Returns:
        pandas.Series: M&Ms-2 subject IDs that could also be included in M&Ms.
    """
    mm2_csv_df = read_dataset_csv(reformatted_data_path / "mm2.csv")
    return mm2_csv_df["SubjectID"][
        [pathology in ["NOR", "HCM", "LV"] for pathology in mm2_csv_df.Pathology]
    ]
