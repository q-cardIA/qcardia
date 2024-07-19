from pathlib import Path

import numpy as np
import pandas as pd

from qcardia_data.utils import dict_to_subject_list, read_dataset_csv


def process_key_pairs(key_pairs, separate_key_pairs=False):
    processed_key_pairs = []

    for key_pair in key_pairs:
        processed_key_pair = []
        for key in key_pair:
            if key.lower() in ["none", ""]:
                key = None
            processed_key_pair.append(key)
        processed_key_pairs.append(processed_key_pair)

    if separate_key_pairs:
        image_keys = [key_pair[0] for key_pair in processed_key_pairs]
        label_keys = [key_pair[1] for key_pair in processed_key_pairs]
        return image_keys, label_keys
    return processed_key_pairs


def build_dataset_paths(
    cached_dataset_path: Path,
    subject_data_split: dict,
    print_summary: str | None = "dev",
) -> dict:
    """Build dictionary of paths to cached data files for each dataset split.

    Args:
        cached_dataset_path (Path): Path of cached dataset
        subject_data_split (dict): Dictionary of subject IDs for each dataset split
        print_summary (str | None, optional): for which of the train/valid/test dataset
            splits to print number of files and subjects. Options: "dev", "test", "all".
            Defaults to "train".

    Raises:
        ValueError: file(s) found for subject, but subject not found in data split

    Returns:
        dict: Dictionary of paths to cached data files for each of the train/valid/test
            dataset splits.
    """
    file_paths = sorted(cached_dataset_path.glob("*.pt"))
    subject_paths_dict = {}
    for file_path in file_paths:
        subject_id = "-".join(file_path.name.split("-")[0:2])
        if subject_id not in subject_paths_dict:
            subject_paths_dict[subject_id] = []
        subject_paths_dict[subject_id].append(file_path)

    # convert lists per datasubset to lists of all subject IDs
    train_subjects = dict_to_subject_list(subject_data_split["train"])
    valid_subjects = dict_to_subject_list(subject_data_split["valid"])
    test_subjects = dict_to_subject_list(subject_data_split["test"])

    # build paths for each dataset split
    subjects = sorted(subject_paths_dict.keys())
    paths = {"train": [], "valid": [], "test": []}
    for subject in subjects:
        if subject in train_subjects:
            paths["train"].extend(subject_paths_dict[subject])
        elif subject in valid_subjects:
            paths["valid"].extend(subject_paths_dict[subject])
        elif subject in test_subjects:
            paths["test"].extend(subject_paths_dict[subject])
        else:  # make sure no subjects are omitted unintentionally
            raise ValueError(f"subject {subject} not in data split")

    # provide number of found files and subjects for cached dataset split(s)
    if print_summary is not None:
        print(f"dataset path: '{cached_dataset_path}'")
        if print_summary == "test":
            print_dataset_summary("test", len(paths["test"]), len(test_subjects))
        else:
            print_dataset_summary("train", len(paths["train"]), len(train_subjects))
            print_dataset_summary("valid", len(paths["valid"]), len(valid_subjects))
            if print_summary == "all":
                print_dataset_summary("test", len(paths["test"]), len(test_subjects))

    return paths


def build_sampler_weights(
    cached_dataset_csv_path: Path,
    data_subset_paths: list[str],
    columns_of_interest: list[str],
) -> np.ndarray:
    """Build a vector of weights for weighted sampling of the dataset. Multiple columns
        of interest can be used to build the weights vector, by weighting it based on
        each unique combination of values in the columns of interest, so each
        combination has an equal weight.

    Args:
        cached_dataset_csv_path (Path): path to cached datset csv file
        data_subset_paths (List[str]): paths to the cached dataset files
        columns_of_interest (List[str]): list of columns in the cached datset csv file
            to use for weighted sampling

    Returns:
        np.ndarray: weights to equalize the number of samples per class
    """
    cached_dataset_df = read_dataset_csv(cached_dataset_csv_path)
    assert all(
        column in cached_dataset_df.columns for column in columns_of_interest
    ), f"columns of interest {columns_of_interest} not in dataframe columns {cached_dataset_df.columns}"

    temp_list = []
    for data_subset_path in data_subset_paths:
        temp_list.append(
            cached_dataset_df[
                cached_dataset_df["file_id"] == Path(data_subset_path).stem
            ][columns_of_interest],
        )
    columns_of_interest_df = pd.concat(temp_list, ignore_index=True)
    # Calculate the weights
    weights_df = (
        1.0
        / (
            columns_of_interest_df.groupby(columns_of_interest, dropna=False).size()
            / len(columns_of_interest_df)
        )
    ).reset_index(name="weights")

    # Merge the weights with the dataframe so each subject has its own weight
    merged_df = columns_of_interest_df.merge(
        weights_df,
        on=columns_of_interest,
        how="left",
    )
    return merged_df["weights"].to_numpy()


def print_dataset_summary(name: str, nr_files: int, nr_subjects: int) -> None:
    """print number of cached files and subjects, if any

    Args:
        name (str): name of the dataset split
        nr_files (int): number of files found in the dataset split
        nr_subjects (int): number of subjects for the dataset split
    """
    if nr_files <= 0:
        print(f"{name}: no cached files found ({nr_subjects} subjects in set)")
    else:
        print(f"{name}: found {nr_files} files from {nr_subjects} subjects")
