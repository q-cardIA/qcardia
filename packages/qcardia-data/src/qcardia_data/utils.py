import json
from pathlib import Path

import pandas as pd
import torch
import yaml


def sample_from_csv_by_group(
    csv_path: Path, sample_nr: int, group_key: str, output_key: str, seed: int = 0
) -> list[str]:
    """Sample output key from csv file by group key

    Args:
        csv_path (Path): path to csv file to load as pandas dataframe
        sample_nr (int): number of samples to take from each group
        group_key (str): key to group by
        output_key (str): key to output
        seed (int, optional): seed for random sampling. Defaults to 0.

    Returns:
        list[str]: sorted list of outputs
    """
    df = pd.read_csv(csv_path, index_col=0, dtype={group_key: str, output_key: str})
    return sorted(
        df.groupby(group_key, group_keys=False, dropna=False).apply(
            lambda x: x.sample(sample_nr, random_state=seed)
        )[output_key]
    )


def print_dict(d: dict, prepend: str = "", max_len: int = 256) -> None:
    """recursively print dictionary with formatting.

    Args:
        d (dict): dictionary to print
        prepend (str, optional): string to prepend to each line. Defaults to "".
        max_len (int, optional): maximum length of each line. Defaults to 256.
    """
    key_max_len = min(max([len(key) for key in d.keys()]) + len(prepend) + 2, max_len)
    value_max_len = min(max([len(str(d[key])) for key in d.keys()]) + 2, max_len)
    for key in d.keys():
        if isinstance(d[key], torch.Tensor) and torch.numel(d[key]) >= 10:
            # print tensor properties for large tensors
            print(f"{prepend + key + ' - tensor properties:'}")
            properties_dict = {
                "shape": d[key].shape,
                "min": torch.min(d[key].float()).item(),
                "max": torch.max(d[key].float()).item(),
                "mean": torch.mean(d[key].float()).item(),
                "std": torch.std(d[key].float()).item(),
            }
            print_dict(properties_dict, prepend=prepend + "  ", max_len=max_len)
        elif isinstance(d[key], dict):
            # recursively print sub-dictionaries
            print(f"{prepend}dict: {key}")
            print_dict(d[key], prepend=prepend + "  ", max_len=max_len)
        else:
            print(
                f"{prepend + key + ':':{key_max_len}}"
                + f"{str(d[key]):{value_max_len}}{type(d[key])}"
            )


def dict_to_subject_list(dataset_dict: dict) -> list[str]:
    """Convert dictionary of datasets with lists of subjects to list of subjects

    Inverse of `subject_list_to_dict` function.

    Args:
        dataset_dict (dict): dictionary of datasets with lists of subjects

    Returns:
        list[str]: list of subjects in the format `dataset-subject`
    """
    subject_list = []
    for dataset in dataset_dict:
        subject_list.extend(
            [f"{dataset}-{subject}" for subject in dataset_dict[dataset]]
        )
    return sorted(subject_list)


def subject_list_to_dict(subject_list: list[str]) -> dict:
    """Convert list of subjects to dictionary of datasets with lists of subjects

    Inverse of `dict_to_subject_list` function.

    Args:
        subject_list (list[str]): list of subjects in the format `dataset-subject`

    Returns:
        dict: dictionary of datasets with lists of subjects
    """
    subject_dict = {}
    for subject in subject_list:
        split_list = subject.split("-")
        dataset_name = split_list[0]
        if dataset_name not in subject_dict:
            subject_dict[dataset_name] = []
        subject_dict[dataset_name].append("-".join(split_list[1:]))
    return subject_dict


def data_to_file(data: dict, file_path: Path | str) -> None:
    """Save data to file in standardized yaml or json format

    Args:
        data (dict): data dictionary to save
        file_path (Path | str): path to save file to, including file extension
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.suffix == ".yaml":
        with file_path.open("w") as f:
            f.write(yaml.dump(data, default_style='"'))
    elif file_path.suffix == ".json":
        with file_path.open("w") as f:
            f.write(json.dumps(data, indent=4))


def load_file(file_path: Path):
    suffix = file_path.resolve().suffix
    if suffix == ".yaml":
        with open(file_path) as f:
            file_contents = yaml.load(f, Loader=yaml.FullLoader)
        return file_contents
    else:
        raise NotImplementedError(f"`{suffix}` not supported")


def read_dataset_csv(csv_path):
    return pd.read_csv(csv_path, index_col=0, converters={1: str})
