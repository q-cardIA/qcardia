import json
from pathlib import Path

import pandas as pd
import torch
import yaml


def sample_from_csv_by_group(csv_path, sample_nr, group_key, output_key, seed=0):
    df = pd.read_csv(csv_path, index_col=0, dtype={group_key: str, output_key: str})
    return sorted(
        df.groupby(group_key, group_keys=False, dropna=False).apply(
            lambda x: x.sample(sample_nr, random_state=seed)
        )[output_key]
    )


def print_dict(d, prepend="", max_len=256):
    key_max_len = min(max([len(key) for key in d.keys()]) + len(prepend) + 2, max_len)
    value_max_len = min(max([len(str(d[key])) for key in d.keys()]) + 2, max_len)
    for key in d.keys():
        if isinstance(d[key], torch.Tensor) and torch.numel(d[key]) >= 10:
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
            print(f"{prepend}dict: {key}")
            print_dict(d[key], prepend=prepend + "  ", max_len=max_len)
        else:
            print(
                f"{prepend + key + ':':{key_max_len}}"
                + f"{str(d[key]):{value_max_len}}{type(d[key])}"
            )


def batch_to_data_dicts(batch):
    data_dicts = []
    data_dict = {}
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch_size = batch[key].shape[0]
            break
    for i in range(batch_size):
        for key in batch.keys():
            if isinstance(batch[key], dict):
                data_dict[key] = {}
                for sub_key in batch[key].keys():
                    data_dict[key][sub_key] = batch[key][sub_key][i]
            else:
                data_dict[key] = batch[key][i]
        data_dicts.append(dict(data_dict))
    return data_dicts


def dict_to_subject_list(dataset_dict):
    subject_list = []
    for dataset in dataset_dict:
        subject_list.extend(
            [f"{dataset}-{subject}" for subject in dataset_dict[dataset]]
        )
    return sorted(subject_list)


def subject_list_to_dict(subject_list):
    subject_dict = {}
    for subject in subject_list:
        split_list = subject.split("-")
        dataset_name = split_list[0]
        if dataset_name not in subject_dict:
            subject_dict[dataset_name] = []
        subject_dict[dataset_name].append("-".join(split_list[1:]))
    return subject_dict


def data_to_file(data, file_path: Path):
    if file_path.suffix == ".yaml":
        with open(file_path, "w") as f:
            f.write(yaml.dump(data, default_style='"'))
    elif file_path.suffix == ".json":
        with open(file_path, "w") as f:
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
