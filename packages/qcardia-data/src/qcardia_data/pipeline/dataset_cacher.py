import hashlib
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    ToTensord,
)
from tqdm import tqdm

from qcardia_data.pipeline.transforms import (
    BuildImageMetaDatad,
    BuildLabelMetaDatad,
    DimensionsTo2Dd,
    DimensionsTo3Dd,
    Ensure4Dd,
    ProcessIntensityd,
)
from qcardia_data.pipeline.utils import process_key_pairs
from qcardia_data.utils import read_dataset_csv


class DatasetCacher:
    """Caches the reformatted data to a dataset folder with cached data dicts and a csv.
        The csv files contains relevant information of the cached data dicts, and is
        used to detect whether a dataset is already cached and can thus be skipped. Can
        provide a dictionary of where to find all the relevant folders/files to a cached
        dataset.

    Attributes:
        config (dict): configuration dictionary
        data_split_dict (dict): dictionary of data split
        reformatted_data_path (Path): path to reformatted data
        cached_data_path (Path): path to folder of all cached datasets
        cached_dataset_path (Path): path to specific cached dataset folder
        cached_dataset_csv_path (Path): path to cached dataset csv file
    """

    def __init__(self, config: dict, data_split_dict: dict) -> None:
        self.reformatted_data_path = Path(config["paths"]["data"]) / "reformatted_data"
        self.cached_data_path = Path(config["paths"]["data"]) / "cached_data"
        self.data_split_dict = data_split_dict

        self.config = config
        self._generate_dataset_paths()

    def get_paths_dict(self) -> dict:
        """Make a dictionary of paths converted to paths, relevant to the
            reformatted/cached data(set).

        Returns:
            dict: dictionary of path strings
        """
        return {
            "reformatted_data": str(self.reformatted_data_path.resolve()),
            "cached_data": str(self.cached_data_path.resolve()),
            "cached_dataset": str(self.cached_dataset_path.resolve()),
            "cached_dataset_csv": str(self.cached_dataset_csv_path.resolve()),
        }

    def cache_dataset(self, overwrite: bool = False) -> None:
        """Caches the dataset to a dataset folder with cached data dicts and a csv.

        Args:
            overwrite (bool, optional): Whether to recache the dataset if t already
                exists. Defaults to False.
        """
        if not self.cached_dataset_csv_path.exists() or overwrite:
            data_dicts = self.get_reformatted_data_dicts()
            self._cache_data_dicts(data_dicts)

    def _generate_dataset_paths(self) -> None:
        """Generates a unique name for the folder and csv file based on the
        configuration. Cached dataset names:
        [dataset type][dimensionality]-[datasubset name]_[nr datasubset subjects]-
        [image key]=[label key](-meta_only_labels)-[included subjects].

        Multiple datasubsets and image/label keypairs are possible. The presenence
        of meta_only_labels indicates the setting of the same name. The included
        subjects are encoded in a unique hash. Full examples:
        1) dev2D-mm1_285-mm2_85-sa_cine=sa_cine_gt-b5296fe704398a111570e2fafa44ae9a

        2) dev2D-mm1_285-mm2_85-sa_cine=sa_cine_gt-meta_only_labels-
            b5296fe704398a111570e2fafa44ae9a
        """
        # experiment type "train" signifies to cache the development set (train + valid)
        if self.config["experiment"]["type"] == "train":
            subjects_dict = self.data_split_dict["dev"]
            dataset_name = "dev"  # dataset type

        elif self.config["experiment"]["type"] == "test":
            subjects_dict = self.data_split_dict["test"]
            dataset_name = "test"  # dataset type

        dataset_name += self.config["data"]["dimensionality"]  # dimensionality

        hasher = hashlib.md5()
        for datasubset_name in sorted(subjects_dict.keys()):
            # datasubset name + nr datasubset subjects
            dataset_name += f"-{datasubset_name}_{len(subjects_dict[datasubset_name])}"
            hasher.update(datasubset_name.encode())  # update hash with dataset name
            for subject in sorted(subjects_dict[datasubset_name]):
                hasher.update(subject.encode())  # update hash with included subjects

        key_pairs = process_key_pairs(self.config["dataset"]["key_pairs"])
        using_labels = False  # used to ignore meta_only_labels if no labels are used
        for key_pair in key_pairs:
            dataset_name += f"-{key_pair[0]}={key_pair[1]}"  # image key + label key
            if key_pair[1] is not None:
                using_labels = True

        if using_labels and self.config["dataset"]["meta_only_labels"]:
            dataset_name += "-meta_only_labels"  # optional "meta_only_labels"

        # add special mode to dataset name if present
        special_mode = self.config["dataset"]["special_mode"]
        self.special_mode = None if special_mode.lower() == "none" else special_mode
        if self.special_mode is not None:
            dataset_name += f"-{self.special_mode}"

        dataset_name += f"-{hasher.hexdigest()}"  # unique code for included subjects

        # make cached dataset/csv paths within general folder for all cached datasets
        self.cached_dataset_path = self.cached_data_path / dataset_name
        self.cached_dataset_csv_path = self.cached_data_path / f"{dataset_name}.csv"

    def get_reformatted_data_dicts(self) -> list[dict]:
        """Get list of data dictionaries of reformatted data path and meta data, to be
            fed to the cacher.

        Raises:
            FileNotFoundError: Reformatted data file not found.

        Returns:
            List[dict]: List of data dictionaries.
        """
        if self.config["experiment"]["type"] == "train":
            subjects_dict = self.data_split_dict["dev"]
        elif self.config["experiment"]["type"] == "test":
            subjects_dict = self.data_split_dict["test"]

        key_pairs = process_key_pairs(self.config["dataset"]["key_pairs"])

        data_dicts = []
        for datasubset_name in sorted(subjects_dict.keys()):
            csv_df = read_dataset_csv(
                self.reformatted_data_path / f"{datasubset_name}.csv",
            )

            for subject_id in subjects_dict[datasubset_name]:
                # save meta data in a meta dictionary
                subject_path = self.reformatted_data_path / datasubset_name / subject_id
                data_dict = {
                    "meta_dict": {
                        "source": str(subject_path.resolve()),
                        "dataset": datasubset_name,
                        "subject_id": subject_id,
                    },
                }

                # add frame information if available
                if "ED" in csv_df.columns and "ES" in csv_df.columns:
                    data_dict["meta_dict"]["selected_frame_nrs"] = torch.tensor(
                        [
                            csv_df[csv_df["SubjectID"] == subject_id]["ED"].item(),
                            csv_df[csv_df["SubjectID"] == subject_id]["ES"].item(),
                        ],
                    )

                # add paths to nifti files to data dictionary for each key pair
                for key_pair in key_pairs:
                    for key in key_pair:
                        # "None" key is as a placeholder key when a dataset does not
                        # have/should not include an image or label
                        if key is None:
                            continue  # skip placeholder keys
                        data_dict[key] = subject_path / f"{subject_id}_{key}.nii.gz"
                        if not data_dict[key].exists():
                            raise FileNotFoundError(f"'{data_dict[key]}' not found")

                data_dicts.append(data_dict)
        return data_dicts

    def _get_cache_transform(self) -> Compose:
        """Build transform to cache data dicts.

        Raises:
            NotImplementedError: Dimensionality not implemented.

        Returns:
            monai.transforms.compose.Compose: composed transform
        """
        # prepare key collections
        image_keys, label_keys = process_key_pairs(
            self.config["dataset"]["key_pairs"],
            separate_key_pairs=True,
        )
        all_keys = image_keys + label_keys

        meta_only_labels = self.config["dataset"]["meta_only_labels"]
        dimensionality_keys = image_keys if meta_only_labels else all_keys

        # slice to target dimensionality
        dimensionality = self.config["data"]["dimensionality"]
        if dimensionality == "2D":
            dimensions_transform = DimensionsTo2Dd(
                keys=dimensionality_keys,
                allow_missing_keys=True,
                ignore_selected_frames=meta_only_labels,
            )
        elif dimensionality == "3D":
            dimensions_transform = DimensionsTo3Dd(
                keys=dimensionality_keys,
                allow_missing_keys=True,
                ignore_selected_frames=meta_only_labels,
            )
        # elif dimensionality == "2D+T":
        # elif dimensionality == "3D+T":

        else:
            raise NotImplementedError(
                f"dimensionality {dimensionality} not implemented",
            )

        # build transform
        transforms = [
            LoadImaged(keys=all_keys, allow_missing_keys=True, image_only=False),
            EnsureChannelFirstd(
                keys=all_keys,
                allow_missing_keys=True,
                channel_dim="no_channel",
            ),
            ToTensord(keys=all_keys, allow_missing_keys=True),
            # ensure shape/size to (channel, height, width, depth, time)
            Ensure4Dd(keys=all_keys, allow_missing_keys=True),
            # ensure expected intensity range
            ProcessIntensityd(keys=all_keys, allow_missing_keys=True),
            # save subject/file level meta data
            BuildImageMetaDatad(
                keys=image_keys,
                allow_missing_keys=True,
                reset_meta_dict=True,
                pixdim=True,
                intensity=True,
            ),
            BuildLabelMetaDatad(
                keys=label_keys,
                allow_missing_keys=True,
                reset_meta_dict=True,
                pixdim=True,
                gt_center_size=True,
            ),
            dimensions_transform,  # slice to target dimensionality
            # save target dimensionality level meta data
            BuildLabelMetaDatad(
                keys=label_keys,
                allow_missing_keys=True,
                present_classes=True,
            ),
        ]
        if meta_only_labels:
            # remove placeholders keys
            delete_keys = [key for key in label_keys if key is not None]
            if len(delete_keys) > 0:
                transforms.append(DeleteItemsd(keys=label_keys))

        return Compose(transforms)

    def _cache_data_dicts(self, data_dicts: list[dict]) -> None:
        """Caches list of data dictionaries to a dataset folder with a caching
            transform.

        Args:
            data_dicts (List[dict]): list of data dictionaries to cache
        """
        # apply transform to process data into desired format + meta data
        dataset = Dataset(data=data_dicts, transform=self._get_cache_transform())
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.config["general"]["num_workers"],
            collate_fn=collate_fn,
        )  # use MONAI DataLoader to utilize multi-threading
        self.cached_dataset_path.mkdir(parents=True, exist_ok=True)

        data_csv_dict = {}
        columns = ["file_id", "dataset", "SubjectID"]  # start with these columns

        # read csv files of all datasets to get relevant information
        for dataset_name in self.config["dataset"]["subsets"]:
            data_csv_dict[dataset_name] = read_dataset_csv(
                self.reformatted_data_path / f"{dataset_name}.csv",
            )
            columns.extend(
                [
                    column
                    for column in data_csv_dict[dataset_name].columns
                    if column not in columns
                ],
            )

        dataset_df = pd.DataFrame(columns=columns)
        print(f"Caching to `{self.cached_dataset_path.resolve()}`:", flush=True)

        # use tqdm for progress bar
        for subject_data_dicts in tqdm(
            data_loader,
            dynamic_ncols=True,
            mininterval=5.0,
            disable=None,
        ):
            for data_dict in subject_data_dicts:
                # build file (to be cached) specific meta data
                dataset = data_dict["meta_dict"]["dataset"]
                subject_id = data_dict["meta_dict"]["subject_id"]

                data_csv = data_csv_dict[dataset]
                file_name = f"{dataset}-{subject_id}"

                # add slice information to file name
                if "slice_nr" in data_dict["meta_dict"]:
                    slice_nr = data_dict["meta_dict"]["slice_nr"]
                    file_name += f"-{slice_nr:02}"
                else:
                    file_name += "-__"  # all slices included/no depth dimension

                # add frame information to file name and meta dict
                if "frame_nr" in data_dict["meta_dict"]:
                    og_frame_nr = data_dict["meta_dict"]["frame_nr"]
                    total_nr_frames = data_dict["meta_dict"]["total_nr_frames"]
                    ed_frame = data_csv[data_csv["SubjectID"] == subject_id][
                        "ED"
                    ].item()
                    es_frame = data_csv[data_csv["SubjectID"] == subject_id][
                        "ES"
                    ].item()
                    if self.special_mode is not None:
                        # when in special mode, skip frames not of interest
                        if (
                            self.special_mode == "ed_only" and ed_frame != og_frame_nr
                        ) or (
                            self.special_mode == "es_only" and es_frame != og_frame_nr
                        ):
                            continue
                    data_dict["meta_dict"]["is_ed"] = ed_frame == og_frame_nr
                    data_dict["meta_dict"]["is_es"] = es_frame == og_frame_nr

                    frame_nr = (og_frame_nr - ed_frame) % total_nr_frames
                    data_dict["meta_dict"]["frame_nr"] = frame_nr
                    file_name += f"-{frame_nr:02}"
                else:
                    file_name += "-__"  # all frames included/no time dimension

                data_dict["meta_dict"]["file_id"] = file_name  # save file id

                # save data dict as pytorch file
                save_path = self.cached_dataset_path / f"{file_name}.pt"
                torch.save(data_dict, save_path)

                # add original subject data + file level meta data to cached dataset csv
                df_entry = self.build_df_entry(
                    deepcopy(data_dict),
                    data_csv,
                )
                if dataset_df.empty:
                    dataset_df = df_entry.copy()
                else:
                    dataset_df = pd.concat([dataset_df, df_entry], ignore_index=True)

        dataset_df = dataset_df[columns]  # order columns
        # ensure integer data types for relevant columns, when present
        if "total_nr_slices" in dataset_df.columns:
            dataset_df = dataset_df.astype(
                {"slice_nr": "int32", "total_nr_slices": "int32"},
            )
        if "total_nr_frames" in dataset_df.columns:
            dataset_df = dataset_df.astype(
                {"frame_nr": "int32", "total_nr_frames": "int32"},
            )

        # save dataset csv file
        dataset_df.to_csv(self.cached_dataset_csv_path)

    def build_df_entry(
        self,
        data_dict: dict,
        data_csv: pd.DataFrame,
    ) -> pd.DataFrame:
        """Copy original subject level data and add relevant information of file level
            data dict to dataframe entry.

        Args:
            data_dict (dict): file level data dictionary
            data_csv (pd.DataFrame): original dataset dataframe of subject information

        Returns:
            pd.DataFrame: dataframe with relevant information of subject
        """
        df_entry = data_csv.loc[
            data_csv["SubjectID"] == data_dict["meta_dict"]["subject_id"]
        ].copy()

        for key in data_dict["meta_dict"]:
            if key not in ["source", "is_ed", "is_es"]:
                df_entry[key] = data_dict["meta_dict"][key]
        return df_entry


def collate_fn(batch: list[list[dict]]) -> list[dict]:
    """Collates a nested list of dicts into a flat list of dicts

    Output equivalent to:
    # output = []
    # for images in batch:
    #     for image in images:
    #         output.append(image)
    # return output

    Args:
        batch (List[List[dict]]): batch

    Returns:
        List[dict]: List of unnested dicts
    """
    return [image for images in batch for image in images]
