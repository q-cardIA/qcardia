"""
Module for predicting data with a model and summarizing the results.

Classes:
    - DataPredictor: Predict, save and/or summarize or visualize data for a specific
        model.
    - BasePredictor: Abstract base class for model specific predictors, used by the
        DataPredictor.

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.data.utils import decollate_batch
from monai.transforms import BatchInverseTransform
from monai.transforms.utils import allow_missing_keys_mode
from PIL import Image

from qcardia_data.pipeline.data_split import split_data_from_config
from qcardia_data.pipeline.dataset_cacher import DatasetCacher
from qcardia_data.pipeline.transforms.compose import build_transform
from qcardia_data.pipeline.utils import build_dataset_paths


class BasePredictor(ABC):
    """
    Abstract base class for model specific predictors.

    A new predictor class should be created (inheriting from this base class) for each
    model type, overriding the abstract methods to handle forward passes and requiered
    summaries, e.g. metrics, plots, tables, etc.

    Attributes:
        config (dict): config dictionary that contains all hyperparameters/settings
        output_key (str): key for the output of the model
        device (torch.device): device to run the model on
    """

    def __init__(self, config: dict, name: str):
        """
        Args:
            config (dict): config dictionary that contains all hyperparameters/settings
        """
        super().__init__()
        self.config = config
        self.name = name
        self.output_key = "output"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            warn("No GPU available; using CPU", stacklevel=1)

    @abstractmethod
    def forward_model(self, data: torch.Tensor) -> torch.Tensor:
        """Feed input tensor through the model and reutrn the output tensor.

        Used to perform forward passes on the model, can be customized to handle which
        output to return when the models returns multiple outputs.

        Args:
            data (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: output tensor of the model
        """

    @abstractmethod
    def process_summary_3d(
        self,
        subject_data_dict: dict,
        image_key: str,
        label_key: str,
    ) -> None:
        """Process summary of data dictionary of a single subject, containing 3D data.

        Args:
            subject_data_dict (dict): 3D data dictionary of a single subject, including
                images, labels, predictions, and cooresponding meta data. Intended to
                save/store processed data for each 3D data dictionary.
            image_key (str): Dictionary key for the image.
            label_key (str): Dictionary key for the label.
        """

    @abstractmethod
    def process_summary_all(self) -> dict:
        """Process all data from saved processed 3D data dicts to return summary dict of
            the full dataset.

        Returns:
            dict: Summary dictionary of the full dataset, summarizing all data.
        """


class DataPredictor:
    """Predict, save and/or summarize or visualize data for a specific model.

    Attributes:
        config_model (dict): config dictionary that contains all hyperparameters/
            settings of the original model.
        config_data (dict): config dictionary for the data predictor, based on
            config_model, but updated to force 3D dimensionality (used to ensure 3D
            data, independent of model dimensionality), as well as updated data split
            files, num_workers and the option to use the test data.
        model_predictor (BasePredictor): model specific data predictor/processor.
        results_path (Path): parent folder path to save results to.
        cached_path_key (str): key for the cached path in the data dictionary.
        data_transform (monai.transforms.Compose): transform to use for processing data,
            required for inverse transform.
        data (monai.data.DataLoader): dataloader to use for loading data, required for
            inverse transform.
        paths (dict): dictionary of paths to the cached data for each dataset type
            (train, valid, test).
    """

    def __init__(
        self,
        config: dict,
        model_predictor: BasePredictor,
        results_path: Path,
        data_split_file_path: Path | str,
        test=False,
    ):
        self.config_model = deepcopy(config)
        self.model_predictor = model_predictor
        self.cached_path_key = "cached_path"

        self.config_data = deepcopy(config)
        self.config_data["data"]["dimensionality"] = "3D"
        self.config_data["dataset"]["split_file"] = data_split_file_path
        self.config_data["general"]["num_workers"] = torch.get_num_threads()
        if test:
            self.config_data["experiment"]["type"] = "test"

        self.results_path = results_path

        data_split, development_split_dict = split_data_from_config(config)
        cache_split = {"dev": development_split_dict, "test": data_split["test"]}
        data_cacher = DatasetCacher(self.config_data, cache_split)
        data_cacher.cache_dataset(overwrite=False)

        self.paths = build_dataset_paths(
            data_cacher.cached_dataset_path.resolve(),
            data_split,
            print_summary="all",
        )

    def get_dataloader(
        self,
        dataset_type: str,
        limit_subjects: list[str] | int | None = None,
        limit_subject_number_seed: int = 0,
    ) -> DataLoader:
        """Provide dataloader of a specific dataset type, optionally limiting the
            (number of) included subjects.

        Args:
            dataset_type (str):  One of "train", "valid", and "test", indicating which
                part of the dataset to make a dataloader for.
            limit_subjects (List[str] | int, optional): Limit the number of subjects by
                specifying a list of strings or a number of subjects. For the list of
                subejcts, provide a list of string that each represent a single subject
                in the format DATASET-SUBJECTID (eg. "mm2-123"). Defaults to None to
                include all subjects.
            limit_subject_number_seed (int, optional): Seed used to sample a random
                number of subjects when applicable. Defaults to 0.

        Returns:
            data_loader (monai.data.DataLoader): Dataloader for the given dataset type,
                only containing a specific set/number of subjects when specified.
        """
        # make list of data dictionaries, each containing a single subject path
        paths = self.paths[dataset_type]
        if limit_subjects is not None:
            if isinstance(limit_subjects, int):
                # randomly select a number of subjects based on seed
                rng = np.random.default_rng(limit_subject_number_seed)
                paths = rng.choice(paths, limit_subjects, replace=False)
            else:  # select specific subjects
                paths = [path for path in paths if path.stem in limit_subjects]
        data_dicts = [{self.cached_path_key: path} for path in paths]

        # build prediction transform/dataset for 3D data with invertible 2D resampling
        self.data_transform = build_transform(
            self.config_data,
            augmentation=False,
            prediction=True,
            cached_path_key=self.cached_path_key,
        )
        dataset = Dataset(data=data_dicts, transform=self.data_transform)

        # build dataloader to output 3D data of 1 subject at a time
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False,
            num_workers=self.config_data["general"]["num_workers"],
            pin_memory=True,
        )

    def add_output_3d(self, data_dict: dict, image_key: str) -> dict:
        """Adds prediction to data dictionary by feeding the 3D image through the model,
            with the given model predictor. The prediction is then added to the data.
            Currently only works for a 2D model, converting the third spatial dimension
            into a batch size.

        Args:
            data_dict (dict): Original data dictionary with an image
            image_key (str): Dictionary key for the image.

        Raises:
            NotImplementedError: Only works for 3D predictions with a 2D model.

        Returns:
            data_dict (dict): Input data dictionary with the added prediction under the
                output key specified by the model predictor.
        """
        dimensionality = self.config_model["data"]["dimensionality"]
        if dimensionality != "2D":
            raise NotImplementedError(f"dimensionality {dimensionality} not supported")

        # move depth dimension to batch dimension
        model_input = data_dict[image_key].squeeze(0).permute([3, 0, 1, 2])

        # feed through model
        with torch.no_grad():
            model_output = self.model_predictor.forward_model(model_input)

        # inverse transform back to original spacing/size
        processed_output = model_output.permute([1, 2, 3, 0]).unsqueeze(0).cpu()
        batch_inverter = BatchInverseTransform(self.data_transform, self.data)
        with allow_missing_keys_mode(self.data_transform):
            preds = batch_inverter({image_key: processed_output})[0]

        # build data dict with 3D dimensionality
        data_dict[self.model_predictor.output_key] = preds[image_key].unsqueeze(0)
        data_dict_3d = decollate_batch(data_dict)[0]

        # load original image and its meta dict to replace their processed versions
        cached_dict = torch.load(data_dict_3d["meta_dict"]["cached_path"])
        data_dict_3d[image_key] = cached_dict[image_key]
        data_dict_3d[f"{image_key}_meta_dict"] = cached_dict[f"{image_key}_meta_dict"]
        return data_dict_3d

    def summarize_all(
        self,
        dataset_type: str,
        image_key: str,
        label_key: str,
        limit_subjects: list[str] | int | None = None,
        limit_subject_number_seed: int = 0,
    ) -> dict:
        """Summarize given dataset type/part with the model predictor.

        Args:
            dataset_type (str): One of "train", "valid", and "test", indicating which
                part of the dataset to consider in the summary.
            image_key (str): Dictionary key for the image.
            label_key (str): Dictionary key for the label.
            limit_subjects (List[str] | int, optional): Limit the number of subjects by
                specifying a list of strings or a number of subjects. For the list of
                subejcts, provide a list of string that each represent a single subject
                in the format DATASET-SUBJECTID (eg. "mm2-123"). Defaults to None to
                include all subjects.
            limit_subject_number_seed (int, optional): Seed used to sample a random
                number of subjects when applicable. Defaults to 0.

        Returns:
            summary (dict): summary dictionary of the model predictions, specified by
                the model predictor.
        """
        self.data = self.get_dataloader(
            dataset_type,
            limit_subjects,
            limit_subject_number_seed,
        )
        for data_dict in self.data:
            # convert from 3D to model dimensionality
            data_dict_3d = self.add_output_3d(data_dict, image_key)

            # process 3D data dict in preparation of summarization
            self.model_predictor.process_summary_3d(data_dict_3d, image_key, label_key)

        # process all data from processing 3D data dict to return summary dict
        return self.model_predictor.process_summary_all()

    def save_predictions(
        self,
        dataset_type: str,
        file_format: str,
        image_key: str,
        limit_subjects: list[str] | int | None = None,
        limit_subject_number_seed: int = 0,
    ) -> None:
        """Save predictions of given dataset type/part with the model predictor. Not yet
            complete: currently only adds the predictions to the data dictionary, but
            does not save it.

        Args:
            dataset_type (str): One of "train", "valid", and "test", indicating which
                part of the dataset to predict for.
            image_key (str): Dictionary key for the image.
            limit_subjects (List[str] | int, optional): Limit the number of subjects by
                specifying a list of strings or a number of subjects. For the list of
                subejcts, provide a list of string that each represent a single subject
                in the format DATASET-SUBJECTID (eg. "mm2-123"). Defaults to None to
                include all subjects.
            limit_subject_number_seed (int, optional): Seed used to sample a random
                number of subjects when applicable. Defaults to 0.
        """
        results_path = self.results_path / "predictions"
        results_path.mkdir(exist_ok=True, parents=True)

        self.data = self.get_dataloader(
            dataset_type,
            limit_subjects,
            limit_subject_number_seed,
        )
        for data_dict in self.data:
            data_dict_3d = self.add_output_3d(data_dict, image_key)
            prediction = data_dict_3d[self.model_predictor.output_key]
            name = Path(data_dict_3d["meta_dict"]["cached_path"]).stem
            if file_format.lower() in ["np", "npy", ".np", ".npy"]:
                np.save(results_path / f"{name}.npy", prediction)
            elif file_format.lower() in ["pt", ".pt"]:
                torch.save(prediction, results_path / f"{name}.pt")
            else:
                raise NotImplementedError(
                    f"save predictions not yet implemented for format {file_format}",
                )

    def save_example_slices(
        self,
        dataset_type: str,
        image_key: str,
        label_key: str,
        limit_subjects: list[str] | int | None = None,
        limit_subject_number_seed: int = 0,
        sub_folder: str | None = None,
    ) -> None:
        """Saves examples slices of the given dataset type/part and included subjects
            as PNGs in subject level folders.

        Args:
            dataset_type (str): One of "train", "valid", and "test", indicating which
                part of the dataset to save example slices for.
            image_key (str): Dictionary key for the image.
            label_key (str): Dictionary key for the label.
            limit_subjects (List[str] | int, optional): Limit the number of subjects by
                specifying a list of strings or a number of subjects. For the list of
                subejcts, provide a list of string that each represent a single subject
                in the format DATASET-SUBJECTID (eg. "mm2-123"). Defaults to None to
                include all subjects.
            limit_subject_number_seed (int, optional): Seed used to sample a random
                number of subjects when applicable. Defaults to 0.
            sub_folder (str, optional): _description_. Defaults to None.
        """
        results_path = self.results_path / "example_slices"
        if sub_folder is not None:
            results_path = results_path / sub_folder
        results_path.mkdir(exist_ok=True, parents=True)

        self.data = self.get_dataloader(
            dataset_type,
            limit_subjects,
            limit_subject_number_seed,
        )
        for data_dict in self.data:
            data_dict_3d = self.add_output_3d(data_dict, image_key)

            # normalize image intensity based on saved values in the meta dict
            min_intensity = data_dict_3d[f"{image_key}_meta_dict"]["min_intensity"]
            max_intensity = data_dict_3d[f"{image_key}_meta_dict"]["max_intensity"]
            image = (data_dict_3d[image_key] - min_intensity) / (
                max_intensity - min_intensity
            )

            # Get label when label key is available
            if label_key is not None:
                label = data_dict_3d[label_key]

            # Get probabilities from output
            probs = torch.nn.functional.softmax(
                data_dict_3d[self.model_predictor.output_key],
                dim=0,
            )

            # Make folder for subject
            volume_id = Path(data_dict_3d["meta_dict"]["cached_path"]).stem
            slices_path = results_path / f"{self.model_predictor.name}_{volume_id}"
            slices_path.mkdir(exist_ok=True)
            nr_slices = image.shape[3]  # shape: (channels (1), height, width, depth)
            for slice_nr in range(nr_slices):
                # update 3D ID with slice number (e.g. mm2-123-__-00 -> mm2-123-00-00)
                slice_id = volume_id.replace("__", f"{slice_nr:02}")

                # clone slices and save as png using PIL
                image_slice = image[0, :, :, slice_nr].clone()
                pil_img = Image.fromarray((image_slice * 255.0).astype(np.uint8))
                pil_img.save(slices_path / f"{slice_id}_image.png")

                if label_key is not None:
                    label_slice = label[1:, :, :, slice_nr].clone().permute([1, 2, 0])
                    pil_img = Image.fromarray((label_slice * 255.0).astype(np.uint8))
                    pil_img.save(slices_path / f"{slice_id}_label.png")

                probs_slice = probs[1:, :, :, slice_nr].clone().permute([1, 2, 0])
                pil_img = Image.fromarray((probs_slice * 255.0).astype(np.uint8))
                pil_img.save(slices_path / f"{slice_id}_probs.png")
