"""
Contains the high level DataModule class, which makes dataloaders based on a config.

Classes:
- DataModule: A class for providing dataloaders for the training, validation, and test 
    datasets, while handling caching and building transforms based on a config dict.

Example usage for training/validation dataloaders:

config = yaml.load(open(config_path), Loader=yaml.FullLoader)
data = DataModule(config)
data.setup()

train_dataloader = data.train_dataloader()
valid_dataloader = data.valid_dataloader()

"""

from monai.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler

from qcardia_data.pipeline.data_split import split_data_from_config
from qcardia_data.pipeline.dataset_cacher import DatasetCacher
from qcardia_data.pipeline.transforms.compose import build_transform
from qcardia_data.pipeline.utils import build_dataset_paths, build_sampler_weights


class DataModule:
    """A class to providing dataloaders and handle caching based on config.

    Attributes (relevant externally, read only):
    - config (dict): configuration dictionary used for the data module.
    - data_split (dict): dictionary containing the subject split of the data into train,
        valid, and test, seperated by data subset.
    - data_cacher (DatasetCacher): a DatasetCacher object that handles data caching.

    Methods:
    - train_dataloader(): Build a monai DataLoader object for the training dataset.
    - valid_dataloader(): Build a monai DataLoader object for the validation dataset.
    - test_dataloader(): Build a monai DataLoader object for the test dataset, must be
        used separately from the training and validation dataloaders.
    """

    def __init__(self, config: dict):
        # split data into train/valid/test for data subsettype selection
        self.data_split, development_split_dict = split_data_from_config(config)

        # use separate development (train + valid) split for data caching
        cache_split = {"dev": development_split_dict, "test": self.data_split["test"]}
        self.data_cacher = DatasetCacher(config, cache_split)

        self.cached_path_key = "cached_path"
        self.config = config

    def setup(self):
        """Setup the data module by caching the dataset and building the transforms and
        paths of to each individual cached data dict.
        """

        # cache dataset if it doesn't exist yet
        self.data_cacher.cache_dataset(overwrite=False)

        # build transforms with and without augmentation
        self.augmented_transform = build_transform(
            self.config,
            augmentation=True,
            prediction=False,
            cached_path_key=self.cached_path_key,
        )
        self.unaugmented_transform = build_transform(
            self.config,
            augmentation=False,
            prediction=False,
            cached_path_key=self.cached_path_key,
        )

        # build paths to cached data dicts
        print_summary = (
            None
            if self.config["general"]["verbosity"] <= 0
            else self.config["experiment"]["type"]
        )
        self.paths = build_dataset_paths(
            self.data_cacher.cached_dataset_path,
            self.data_split,
            print_summary=print_summary,
        )
        columns_of_interest = self.config["dataset"]["train_weighted_sampling_columns"]
        if (
            isinstance(columns_of_interest, str)
            and columns_of_interest.lower() == "none"
        ):
            self.train_sampler_weights = None
        else:
            self.train_sampler_weights = build_sampler_weights(
                self.data_cacher.cached_dataset_csv_path,
                self.paths["train"],
                columns_of_interest,
            )

    def train_dataloader(self):
        """Build a monai DataLoader object for the training dataset.

        Returns:
            monai.data.DataLoader: A monai DataLoader object for the training dataset.
        """
        return self._dataloader("train")

    def valid_dataloader(self):
        """Build a monai DataLoader object for the validation dataset.

        Returns:
            monai.data.DataLoader: A monai DataLoader object for the validation dataset.
        """
        return self._dataloader("valid")

    def test_dataloader(self):
        """Build a monai DataLoader object for the test dataset.

        Returns:
            monai.data.DataLoader: A monai DataLoader object for the test dataset.
        """
        return self._dataloader("test")

    def _dataloader(self, dataset_type: str):
        """Build a monai DataLoader object for the given type/part of the cached dataset
            according to the config.

        Args:
            dataset_type (str):  One of "train", "valid", and "test", indicating which
                type/part of the dataset to build the dataloader for.

        Returns:
            monai.data.DataLoader: A monai DataLoader object.
        """

        # Build dataset from cached data dicts with appropriate preset transform
        data_dicts = [{self.cached_path_key: path} for path in self.paths[dataset_type]]
        transform = (
            self.augmented_transform
            if self.config["dataloader"][dataset_type]["augmentation"]
            else self.unaugmented_transform
        )
        dataset = Dataset(data=data_dicts, transform=transform)

        # Check for (subject level) weighting of the data
        if self.train_sampler_weights is not None and dataset_type == "train":
            sampler = WeightedRandomSampler(self.train_sampler_weights, len(dataset))
            return DataLoader(
                dataset=dataset,
                batch_size=self.config["dataloader"][dataset_type]["batch_size"],
                drop_last=self.config["dataloader"][dataset_type]["drop_last"],
                num_workers=self.config["general"]["num_workers"],
                pin_memory=True,
                sampler=sampler,
                persistent_workers=self.config["general"]["persistent_workers"],
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.config["dataloader"][dataset_type]["batch_size"],
            drop_last=self.config["dataloader"][dataset_type]["drop_last"],
            shuffle=self.config["dataloader"][dataset_type]["shuffle"],
            num_workers=self.config["general"]["num_workers"],
            pin_memory=True,
            persistent_workers=self.config["general"]["persistent_workers"],
        )
