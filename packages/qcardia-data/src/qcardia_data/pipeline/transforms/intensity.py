from copy import deepcopy

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform


class StandardizeIntensityd(MapTransform):
    """Standardize the intensity of input data to zero mean and unit variance based on
    a saved mean and standard deviation (from original uncropped 4D/3D image), or
    the mean and standard deviation of the input data. Also updates the min and max
    intensity values in the meta_dict accordingly.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        use_saved_mean_std: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.use_saved_mean_std = use_saved_mean_std

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            if self.use_saved_mean_std:
                mean = d[f"{key}_meta_dict"]["mean_intensity"]
                std = d[f"{key}_meta_dict"]["std_intensity"]
            else:
                mean = torch.mean(d[key])
                std = torch.std(d[key])
                d[f"{key}_meta_dict"]["min_intensity"] = torch.min(d[key]).item()
                d[f"{key}_meta_dict"]["max_intensity"] = torch.max(d[key]).item()
            d[key] = (d[key] - mean) / std
            d[f"{key}_meta_dict"]["min_intensity"] = (
                d[f"{key}_meta_dict"]["min_intensity"] - mean
            ) / std
            d[f"{key}_meta_dict"]["max_intensity"] = (
                d[f"{key}_meta_dict"]["max_intensity"] - mean
            ) / std
        return d


class NormalizeIntensityd(MapTransform):
    """Normalize input based on a saved min and max value (from original uncropped 4D/3D
    image), or the min and max value of the input data. Also updates the min and max
    intensity values in the meta_dict accordingly."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        target_min=None,
        target_max=None,
        source_min=None,
        source_max=None,
        use_saved_min_max=False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_min = target_min
        self.target_max = target_max
        self.source_min = source_min
        self.source_max = source_max
        self.use_saved_min_max = use_saved_min_max

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            if self.use_saved_min_max:
                source_min = d[f"{key}_meta_dict"]["min_intensity"]
                source_max = d[f"{key}_meta_dict"]["max_intensity"]
            else:
                source_min = (
                    torch.min(d[key]) if self.source_min is None else self.source_min
                )
                source_max = (
                    torch.max(d[key]) if self.source_max is None else self.source_max
                )
            source_range = source_max - source_min
            d[key] = (d[key] - source_min) / source_range
            d[f"{key}_meta_dict"]["min_intensity"] = (
                d[f"{key}_meta_dict"]["min_intensity"] - source_min
            ) / source_range
            d[f"{key}_meta_dict"]["max_intensity"] = (
                d[f"{key}_meta_dict"]["max_intensity"] - source_min
            ) / source_range
            if self.target_min is not None and self.target_max is not None:
                target_range = self.target_max - self.target_min
                d[key] = d[key] * target_range + self.target_min
                d[f"{key}_meta_dict"]["min_intensity"] = (
                    d[f"{key}_meta_dict"]["min_intensity"] * target_range
                    + self.target_min
                )
                d[f"{key}_meta_dict"]["max_intensity"] = (
                    d[f"{key}_meta_dict"]["max_intensity"] * target_range
                    + self.target_min
                )
        return d


class ClampIntensityd(MapTransform):
    """Clamp the intensity of input data based on a saved min and max value (from
    original uncropped 4D/3D image), or a specified min and max value."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        min_intensity=None,
        max_intensity=None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            min_intensity = (
                d[f"{key}_meta_dict"]["min_intensity"]
                if self.min_intensity is None
                else self.min_intensity
            )
            max_intensity = (
                d[f"{key}_meta_dict"]["max_intensity"]
                if self.max_intensity is None
                else self.max_intensity
            )
            d[key] = torch.clamp(d[key], min_intensity, max_intensity)
        return d


class RandSolarized(RandomizableTransform, MapTransform):
    """Randomly apply Solarization to the input data."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        prob: float = 1.0,
        threshold: float = 0.0,
        use_saved_max: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.threshold = threshold
        self.use_saved_max = use_saved_max

    def __call__(self, data):
        d = deepcopy(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            max_intensity = (
                d[f"{key}_meta_dict"]["max_intensity"]
                if self.use_saved_max
                else torch.max(d[key])
            )
            d[key][d[key] > self.threshold] = (
                max_intensity - d[key][d[key] > self.threshold]
            )
        return d
