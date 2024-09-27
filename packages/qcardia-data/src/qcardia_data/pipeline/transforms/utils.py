from copy import deepcopy

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, MultiSampleTrait, RandomizableTransform
from smallestenclosingcircle import make_circle


class Ensure4Dd(MapTransform):
    """Add dimensions of size 1 to the input data to make sure it is 4D (in spatial /
    temporal dimensions).

    Assumes the input data contains a channel dimension first, which is not counted in
    the number of dimensions.
    Example shapes before and after transformation:
        [1, 256, 256]           ->  [1, 256, 256, 1, 1]
        [1, 256, 256, 12]       ->  [1, 256, 256, 12, 1]
        [1, 256, 256, 12, 25]   ->  [1, 256, 256, 12, 25]
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            while len(d[key].shape) < 5:
                d[key] = d[key].unsqueeze(-1)
        return d


class ProcessIntensityd(MapTransform):
    """Process the intensity of input data to be more consistent/as expected.

    For now only setting the lowest value of the image to 0.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            d[key] -= torch.min(d[key])
        return d


class BuildImageMetaDatad(MapTransform):
    """Add meta data of the image to the image meta dict.

    The meta data includes the original pixel dimensions, and image min, max, mean, and
    standard deviation of the image intensities.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        reset_meta_dict: bool = False,
        pixdim: bool = False,
        intensity: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.reset_meta_dict = reset_meta_dict
        self.pixdim = pixdim
        self.intensity = intensity

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            if self.reset_meta_dict:
                d[f"{key}_meta_dict"] = {}

            if self.pixdim:
                d[f"{key}_meta_dict"]["pixdim"] = torch.tensor(
                    data[f"{key}_meta_dict"]["pixdim"][1:4]
                )

            if self.intensity:
                d[f"{key}_meta_dict"]["mean_intensity"] = torch.mean(d[key]).item()
                d[f"{key}_meta_dict"]["std_intensity"] = torch.std(d[key]).item()
                d[f"{key}_meta_dict"]["min_intensity"] = torch.min(d[key]).item()
                d[f"{key}_meta_dict"]["max_intensity"] = torch.max(d[key]).item()

        return d


class BuildLabelMetaDatad(MapTransform):
    """Add meta data of the ground truth label to the label meta dict.

    The meta data includes the original pixel dimensions, center and size of the ground
    truth (gt) / label, as well as a string of the present classes in the label.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        reset_meta_dict: bool = False,
        pixdim: bool = False,
        gt_center_size: bool = False,
        present_classes: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.reset_meta_dict = reset_meta_dict
        self.pixdim = pixdim
        self.gt_center_size = gt_center_size
        self.present_classes = present_classes

    def __call__(self, data):
        d = deepcopy(data)
        for key in self.key_iterator(d):
            if self.reset_meta_dict:
                d[f"{key}_meta_dict"] = {}

            if self.pixdim:
                d[f"{key}_meta_dict"]["pixdim"] = torch.tensor(
                    data[f"{key}_meta_dict"]["pixdim"][1:4]
                )

            if self.gt_center_size:
                points = torch.nonzero(torch.sum(d[key], dim=(0, 3, 4)))
                center_coor_0, center_coor_1, radius = make_circle(points)
                center_coors = torch.tensor([center_coor_0, center_coor_1])
                d[f"{key}_meta_dict"]["gt_center"] = center_coors
                d[f"{key}_meta_dict"]["gt_size"] = radius * 2.0

            if self.present_classes:
                # quick fix for varying size batch collation -> turn into string
                present_classes_str = "_".join(
                    map(str, torch.unique(d[key].long()).tolist())
                )
                d[f"{key}_meta_dict"]["present_classes"] = present_classes_str

        return d


class LoadCachedDatad(MapTransform):
    """Load data cached by a DatasetCacher from the file and convert it into a data
    dictionary."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        ignored_keys=None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ignored_keys = ignored_keys

    def __call__(self, data):
        data_dict = {}
        for cached_key in self.key_iterator(data):
            cached_data = torch.load(data[cached_key], weights_only=False)
            for key in cached_data.keys():
                if self.ignored_keys is None or key not in self.ignored_keys:
                    data_dict[key] = cached_data[key]
            if "meta_dict" not in data_dict:
                data_dict["meta_dict"] = {}
            data_dict["meta_dict"][cached_key] = str(data[cached_key])
        return data_dict


class CopySamplesd(MapTransform, MultiSampleTrait):
    """Add copies of the data dict."""

    def __init__(self, keys: KeysCollection, nr_sample_copies=1) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.nr_sample_copies = nr_sample_copies

    def __call__(self, data):
        data_dicts = [data]
        for i in range(self.nr_sample_copies):
            data_dicts.append(deepcopy(data))
        return data_dicts


class RandChangeSeedd(RandomizableTransform, MapTransform):
    """Adds additional random operation (that cannot do anything) in the transform
    pipeline to change random state for other transforms.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        prob: float = 1.0,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def __call__(self, data):
        self.randomize(None)
        return data
