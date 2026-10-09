from copy import deepcopy

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, MultiSampleTrait


class SplitDimensionsd(MapTransform, MultiSampleTrait):
    """Base class for splitting 3D or 4D data into frames or slices."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        ignore_selected_frames: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ignore_selected_frames = ignore_selected_frames

    def split_frames(self, data: dict) -> list[dict]:
        d = deepcopy(data)

        # if only one frame is present, return the data without the frame dimension
        total_nr_frames = d[self.first_valid_key(d)].shape[4]
        if total_nr_frames == 1:
            for key in self.key_iterator(d):
                d[key] = data[key][:, :, :, :, 0].clone()
            return [d]

        # if multiple frames are present, add total_nr_frames to meta_dict and continue
        d["meta_dict"]["total_nr_frames"] = total_nr_frames

        # select frames to include and remove selected_frame_nrs from meta_dict
        frame_nrs = d["meta_dict"].pop("selected_frame_nrs")
        if self.ignore_selected_frames:
            frame_nrs = torch.arange(total_nr_frames)

        frames = []
        for idx, frame_nr in enumerate(frame_nrs):
            d["meta_dict"]["frame_nr"] = frame_nr.item()
            for key in self.key_iterator(d):
                if self.ignore_selected_frames or data[key].shape[4] != len(frame_nrs):
                    d[key] = data[key][:, :, :, :, frame_nr].clone()  # selected frames
                else:
                    d[key] = data[key][:, :, :, :, idx].clone()  # all time frames
            frames.append(deepcopy(d))
        return frames

    def split_slices(self, data: dict) -> list[dict]:
        d = deepcopy(data)

        # if only one slice is present, return the data without the slice dimension
        total_nr_slices = d[self.first_valid_key(d)].shape[3]
        if total_nr_slices == 1:
            for key in self.key_iterator(d):
                d[key] = data[key][:, :, :, 0, ...].clone()
            return [d]

        # if multiple slices are present, add total_nr_slices to meta_dict and continue
        d["meta_dict"]["total_nr_slices"] = total_nr_slices

        slices = []
        for slice_nr in range(total_nr_slices):
            d["meta_dict"]["slice_nr"] = slice_nr
            for key in self.key_iterator(d):
                d[key] = data[key][:, :, :, slice_nr, ...].clone()
            slices.append(deepcopy(d))
        return slices

    def first_valid_key(self, d: dict):
        for key in self.key_iterator(d):
            if key is not None:
                return key
        raise ValueError(f"No valid key found in: {self.keys}")


class DimensionsTo2Dd(SplitDimensionsd):
    """Split 3D or 4D data into 2D slices/images."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        ignore_selected_frames: bool = False,
    ):
        super().__init__(keys, allow_missing_keys, ignore_selected_frames)

    def __call__(self, data):
        frames = self.split_frames(data)
        slices = []
        for frame_data in frames:
            slices.extend(self.split_slices(frame_data))
        return slices


class DimensionsTo3Dd(SplitDimensionsd):
    """Split 3D or 4D data into 3D frames/volumes."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        ignore_selected_frames: bool = False,
    ):
        super().__init__(keys, allow_missing_keys, ignore_selected_frames)

    def __call__(self, data):
        frames = self.split_frames(data)
        return frames
