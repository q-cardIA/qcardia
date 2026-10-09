from copy import deepcopy

import torch
from monai.config import KeysCollection
from monai.transforms import InvertibleTransform, MapTransform, Randomizable
from monai.utils import TraceKeys
from torch.nn import functional as F

from .transformation_matrices import T_2D_rotate, T_2D_scale, T_2D_translate


class RandResample2Dd(Randomizable, MapTransform):
    """Resample 2D data (image, label, etc.) to the target pixel spacing and size, with
    the option apply random translations, scalings, flips or rotations to the resampling
    grid before resampling, based on given application probabilities and value ranges.
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_meta_keys,
        target_pixdim,
        target_size,
        grid_sample_modes,
        allow_missing_keys: bool = False,
        translation_prob=0.0,
        translation_range=(-20, 20),
        rotation_prob=0.0,
        rotation_range=(-180, 180),
        scale_prob=0.0,
        scale_range=(0.6, 1.5),
        flip_prob=0.0,
        linked_scale=True,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.label_meta_keys = label_meta_keys
        self.target_pixdim = torch.tensor(target_pixdim)
        self.target_size = torch.tensor(target_size)
        self.real_target_size = self.target_pixdim * self.target_size
        self.grid_size = (1, 1, self.target_size[0], self.target_size[1])
        self.grid_sample_modes = grid_sample_modes
        self.translation_prob = translation_prob
        self.translation_range = translation_range
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        if not hasattr(flip_prob, "__len__"):
            self.flip_prob = (flip_prob, flip_prob)
        else:
            self.flip_prob = flip_prob
        self.linked_scale = linked_scale

    def probability_decision(self, p):
        return self.R.rand(*torch.tensor(p).shape) < p

    def randomize(self, data):
        # rotation
        if self.probability_decision(self.rotation_prob):
            self.rotation_angle = self.R.uniform(*self.rotation_range)
        else:
            self.rotation_angle = 0
        self.rotation_angle = torch.tensor(self.rotation_angle, dtype=torch.float32)

        # scaling
        if self.probability_decision(self.scale_prob):
            if self.linked_scale:
                scale = self.R.uniform(*self.scale_range)
                self.scale_factor = [scale, scale]
            else:
                self.scale_factor = self.R.uniform(*self.translation_range, size=2)
        else:
            self.scale_factor = [1, 1]
        self.scale_factor = torch.tensor(self.scale_factor, dtype=torch.float32)

        # flipping
        flip_bools = torch.tensor(
            self.probability_decision(self.flip_prob),
            dtype=torch.float32,
        )
        self.scale_factor *= (flip_bools - 0.5) * -2.0

        # translation
        if self.probability_decision(self.translation_prob):
            self.translation = self.R.uniform(*self.translation_range, size=2)
        else:
            self.translation = [0, 0]
        self.translation = torch.tensor(self.translation, dtype=torch.float32)

        # T_translate simplified and before (order of multiplication) scale
        # T_translate after (order of multiplication) scale:
        # self.translation * self.target_pixdim / real_source_size * 2 * self.scale_factor
        self.translation = self.translation / self.target_size * 2

    def __call__(self, data):
        self.randomize(data)

        d = deepcopy(data)

        # Transformations in order of multiplication:
        # - translate to center (optional)
        # - scale to compensate for aspect ratio [grid size scale]
        # - rotate [degrees]
        # - scale
        # - translate [pixels in target image]

        for key, label_meta_key, grid_sample_mode in self.key_iterator(
            d, self.label_meta_keys, self.grid_sample_modes
        ):
            source_shape = torch.tensor(d[key].shape[1:], dtype=torch.float32)
            real_source_size = d[f"{key}_meta_dict"]["pixdim"][:2] * source_shape
            dimension_scale_factor = self.real_target_size / real_source_size
            T = (
                (
                    torch.eye(3)
                    if label_meta_key is None
                    else T_2D_translate(
                        d[f"{label_meta_key}_meta_dict"]["gt_center"]
                        / (source_shape - 1.0)
                        * 2.0
                        - 1.0
                    )
                )
                @ T_2D_scale(dimension_scale_factor)
                @ T_2D_rotate(self.rotation_angle)
                @ T_2D_scale(self.scale_factor)
                @ T_2D_translate(self.translation)
            )

            grid = F.affine_grid(
                theta=T[:-1, :].unsqueeze(0),
                size=self.grid_size,
                align_corners=False,
            )
            d[key] = F.grid_sample(
                d[key].unsqueeze(0),
                grid,
                align_corners=False,
                mode=grid_sample_mode,
                padding_mode="zeros",
            ).squeeze(0)
        return d


class PredictionResample3Dd(MapTransform, InvertibleTransform):
    """Resample 3D data to the target pixel spacing and size in 2D (looping over 3rd
    dimension), while saving required information to reverse the resampling back to
    convert the (processed) image back to the original spacing and size.
    """

    def __init__(
        self,
        keys: KeysCollection,
        target_pixdim,
        target_size,
        grid_sample_modes,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_pixdim = torch.tensor(target_pixdim)
        self.target_size = torch.tensor(target_size)
        self.grid_sample_modes = grid_sample_modes
        self.real_target_size = self.target_pixdim * self.target_size
        self.grid_size = [1, 1, self.target_size[0], self.target_size[1]]

    def __call__(self, data):
        d = deepcopy(data)
        for key, grid_sample_mode in self.key_iterator(d, self.grid_sample_modes):
            source_shape = torch.tensor(d[key].shape[1:3], dtype=torch.float32)
            real_source_size = d[f"{key}_meta_dict"]["pixdim"][:2] * source_shape
            dimension_scale_factor = self.real_target_size / real_source_size
            T = T_2D_scale(dimension_scale_factor)
            d[key] = d[key].permute([3, 0, 1, 2])
            extra_info = {"transform": T, "source_shape": d[key].shape}
            nr_slices = d[key].shape[0]
            self.grid_size[0] = nr_slices
            grid = F.affine_grid(
                theta=torch.repeat_interleave(T[:-1, :].unsqueeze(0), nr_slices, dim=0),
                size=self.grid_size,
                align_corners=False,
            )
            d[key] = F.grid_sample(
                d[key],
                grid,
                align_corners=False,
                mode=grid_sample_mode,
                padding_mode="zeros",
            ).permute([1, 2, 3, 0])
            self.push_transform(d, key, extra_info=extra_info)
        return d

    def inverse(self, data):
        d = deepcopy(data)
        for key, grid_sample_mode in self.key_iterator(d, self.grid_sample_modes):
            info = self.pop_transform(d[key])
            T = torch.inverse(info[TraceKeys.EXTRA_INFO]["transform"])
            nr_slices = d[key].shape[3]
            grid = F.affine_grid(
                theta=torch.repeat_interleave(T[:-1, :].unsqueeze(0), nr_slices, dim=0),
                size=info[TraceKeys.EXTRA_INFO]["source_shape"],
                align_corners=False,
            )
            d[key] = F.grid_sample(
                torch.permute(d[key], [3, 0, 1, 2]),
                grid,
                align_corners=False,
                mode=grid_sample_mode,
                padding_mode="border",
            ).permute([1, 2, 3, 0])
        return d
