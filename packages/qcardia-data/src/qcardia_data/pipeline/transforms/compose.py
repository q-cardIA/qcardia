from monai.transforms import (
    AsDiscreted,
    Compose,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from qcardia_data.pipeline.transforms import (
    BuildImageMetaDatad,
    ClampIntensityd,
    CopySamplesd,
    LoadCachedDatad,
    NormalizeIntensityd,
    PredictionResample3Dd,
    RandChangeSeedd,
    RandResample2Dd,
    RandSolarized,
    StandardizeIntensityd,
)
from qcardia_data.pipeline.utils import process_key_pairs


def build_transform(
    config: dict, augmentation: bool, prediction: bool, cached_path_key: str
):
    """Build composed transform to apply all specified transforms in the intended order.

    Has different modes for training and prediction, and can disable all data
    augmentation.

    Args:
        config (config): configuration dictionary, specifying the data processing
        augmentation (bool): whether to apply augmentation
        prediction (bool): whether to use invertible 3D resampling for predictions
        cached_path_key (str): key name of the cached path in the data dictionary

    Raises:
        NotImplementedError: not all dimensionality options are implemented, for now
            only 2D is possible for default training behaviour.

    Returns:
        monai.transforms.compose.Compose: composed transform to apply all specified
            transforms in order
    """
    image_keys, label_keys = process_key_pairs(
        config["dataset"]["key_pairs"],
        separate_key_pairs=True,
    )
    all_keys = image_keys + label_keys
    transform_list = [
        LoadCachedDatad(keys=cached_path_key, ignored_keys=None),
    ]

    # add additional random operation (that cannot do anything) in the data pipeline to
    # change random state before the actual data augmentation, so multiple synced
    # dataloaders can output the same images with different augmentations
    if "solarization" in config["data"]["augmentation"]:
        transform_list.append(
            RandChangeSeedd(keys=image_keys, allow_missing_keys=True, prob=1.0),
        )

    if config["data"]["intensity"]["reference_level"] == "image":
        transform_list.append(BuildImageMetaDatad(keys=image_keys, intensity=True))

    if config["data"]["nr_sample_copies"] > 0:
        transform_list.append(
            CopySamplesd(
                keys=all_keys,
                nr_sample_copies=config["data"]["nr_sample_copies"],
            ),
        )

    dimensionality = config["data"]["dimensionality"]
    grid_sample_modes = (
        [config["data"]["image_grid_sample_mode"]] * len(image_keys)
    ) + ([config["data"]["label_grid_sample_mode"]] * len(label_keys))
    if prediction:
        transform_list.append(
            PredictionResample3Dd(
                keys=image_keys,
                allow_missing_keys=True,
                target_pixdim=config["data"]["target_pixdim"],
                target_size=config["data"]["target_size"],
                grid_sample_modes=[config["data"]["image_grid_sample_mode"]]
                * len(image_keys),
            ),
        )
    else:
        if dimensionality == "2D":
            resample_transform = RandResample2Dd
        # elif dimensionality == "3D":
        #     resample_transform = RandResample3Dd
        # elif dimensionality == "2D+T":
        # elif dimensionality == "3D+T":
        else:
            raise NotImplementedError(
                f"resampling not implemented for dimensionality {dimensionality}",
            )

        grid_sample_modes = (
            [config["data"]["image_grid_sample_mode"]] * len(image_keys)
        ) + ([config["data"]["label_grid_sample_mode"]] * len(label_keys))
        if augmentation:
            augmentation_dict = config["data"]["augmentation"]
            if config["data"]["translate_to_center"]:
                label_meta_keys = label_keys * 2
            else:
                label_meta_keys = [None] * len(all_keys)
            transform_list.append(
                resample_transform(
                    keys=all_keys,
                    allow_missing_keys=True,
                    label_meta_keys=label_meta_keys,
                    target_pixdim=config["data"]["target_pixdim"],
                    target_size=config["data"]["target_size"],
                    grid_sample_modes=grid_sample_modes,
                    translation_prob=augmentation_dict["translation"]["prob"],
                    translation_range=augmentation_dict["translation"]["range"],
                    rotation_prob=augmentation_dict["rotation"]["prob"],
                    rotation_range=augmentation_dict["rotation"]["range"],
                    scale_prob=augmentation_dict["scale"]["prob"],
                    scale_range=augmentation_dict["scale"]["range"],
                    linked_scale=augmentation_dict["scale"]["linked"],
                    flip_prob=augmentation_dict["flip_prob"],
                ),
            )
        else:
            transform_list.append(
                resample_transform(
                    keys=all_keys,
                    allow_missing_keys=True,
                    label_meta_keys=[None] * len(all_keys),
                    target_pixdim=config["data"]["target_pixdim"],
                    target_size=config["data"]["target_size"],
                    grid_sample_modes=grid_sample_modes,
                    translation_prob=0.0,
                    rotation_prob=0.0,
                    scale_prob=0.0,
                    flip_prob=0.0,
                ),
            )

    if config["data"]["intensity"]["normalization_mode"] == "standardize":
        use_saved_mean_std = config["data"]["intensity"]["reference_level"] != "current"
        transform_list.append(
            StandardizeIntensityd(
                keys=image_keys,
                use_saved_mean_std=use_saved_mean_std,
            ),
        )
    elif config["data"]["intensity"]["normalization_mode"] == "normalize":
        normalize_dict = config["data"]["intensity"]["normalize"]
        use_saved_min_max = config["data"]["intensity"]["reference_level"] != "current"
        transform_list.append(
            NormalizeIntensityd(
                keys=image_keys,
                target_min=normalize_dict["target_min"],
                target_max=normalize_dict["target_max"],
                source_min=normalize_dict["source_min"],
                source_max=normalize_dict["source_max"],
                use_saved_min_max=use_saved_min_max,
            ),
        )

    if augmentation:
        transform_list.extend(
            [
                RandGaussianNoised(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["gaussian_noise"]["prob"],
                    mean=augmentation_dict["gaussian_noise"]["mean"],
                    std=augmentation_dict["gaussian_noise"]["std"],
                ),
                RandGaussianSmoothd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["gaussian_smooth"]["prob"],
                    sigma_x=augmentation_dict["gaussian_smooth"]["sigma_x"],
                    sigma_y=augmentation_dict["gaussian_smooth"]["sigma_y"],
                ),
                RandScaleIntensityd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["scale_intensity"]["prob"],
                    factors=augmentation_dict["scale_intensity"]["factors"],
                ),
                RandShiftIntensityd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["shift_intensity"]["prob"],
                    offsets=augmentation_dict["shift_intensity"]["offsets"],
                ),
                RandAdjustContrastd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["adjust_contrast"]["prob"],
                    gamma=augmentation_dict["adjust_contrast"]["gamma"],
                ),
            ],
        )
        if "solarization" in augmentation_dict:
            use_saved_max = config["data"]["intensity"]["reference_level"] != "current"
            transform_list.append(
                RandSolarized(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=augmentation_dict["solarization"]["prob"],
                    threshold=augmentation_dict["solarization"]["threshold"],
                    use_saved_max=use_saved_max,
                ),
            )

    if config["data"]["intensity"]["clamp"]["active"]:
        transform_list.append(
            ClampIntensityd(
                keys=image_keys,
                min_intensity=config["data"]["intensity"]["clamp"]["min_intensity"],
                max_intensity=config["data"]["intensity"]["clamp"]["max_intensity"],
            ),
        )

    if config["data"]["to_one_hot"]["active"]:
        transform_list.append(
            AsDiscreted(
                keys=label_keys,
                allow_missing_keys=True,
                to_onehot=config["data"]["to_one_hot"]["nr_classes"],
            ),
        )
    return Compose(transform_list)
