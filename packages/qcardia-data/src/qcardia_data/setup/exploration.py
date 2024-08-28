from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from qcardia_data.pipeline.data_split import split_data_from_config


def explore_development_data_from_config(
    config_path: Path, return_results: bool = False
) -> None:
    """Explore development data from a configuration file.

    Provides statistics for image array dimensions and pixel dimensions, and uses these
    to calculate image size in mm, as well as calculate image array size for consistent
    pixel dimensions for all images. These values can be used to choose appropriate
    image sizes and pixel dimensions.

    Args:
        config_path (Path): path to the configuration file
        return_results (bool): whether to return the results instead of printing them
    """
    config = yaml.load(config_path.open(), Loader=yaml.FullLoader)
    _, development_dict = split_data_from_config(config)

    cine_paths = []
    reformatted_path = Path(config["paths"]["data"]) / "reformatted_data"
    print(f"dataset(s): {sorted(development_dict.keys())}")
    for dataset_name in development_dict:
        for subject_id in development_dict[dataset_name]:
            cine_paths.append(
                reformatted_path
                / dataset_name
                / subject_id
                / f"{subject_id}_sa_cine.nii.gz"
            )

    header_dict = {"dim": [], "pixdim": []}
    for cine_path in cine_paths:
        cine_header = nib.load(cine_path).header
        header_dict["dim"].append(cine_header["dim"][1:5])
        header_dict["pixdim"].append(cine_header["pixdim"][1:4])

    split_dict = {}
    for dict_key in header_dict:
        header_dict[dict_key] = np.array(header_dict[dict_key])
        for dim in range(header_dict[dict_key].shape[1]):
            split_dict[f"{dict_key}_{dim}"] = header_dict[dict_key][:, dim]

    median_pixdims = [np.median(split_dict[f"pixdim_{dim}"]) for dim in range(3)]

    for dim in range(3):
        # units: dim_size [mm] = dim [pixel] * pixdim [mm / pixel]
        applied_size = split_dict[f"dim_{dim}"] * split_dict[f"pixdim_{dim}"]
        split_dict[f"applied_size_{dim}"] = applied_size

    for dim in range(3):
        # units: applied_dim [pixel] = dim_size [mm] / median_pixdim [mm / pixel]
        applied_dim = split_dict[f"applied_size_{dim}"] / median_pixdims[dim]
        split_dict[f"uniform_dim_{dim}"] = applied_dim

    exploration_df = get_exploration_df(split_dict)

    print(
        f"{'array dim:':12} spatial (0-2) and temporal (3) image array dimensions [pixel]",
        f"\n{'pixdim:':12} dimensions of each pixel [mm]",
        f"\n{'applied dim:':12} total image dimensions after applying pixdim [mm] (= dim * pixdim)",
        f"\n{'uniform dim:':12} image array dimensions after median pixdim has been",
        "applied, resulting in consistent/uniform pixdims [pixel]",
        f"\n{' ':12} (= dim * pixdim / median_pixdim), where median pixdim={median_pixdims}",
    )

    if return_results:
        return exploration_df

    pd.options.display.float_format = "{:,.2f}".format
    print(exploration_df)


def get_exploration_df(
    array_dict,
    functions=[np.mean, np.median, np.std],
    quantiles=[0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0],
):
    # caluclate statistics for each data column
    exploration_dict = {}
    for column in array_dict:
        for f in functions:
            if f.__name__ not in exploration_dict:
                exploration_dict[f.__name__] = []
            exploration_dict[f.__name__].append(f(array_dict[column]))
        for q in quantiles:
            name = f"q: {q:0.2f}"
            if name not in exploration_dict:
                exploration_dict[name] = []
            exploration_dict[name].append(
                np.quantile(array_dict[column], q, method="nearest")
            )

    # create dataframe from dictionary
    columns = [column.replace("_", " ") for column in array_dict]
    exploration_df = pd.DataFrame.from_dict(exploration_dict, orient="index")
    exploration_df.columns = columns
    return exploration_df


if __name__ == "__main__":
    config_path = Path("data-config.yaml")
    explore_development_data_from_config(config_path)
