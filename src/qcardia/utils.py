from pathlib import Path
from typing import List

import torch


def t_2d_scale(scales: List[float]):
    """
    Create a 2D scale transformation matrix.

    This function creates a 2D scale transformation matrix from the given scales.
    The scales are used to scale the x and y coordinates.

    Args:
        scales (list): A list of two scale factors for the x and y coordinates.

    Returns:
        torch.Tensor: A 2D scale transformation matrix.
    """
    t_scale = torch.tensor(
        [
            [scales[1], 0, 0],
            [0, scales[0], 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return t_scale


def standardise(tensor: torch.Tensor):
    """
    Standardise a tensor.

    Args:
        tensor (torch.Tensor): The tensor to be standardised.

    Returns:
        torch.Tensor: The standardised tensor.
    """
    return (tensor - tensor.mean()) / tensor.std()


def get_data_directory(chamber_dir: Path) -> Path | None:
    """Find the folder holding a chamber's DICOM files.

    Short-axis exports keep them directly in the chamber directory; long-axis
    exports put them one level down, sometimes alongside other series.

    Args:
        chamber_dir (Path): A chamber directory, e.g. CINE_SAX.

    Returns:
        Path to the DICOM folder, or None if there is no candidate.
    """
    if list(chamber_dir.glob("*.dcm")):
        return chamber_dir

    subdirs = [
        d
        for d in chamber_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and "segmentation" not in d.name.lower()
        and "result" not in d.name.lower()
    ]
    if not subdirs:
        return None

    preferred = {"sa stack", "2ch", "3ch", "4ch", "scar_sa", "scar_2ch"}
    for subdir in subdirs:
        if subdir.name.lower() in preferred:
            return subdir

    def dcm_count(directory: Path) -> int:
        return sum(
            1
            for f in directory.iterdir()
            if f.suffix.lower() == ".dcm" and not f.name.startswith(".")
        )

    return max(subdirs, key=dcm_count)
