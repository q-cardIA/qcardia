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
