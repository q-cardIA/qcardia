"""
This module contains utility functions for the baseline-unet package.

"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility. It also sets the `deterministic` flag for the
    PyTorch backend to ensure that the results are reproducible across different runs.

    Example usage:
        import utils

    # Set the random seed to 42
    utils.seed_everything(42)

    Args:
        seed: The random seed to use.

    Returns:
        None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch.use_deterministic_algorithms(True) # Disabled for ease of use.
    # When uncommented, some trainings may give a RuntimeError with this fix: To enable
    # deterministic behavior in this case, you must set an environment variable before
    # running your PyTorch application:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8
