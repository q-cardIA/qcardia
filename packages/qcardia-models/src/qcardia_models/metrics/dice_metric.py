"""
This module contains metric classes and functions for evaluating the
models implemented in this package.

Classes: 
- DiceMetric: Calculates the Dice coefficient for a given set of predicted and
ground truth segmentation masks.

"""

import torch
from torch import Tensor, nn


class DiceMetric(nn.Module):
    """Calculates the Dice coefficient for a given set of predicted and ground truth
        segmentation masks.

    Attributes:
        included_class_idxs: A list of integers representing the indices of the
            classes to include in the calculation.

    Methods:
        forward(outputs, targets):
            Calculates the Dice coefficient for the given predicted and ground
            truth segmentation masks.

    """

    def __init__(self, included_class_idxs: list[int]) -> None:
        super().__init__()
        self.included_class_idxs = included_class_idxs

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Calculates the Dice coefficient for the given predicted and ground truth
            segmentation masks.

        Args:
            outputs: A tensor representing the predicted segmentation masks.
            targets: A tensor representing the ground truth segmentation masks.

        Returns:
            A tensor representing the Dice coefficient for each included class.

        """
        # Get the predicted class indices by taking the argmax of the predicted masks.
        class_idx_preds = torch.argmax(outputs, dim=1, keepdim=True)
        # Convert the predicted class indices to one-hot vectors.
        outputs = torch.scatter(torch.zeros_like(outputs), 1, class_idx_preds, 1.0)
        # Only include the specified classes in the calculation.
        outputs = outputs[:, self.included_class_idxs, ...]
        targets = targets[:, self.included_class_idxs, ...]

        sum_dim = torch.arange(2, len(outputs.shape)).tolist()
        intersection = torch.sum(outputs * targets, dim=sum_dim)
        outputs = torch.sum(outputs, dim=sum_dim)
        targets = torch.sum(targets, dim=sum_dim)

        dice = (2.0 * intersection) / (outputs + targets)

        # Handle the case where both outputs and targets are 0.
        dice[(outputs == 0.0) & (targets == 0.0)] = 1.0
        # Handle the case where either outputs or targets is 0.
        dice[(outputs == 0.0) ^ (targets == 0.0)] = 0.0
        return torch.mean(dice, dim=0)
