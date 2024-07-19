"""
This module contains loss classes and functions for training and evaluating the
models implemented in this package.

Classes: 
- DiceLoss: Calculates the Dice loss for a given set of predicted and ground truth 
segmentation masks. 
- DiceCELoss: Calculates a combination of cross-entropy and Dice loss for a given set 
of predicted and ground truth segmentation masks. 

"""

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """A class for calculating the Dice loss for semantic segmentation tasks.

    The Dice loss is a measure of the overlap between the predicted and ground truth
    segmentation masks. It is calculated as the sum of the element-wise product of the
    predicted and ground truth masks, divided by the sum of the predicted and ground
    truth masks. The Dice loss is then subtracted from 1 to get a loss value that is
    lower for better overlap.

    Args:
    - class_weights (List[float]): A list of weights for each class in the segmentation.
        The weights are normalized to sum to 1.
    - logits_to_probabilities (bool): A flag indicating whether to convert the output
        logits to probabilities using a sigmoid or softmax activation function.
        Defaults to True.
    - smooth (float): A small constant added to the numerator and denominator of the
        Dice coefficient to prevent division by zero. Defaults to 1e-5.

    Attributes:
    - class_weights (Tensor): A tensor of the class weights, moved to the same device
        as the predicted masks.
    - logits_to_probabilities (bool): A flag indicating whether to convert the output
        logits to probabilities.
    - smooth (float): A small constant added to prevent division by zero.

    Methods:
    - forward(outputs, targets): Calculates the Dice loss for the given predicted and
        ground truth masks.

    """

    def __init__(
        self,
        class_weights: list[float],
        logits_to_probabilities: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        class_weights = torch.tensor(class_weights)
        # Normalize the class weights to sum to 1.
        self.class_weights = (class_weights / torch.sum(class_weights)).unsqueeze(0)
        self.logits_to_probabilities = logits_to_probabilities
        self.smooth = smooth

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Calculates the Dice loss for the given predicted and ground truth masks.

        Args:
            outputs: A tensor representing the predicted segmentation masks.
            targets: A tensor representing the ground truth segmentation masks.

        Returns:
            A tensor representing the Dice loss for the included classes.

        """
        if outputs.get_device() >= 0:
            # Move the class weights to the same device as the predicted masks.
            self.class_weights = self.class_weights.to(outputs.get_device())
        if self.logits_to_probabilities:
            if self.class_weights.shape[1] == 1:
                # sigmoid activation for binary classification
                outputs = torch.sigmoid(outputs)
            else:
                # softmax activation for multiclass classification
                outputs = torch.softmax(outputs, 1)

        sum_dim = torch.arange(2, len(outputs.shape)).tolist()
        intersection = torch.sum(outputs * targets, dim=sum_dim)
        outputs = torch.sum(outputs, dim=sum_dim)
        targets = torch.sum(targets, dim=sum_dim)

        # smooth to prevent division by zero
        dice = (2.0 * intersection + self.smooth) / (outputs + targets + self.smooth)
        loss = torch.sum((1.0 - dice) * self.class_weights)  # multiply by class weights
        return loss / outputs.shape[0]


class DiceCELoss(nn.Module):
    """Calculates a combined Dice and cross-entropy loss for a given set of predicted
        and ground truth segmentation masks.

    Attributes:
        cross_entropy_weight: A float representing the weight to apply to the
            cross-entropy loss.
        dice_weight: A float representing the weight to apply to the Dice loss.
        dice_classes_weights: A list of floats representing the weights to apply to
            each class in the Dice loss calculation.
        dice_smooth: A float representing the smoothing factor to use in the Dice
            loss calculation.

    Methods:
        forward(outputs, targets):
            Calculates the combined Dice and cross-entropy loss for the given predicted
            and ground truth segmentation masks.

    """

    def __init__(
        self,
        cross_entropy_loss_weight: float,
        dice_loss_weight: float,
        dice_classes_weights: list[float],
        dice_smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cross_entropy_weight = cross_entropy_loss_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.dice_weight = dice_loss_weight
        self.dice_loss = DiceLoss(
            dice_classes_weights,
            logits_to_probabilities=True,
            smooth=dice_smooth,
        )

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Calculates the combined Dice and cross-entropy loss for the given predicted
            and ground truth segmentation masks.

        Args:
            outputs: A tensor representing the predicted segmentation masks.
            targets: A tensor representing the ground truth segmentation masks.

        Returns:
            A tensor representing the combined loss, cross-entropy loss, and Dice loss.

        """
        cross_entropy = self.cross_entropy_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        total = cross_entropy * self.cross_entropy_weight + dice * self.dice_weight
        return torch.cat([total.reshape(1), cross_entropy.reshape(1), dice.reshape(1)])
