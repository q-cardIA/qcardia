"""
This module contains utility classes and functions for training and evaluating the
models implemented in this package.

Classes: 
- EarlyStopper: Implements early stopping to prevent overfitting during model
training. 
- MultiScaleLoss: Calculates the loss for a multi-scale UNet model during training.

"""

import numpy as np
import torch
from torch import Tensor, nn


class EarlyStopper:
    """Implements early stopping to prevent overfitting during model training.

    Attributes:
        patience: An integer representing the number of epochs to wait before stopping
            training if the validation loss does not improve. Defualts to 2.
        min_delta: A float representing the minimum change in validation loss required
            to be considered an improvement. Defaults to 0.
        counter: An integer representing the number of epochs since the last improvement
            in validation loss.
        min_validation_loss: A float representing the minimum validation loss
            observed during training.

    Methods:
        early_stop(validation_loss):
            Checks if training should be stopped early based on the current validation
            loss.


    """

    def __init__(self, patience: int = 2, min_delta: float = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """Checks if training should be stopped early based on the current validation
            loss.

        Args:
            validation_loss: the current validation loss.

        Returns:
            A boolean value indicating whether training should be stopped (True) or not
            (False).

        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MultiScaleLoss(nn.Module):
    """Calculates a multi-scale loss for a given set of predicted and ground truth
        segmentation masks.

    Attributes:
        loss_function: The loss function to use for calculating the loss.

    Methods:
        forward(outputs, labels):
            Calculates the multi-scale loss for the given predicted and ground truth
                segmentation masks.

    """

    def __init__(self, loss_function: nn.Module) -> None:
        super().__init__()
        self.loss_function = loss_function

    def forward(self, outputs: list[Tensor], labels: Tensor) -> Tensor:
        """Calculates the multi-scale loss of the given predicted and ground truth masks

        The multi-scale loss is calculated by first calculating the loss for the highest
        resolution output, and then adding the losses for the lower resolution
        outputs, weighted by a scale factor.

        Args:
            outputs: A list of tensors representing the predicted segmentation masks at
                different scales.
            labels: A tensor representing the ground truth segmentation masks.

        Returns:
            A tensor representing the multi-scale loss for the included classes.

        """
        # Calculate the loss for the highest resolution output.
        loss = self.loss_function(outputs[0], labels)
        nr_outputs = len(outputs)
        # If there are lower resolution outputs, calculate the loss for each of them.
        if nr_outputs > 1:
            scale_factors = 2.0 ** -torch.arange(nr_outputs)
            loss_weights = scale_factors / torch.sum(scale_factors)
            loss *= loss_weights[0]
            for i in range(1, nr_outputs):
                # Downsample the labels to match the size of the current output.
                downsampled_labels = torch.nn.functional.interpolate(
                    labels, size=outputs[i].shape[2:], mode="nearest-exact"
                )
                loss += (
                    self.loss_function(outputs[i], downsampled_labels) * loss_weights[i]
                )
        return loss
