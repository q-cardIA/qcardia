"""A module containing a 2D U-Net encoder + multilayer perceptron implementation for 
    U-Net pretraining.

Classes:
- EncoderMLP2d: A representation for a 2D U-Net encoder + multilayer perceptron.

"""

import torch
import torch.nn as nn

from qcardia_models.models.building_blocks import Encoder, MultiLayerPerceptron


class EncoderMLP2d(nn.Module):
    """A representation for a 2D U-Net encoder + multilayer perceptron projection head.

    Args:
    - nr_input_channels (int): The number of channels of the input image.
    - encoder_channels_list (list[int]): A list holding the number of channels for each
        encoder output block.
    - mlp_channels_list (list[int]): A list holding the number of channels for each MLP
        hidden layer output.
    - output_feature_size (int): The size of the feature vector output.

    Methods:
    - forward(x): Performs the forward pass of the U-Net encoder + multilayer perceptron
        projection head.
    """

    def __init__(
        self,
        nr_input_channels: int,
        encoder_channels_list: list[int],
        mlp_channels_list: list[int],
        mlp_relu_slope: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(nr_input_channels, encoder_channels_list)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MultiLayerPerceptron(
            input_feature_size=encoder_channels_list[-1],
            output_feature_size_list=mlp_channels_list,
            relu_slope=mlp_relu_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the U-Net encoder and MLP.

        Uses global average pooling and flattening to convert the encoder output into a
            feature vector to feed to the MLP.

        Args:
            x (torch.Tensor): The input to the model (image).

        Returns:
            torch.Tensor: The output of the model (feature vector).
        """
        encoder_features = self.encoder(x)[-1]  # get deepest encoder features only
        pooled_features = self.pooling(encoder_features)  # global average pooling
        flat_features = torch.flatten(pooled_features, start_dim=1)  # flatten features
        return self.mlp(flat_features)
