"""A module containing building blocks for networks.

Classes:
- Block: A representation for the basic convolutional building block of the U-Net.
- Encoder: A representation for the encoder part of the U-Net.
- Decoder: A representation for the decoder part of the U-Net.
- MultiLayerPerceptron: A representation for a multi-layer perceptron.

"""

import torch
import torch.nn as nn
from torch import Tensor


class Block(nn.Module):
    """A representation for the basic convolutional building block of the U-Net.

    Args:
    - input_channels (int): The number of input channels to the block.
    - output_channels (int): The number of output channels of the block.
    - initial_stride (int): The stride of the first convolutional layer. Defaults to 1.

    Attributes:
    - conv1 (nn.Conv2d): The first convolutional layer.
    - bn1 (nn.BatchNorm2d): The batch normalization layer after the first convolutional
        layer.
    - conv2 (nn.Conv2d): The second convolutional layer.
    - bn2 (nn.BatchNorm2d): The batch normalization layer after the second convolutional
        layer.
    - lrelu (nn.LeakyReLU): The LeakyReLU activation function.

    Methods:
    - forward(x): Performs the forward pass of the block.

    """

    def __init__(
        self, input_channels: int, output_channels: int, initial_stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=initial_stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the block.

        Args:
            x (Tensor): The input to the block.

        Returns:
            Tensor: The output of the forward pass.

        """

        # a block consists of two convolutional layers
        # with Leaky ReLU activations and batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        return x


class Encoder(nn.Module):
    """A representation for the encoder part of the U-Net.

    Args:
    - input_channels (int): The number of input channels of the original image.
    - channels (list[int]): A list holding the number of output channels of each
        block in the encoder.

    Methods:
    - forward(x): Performs the forward pass for all blocks in the encoder.

    """

    def __init__(self, input_channels: int, channels: list[int]) -> None:
        super().__init__()
        blocks = [Block(input_channels, channels[0], 1)] + [
            Block(channels[i], channels[i + 1], 2) for i in range(len(channels) - 1)
        ]
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Performs the forward pass for all blocks in the encoder.

        Args:
            x (Tensor): The input to the encoder.

        Returns:
            list[Tensor]: A list containing the outputs of each block in the encoder.

        """
        features = []  # a list to store features
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)  # save features to concatenate to decoder blocks
        return features


class Decoder(nn.Module):
    """A representation for the decoder part of the U-Net.

    Args:
    - channels (list[int]): A list holding the number of input channels of each
        block in the decoder.
    - num_classes (int): The number of output classes of the segmentation.
    - nr_output_scales (int): The number of output scales to use. Defaults to 1.
    - dropout (float): The dropout probability to use before the output convolutions. Defaults to 0.0.

    Methods:
    - forward(x, encoder_features): Performs the forward pass for all blocks in the decoder.

    """

    def __init__(
        self,
        channels: list[int],
        num_classes: int,
        nr_output_scales: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.nr_blocks = len(channels)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels[i - 1], channels[i], kernel_size=2, stride=2
                )
                for i in range(1, self.nr_blocks)
            ]
        )
        # the initial filter size of the Block is doubled, due to the concatenation
        # of the features (which creates twice as many).
        self.dec_blocks = nn.ModuleList(
            [Block(2 * channels[i], channels[i]) for i in range(1, self.nr_blocks)]
        )

        if nr_output_scales == 0:
            nr_output_convs = self.nr_blocks
        elif nr_output_scales > 0:
            nr_output_convs = min(self.nr_blocks, nr_output_scales)
        else:
            nr_output_convs = max(1, self.nr_blocks + nr_output_scales)

        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[self.nr_blocks - i], num_classes, 1)
                for i in range(nr_output_convs, 0, -1)
            ]
        )
        self.output_threshold_idx = self.nr_blocks - nr_output_convs
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor, encoder_features: list[Tensor]) -> list[Tensor]:
        """Performs the forward pass for all blocks in the decoder.

        Args:
            x (Tensor): The input to the decoder.
            encoder_features (list[Tensor]): A list containing the encoder features
                to be concatenated to the corresponding level of the decoder.

        Returns:
            list[Tensor]: A list containing the outputs of each block in the decoder.

        """
        outputs = []
        for i in range(self.nr_blocks - 1):
            if i >= self.output_threshold_idx:
                outputs.append(
                    self.output_convs[i - self.output_threshold_idx](self.dropout(x))
                )

            # transposed convolution
            x = self.upconvs[i](x)

            # concatenate features from the corresponding level of the encoder to x
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        outputs.append(self.output_convs[-1](self.dropout(x)))

        return outputs[::-1]


class MultiLayerPerceptron(nn.Module):
    """A representation for a basic multilayer perceptron, a dense/fully connected feed
        forward network with (Leaky)ReLU activations.

    Args:
        input_feature_size (int): Expected size of the input feature vector.
        output_feature_size_list (list[int]): List of output feature sizes for each
            layer, the final value is the output feature size of the MLP.
        relu_slope (float, optional): negative slope of the Leaky ReLU activation.
            Defaults to 0.0, which is equivalent to a classic ReLU activation.

    Methods:
    - forward(x): Performs the forward pass of the block.

    """

    def __init__(
        self,
        input_feature_size: int,
        output_feature_size_list: list[int],
        relu_slope: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_feature_size, output_feature_size_list[0])]
        for i in range(1, len(output_feature_size_list)):
            layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            layers.append(
                nn.Linear(output_feature_size_list[i - 1], output_feature_size_list[i])
            )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the block.

        Args:
            x (Tensor): The input to the block.

        Returns:
            Tensor: The output of the forward pass.

        """
        return self.mlp(x)
