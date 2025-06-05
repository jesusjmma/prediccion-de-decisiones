"""
Author: Jesús Maldonado
Description: "CNN 1D" algorithm for EEG data classification.
"""

import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    CNN 1D model for EEG data classification.

    Parameters:
        in_channels (int): Number of input channels (default: 28).
        hidden_channels (list[int]): List of hidden channels for each convolutional layer (default: [16, 32]).
        kernel_sizes (list[int]): List of kernel sizes for each convolutional layer (default: [5, 3]).
        dropout_rate (float): Dropout probability (default: 0.2).
    """

    def __init__(self,
                 in_channels:     int       = 28,
                 hidden_channels: list[int] = [16, 32],
                 kernel_sizes:    list[int] = [5, 3],
                 dropout_rate:    float     = 0.2
                ):
        
        super(CNN1D, self).__init__()

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes    = kernel_sizes
        self.dropout_rate    = dropout_rate
        
        assert len(hidden_channels) == len(kernel_sizes), "hidden_channels and kernel_sizes must have the same length"

        layers = []
        prev_ch = in_channels
        for out_ch, k in zip(hidden_channels, kernel_sizes):
            layers += [nn.Conv1d(prev_ch, out_ch, kernel_size=k, padding=k//2, bias=False),
                       nn.BatchNorm1d(out_ch),
                       nn.ReLU(inplace=True)
            ]
            prev_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout_rate)
        self.fc   = nn.Linear(prev_ch, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv1d):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN 1D model.
        This method applies the following operations:
            1. For each convolutional layer:
                - Convolution
                - Batch normalization
                - ReLU activation
            2. Pooling:
                - Adaptive max pooling to reduce the output to a fixed size (1 in this case).
                - Squeeze to remove the last dimension (because the output of the pooling layer is (batch_size, channels, 1)).
            3. Dropout for regularization when the result
            4. Fully connected layer to produce the final output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        out = self.fc(x)

        return out