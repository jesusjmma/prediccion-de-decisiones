"""
Author: Jesús Maldonado
Description: "CNN 1D" algorithm for EEG data classification.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.logger_utils import setup_logger
logger = setup_logger(name=Path(__file__).name, level=10)

class CNN1D(nn.Module):
    """
    CNN 1D model for EEG data classification.

    Args:
        in_channels (int): Number of input channels (default: 20).
        num_classes (int): Number of output classes (default: 2).
        hidden_channels (list[int]): List of hidden channels for each convolutional layer (default: [32, 64, 128]).
        kernel_sizes (list[int]): List of kernel sizes for each convolutional layer (default: [7, 5, 3]).
        pool_kernel (int): Kernel size for max pooling layer (default: 2).
        dropout (float): Dropout probability (default: 0.5).
    """

    def __init__(self, in_channels: int = 20, num_classes: int = 2, hidden_channels: list[int] = [16, 32, 64], kernel_sizes: list[int] = [5, 3, 2], pool_kernel: int = 1, dropout: float = 0.5):
        super(CNN1D, self).__init__()

        assert len(hidden_channels) == len(kernel_sizes), "hidden_channels and kernel_sizes must have the same length"
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        prev_ch = in_channels
        for out_ch, k in zip(hidden_channels, kernel_sizes):
            self.conv_layers.append(nn.Conv1d(prev_ch, out_ch, kernel_size=k))
            self.bn_layers.append(nn.BatchNorm1d(out_ch))
            prev_ch = out_ch

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_channels[-1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN 1D model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.pool(x)
        
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x