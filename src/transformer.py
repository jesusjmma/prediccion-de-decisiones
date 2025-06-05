"""
Author: Jesús Maldonado
Description: "Transformer" algorithm for EEG data classification.
"""

import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder-based model for EEG data classification.

    Parameters:
        in_channels (int): Number of input channels (features per time step, default: 28).
        d_model (int): Dimension of the embedding/input projection (default: 128).
        nhead (int): Number of attention heads (default: 8).
        num_layers (int): Number of TransformerEncoder layers (default: 2).
        dim_feedforward (int): Dimension of the feedforward network model (default: 256).
        dropout_rate (float): Dropout probability (default: 0.1).
    """
    def __init__(self,
                 in_channels: int = 28,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate

        # Input projection to d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout and output layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, 1)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.TransformerEncoderLayer):
            # The submodules inside the layer are initialized by default
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # (batch, in_channels, seq_length) -> (batch, seq_length, in_channels)
        x = x.permute(0, 2, 1)

        # Project to d_model
        x = self.input_proj(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Pool over sequence (mean)
        x = x.mean(dim=1)

        # Dropout and fully connected layer
        x = self.dropout(x)
        out = self.fc(x)
        return out
