"""
Author: Jesús Maldonado
Description: "CNN 1D" algorithm for EEG data classification.
"""

import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    GRU model for EEG data classification.

    Parameters:
        in_channels (int): Number of input channels (features per time step, default: 28).
        hidden_size (int): Number of features in the hidden state (default: 64).
        num_layers (int): Number of recurrent layers (default: 2).
        bidirectional (bool): If True, becomes a bidirectional GRU (default: False).
        dropout_rate (float): Dropout probability between GRU layers (default: 0.2).
    """
    def __init__(self,
                 in_channels: int = 28,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout_rate: float = 0.2
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        # GRU layer
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional, 
            device=None, 
            dtype=None
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected output layer
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Permute to (batch_size, seq_length, in_channels)
        x = x.permute(0, 2, 1)

        # GRU forward
        output, h_n = self.gru(x)

        # Select last hidden state
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_size)
            forward_last = h_n[-2, :, :]
            backward_last = h_n[-1, :, :]
            last_hidden = torch.cat((forward_last, backward_last), dim=1)
        else:
            # h_n shape: (num_layers, batch, hidden_size)
            last_hidden = h_n[-1, :, :]

        # Apply dropout and fully connected layer
        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out
