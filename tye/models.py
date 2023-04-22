"""
Models for evaluating the performance of gnn embedding.
"""
import torch
from torch import nn
from torch.utils.data import Dataset


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    """

    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.num_levels = len(num_channels)
        self.layers = nn.ModuleList()
        for i in range(self.num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dilation=dilation_size,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_inputs, seq_len)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        y = x
        for i in range(self.num_levels):
            y = self.layers[i](y)
        return y
