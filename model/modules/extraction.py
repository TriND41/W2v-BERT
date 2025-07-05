import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionSubsampling(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels)
        self.conv_2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        batch_size, _, feature_size, context_length = x.size()
        x = x.permute([0, 3, 1, 2]).contiguous().view([batch_size, context_length, feature_size * self.hidden_channels])
        return x