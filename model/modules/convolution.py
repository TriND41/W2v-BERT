import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.activation import GLU, Swish

class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(normalized_shape=channels)
        self.pointwise_conv_1 = nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1)
        self.glu = GLU(dim=1)
        self.deepwise_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels)
        self.batch_norm = nn.BatchNorm1d(num_features=channels)
        self.swish = Swish()
        self.pointwise_conv_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x).transpose(1, 2)
        y = self.pointwise_conv_1(y)
        y = self.glu(y)
        y = self.deepwise_conv(y)
        y = self.batch_norm(y)
        y = self.swish(y)
        y = self.pointwise_conv_2(y)
        y = F.dropout(y.transpose(1, 2), p=self.dropout_p, training=self.training)
        return x + y