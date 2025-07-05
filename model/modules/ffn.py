import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.activation import Swish

class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, n_factors: int = 4, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.hidden_layer = nn.Linear(in_features=dim, out_features=n_factors*dim)
        self.swish = Swish()
        self.out_layer = nn.Linear(in_features=n_factors*dim, out_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.hidden_layer(y)
        y = self.swish(y)
        y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out_layer(y)
        y = F.dropout(y, p=self.dropout_p, training=self.training)
        return x + 0.5*y