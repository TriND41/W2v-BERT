import torch
import torch.nn as nn
import math
from typing import Optional

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: Optional[int] = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if max_positions is None:
            self.register_buffer('pe', torch.empty(0))
        else:
            self.register_buffer('pe', self.__encode_positions(max_positions))

    def __encode_positions(self, length: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        div_term = torch.exp(
            (torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.embedding_dim)).unsqueeze(dim=0)
        )
        positions = torch.arange(length, dtype=torch.float, device=device).unsqueeze(dim=1)

        angles = torch.matmul(div_term, positions)

        positive_pe = torch.zeros([length, self.embedding_dim], dtype=torch.float, device=device)
        negative_pe = torch.zeros([length, self.embedding_dim], dtype=torch.float, device=device)

        positive_pe[:, 0::2] = torch.sin(angles)
        positive_pe[:, 1::2] = torch.cos(angles)
        negative_pe[:, 0::2] = torch.sin(-angles)
        negative_pe[:, 1::2] = torch.cos(-angles)

        positive_pe = torch.flip(positive_pe, dims=[0]).unsqueeze(dim=0)
        negative_pe = negative_pe[1:].unsqueeze(dim=0)

        pe = torch.concatenate([negative_pe, positive_pe], dim=1)
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe.numel() == 0 or self.pe.size(1) < x.size(1) * 2 - 1:
            self.pe = self.__encode_positions(x.size(1), device=x.device)
        return self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]