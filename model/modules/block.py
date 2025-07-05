import torch
import torch.nn as nn
from model.modules.attention import MultiHeadSelfAttentionModule
from model.modules.ffn import FeedForwardModule
from model.modules.convolution import ConvolutionModule
from typing import Optional

class ConformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, kernel_size: int, n_ffn_factors: int = 4, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ffn_1 = FeedForwardModule(dim=embedding_dim, n_factors=n_ffn_factors, dropout_p=dropout_p)
        self.attention = MultiHeadSelfAttentionModule(embedding_dim=embedding_dim, n_heads=n_heads, dropout_p=dropout_p)
        self.convolution = ConvolutionModule(channels=embedding_dim, kernel_size=kernel_size, dropout_p=dropout_p)
        self.ffn_2 = FeedForwardModule(dim=embedding_dim, n_factors=n_ffn_factors, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.ffn_1(x)
        x = self.attention(x, pos_embedding, attn_mask)
        x = self.convolution(x)
        x = self.ffn_2(x)
        return x