import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.attention import RelativeMultiHeadAttention
from typing import Optional

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attention = RelativeMultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads, dropout_p=dropout_p)
    
    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.attention(y, pos_embedding, attn_mask)
        y = F.dropout(y, p=self.dropout_p, training=self.training)
        return x + y