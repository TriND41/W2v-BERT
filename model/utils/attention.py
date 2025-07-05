import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def get_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).min,
    eps32: float = torch.finfo(torch.float32).min,
    eps64: float = torch.finfo(torch.float64).min
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        return -torch.inf

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(in_features=embedding_dim, out_features=3*embedding_dim)
        self.position_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.dropout = nn.Dropout(p=dropout_p)
        self.content_bias = nn.Parameter(torch.zeros([n_heads, self.head_dim]), requires_grad=True)
        self.position_bias = nn.Parameter(torch.zeros([n_heads, self.head_dim]), requires_grad=True)

    def __relative_shift(self, pos_scores: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length1, seq_length2 = pos_scores.size()

        zeros = torch.zeros([batch_size, self.n_heads, seq_length1, 1], dtype=pos_scores.dtype, device=pos_scores.device)
        padded_pos_scores = torch.concatenate([zeros, pos_scores], dim=3)

        padded_pos_scores = padded_pos_scores.view([batch_size, self.n_heads, seq_length2 + 1, seq_length1])
        return padded_pos_scores[:, :, 1:].view([batch_size, self.n_heads, seq_length1, seq_length2])[:, :, :, :seq_length2//2 + 1]
    
    def __scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_embedding: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        content_scores = torch.matmul((q + self.content_bias).transpose(1, 2), k.transpose(2, 3))
        
        position_scores = torch.matmul((q + self.position_bias).transpose(1, 2), pos_embedding)
        position_scores = self.__relative_shift(position_scores)

        attn_scores = (content_scores + position_scores) * self.scale_factor
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, value=get_eps(attn_scores))

        attn_weights = F.softmax(attn_scores, dim=3)
        attn_weights = self.dropout(attn_weights)

        attn_context = torch.matmul(attn_weights, v)
        return attn_context
    
    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, length, _ = x.size()

        q, k, v = torch.chunk(self.qkv_proj(x), chunks=3, dim=2)

        q = q.view([batch_size, length, self.n_heads, self.head_dim])
        k = k.view([batch_size, length, self.n_heads, self.head_dim]).transpose(1, 2)
        v = v.view([batch_size, length, self.n_heads, self.head_dim]).transpose(1, 2)

        pos_embedding = self.position_proj(pos_embedding)

        attn_context = self.__scaled_dot_product_attention(q, k, v, pos_embedding, attn_mask)
        attn_context = attn_context.transpose(1, 2).contiguous().view([batch_size, length, self.embedding_dim])
        attn_context = self.out_proj(attn_context)

        return attn_context