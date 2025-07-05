import torch
import torch.nn as nn
from model.utils.position import RelativePositionalEncoding
from model.utils.masking import sample_mask
from model.modules.extraction import ConvolutionSubsampling
from model.modules.block import ConformerBlock
from typing import Optional, Tuple

class Encoder(nn.Module):
    def __init__(
        self,
        n_contrastive_blocks: int,
        n_mlm_blocks: int,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        n_heads: int,
        kernel_size: int,
        max_positions: Optional[int] = None,
        n_ffn_factors: int = 4,
        dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=((in_channels // 2 - 1) // 2 - 1) * hidden_channels, out_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.relative_pe = RelativePositionalEncoding(embedding_dim=embedding_dim, max_positions=max_positions)
        
        self.contrastive_blocks = nn.ModuleList([ConformerBlock(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            kernel_size=kernel_size,
            n_ffn_factors=n_ffn_factors,
            dropout_p=dropout_p
        ) for _ in range(n_contrastive_blocks)])

        self.mlm_blocks = nn.ModuleList([ConformerBlock(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            kernel_size=kernel_size,
            n_ffn_factors=n_ffn_factors,
            dropout_p=dropout_p
        ) for _ in range(n_mlm_blocks)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.dropout(x)
        pos_embedding = self.relative_pe(x)
        
        for block in self.contrastive_blocks:
            x = block(x, pos_embedding, attn_mask)
        context_vectors = x.clone()
        
        for block in self.mlm_blocks:
            x = block(x, pos_embedding, attn_mask)

        return x, context_vectors