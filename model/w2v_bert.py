import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.extraction import ConvolutionSubsampling
from model.modules.quantization import GumbelQuantizer
from model.modules.augmentation import TimeMasking
from model.utils.masking import sample_mask
from typing import Optional, Tuple

class W2vBERT(nn.Module):
    def __init__(
        self,
        # Encoder configs
        n_contrastive_blocks: int = 6,
        n_mlm_blocks: int = 6,
        in_channels: int = 80,
        hidden_channels: int = 512,
        embedding_dim: int = 512,
        n_heads: int = 8,
        kernel_size: int = 5,
        max_positions: Optional[int] = None,
        n_ffn_factors: int = 4,
        dropout_p: float = 0.0,
        # Quantization configs
        num_groups: int = 2,
        num_vars: int = 320,
        vq_dim: int = 768,
        temperature: float = 2.0,
        weight_proj_depth: int = 1,
        weight_proj_factor: int = 1,
        activation: str = 'gelu',
        hard: bool = True,
        std: float = 0.0,
        # Masking config
        n_time_masks: int = 2,
        time_mask_param: int = 27,
        time_ratio: float = 1.0,
        zero_masking: bool = True
    ):
        super().__init__()
        self.extractor = ConvolutionSubsampling(in_channels=in_channels, hidden_channels=hidden_channels)
        
        self.encoder = Encoder(
            n_contrastive_blocks=n_contrastive_blocks,
            n_mlm_blocks=n_mlm_blocks,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            kernel_size=kernel_size,
            max_positions=max_positions,
            n_ffn_factors=n_ffn_factors,
            dropout_p=dropout_p
        )

        self.quantization = GumbelQuantizer(
            input_dim=((in_channels // 2 - 1) // 2 - 1) * hidden_channels,
            num_groups=num_groups,
            num_vars=num_vars,
            vq_dim=vq_dim,
            temperature=temperature,
            weight_proj_depth=weight_proj_depth,
            weight_proj_factor=weight_proj_factor,
            activation=activation,
            hard=hard,
            std=std
        )

        self.time_masking = TimeMasking(
            n_masks=n_time_masks,
            mask_param=time_mask_param,
            ratio=time_ratio,
            zero_masking=zero_masking
        )

        self.contrastive_proj = nn.Linear(in_features=embedding_dim, out_features=vq_dim)
        self.mlm_proj = nn.Linear(in_features=embedding_dim, out_features=num_vars)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.extractor(x)
        if attn_mask is not None:
            attn_mask = sample_mask(x, attn_mask)
        
        target_context_vectors, discreted_ids, prob_perplexity = self.quantization(x)

        with torch.no_grad():
            masked_x, tmask = self.time_masking(x)

        mlm_vectors, context_vectors = self.encoder(masked_x, attn_mask)
        context_vectors = self.contrastive_proj(context_vectors)
        mlm_vectors = self.mlm_proj(mlm_vectors)

        return (context_vectors, mlm_vectors, tmask), (target_context_vectors, discreted_ids, prob_perplexity)