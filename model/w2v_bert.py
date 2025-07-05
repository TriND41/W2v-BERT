import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.extraction import ConvolutionSubsampling
from model.modules.quantization import GumbelQuantizer
from model.modules.augmentation import TimeMasking
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
        
    ):
        super().__init__()
        self.extractor = ConvolutionSubsampling(in_channels=in_channels, hidden_channels=hidden_channels)
        