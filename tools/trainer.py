import os

import torch

from typing import Optional, Literal

class Trainer:
    def __init__(
        self,
        rank: int,
        # Audio configs
        sample_rate: int = 16000,
        # Training configs
        n_fft: int = 400,
        win_length: Optional[int] = 400,
        hop_length: Optional[int] = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 80,
        window_fn: Literal['hann', 'hamming'] = 'hann',
        power: Optional[float] = 2.0,
        frame_norm: bool = False,
        window_norm: bool = False,
        center: bool = True,
        pad_mode: str = 'reflect',
        slaney_norm: bool = False,
        mel_scale: Literal['htk', 'slaney'] = 'htk',
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
        dropout_p: float = 0.1,
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
        # Masking
        n_time_masks: int = 2,
        time_mask_param: int = 27,
        time_ratio: float = 1.0,
        zero_masking: bool = True,

    ) -> None:
        pass