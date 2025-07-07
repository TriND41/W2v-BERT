import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed

from handlers.loaders import AudioDataset
from handlers.checkpoint import CheckpointManager, load_checkpoint
from handlers.configs import W2vBERTConfig, AudioProcessorConfig, LogMelSpectrogramConfig
from handlers.symbols import CheckpointKey

from typing import Optional, Literal, Dict, Any

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
        # Training
        lr: Optional[float] = None,
        # Checkpoint
        checkpoint_path: Optional[str] = None,
        checkpoint_folder: str = "./checkpoints",
        n_saved_checkpoints: int = 3,
        save_checkpoint_after_steps: Optional[int] = None,
        save_checkpoint_after_epochs: int = 1,
        # Early Stopping
        early_stopping: bool = False,
        n_patiences: int = 3,
        observe: Literal['loss', 'score'] = 'loss',
        # Logging
        logging: bool = False,
        logging_project: str = "W2v - BERT",
        logging_name: Optional[str] = None
    ) -> None:
        self.rank = rank

        checkpoint: Optional[Dict[str, Any]] = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path)

            self.audio_configs = AudioProcessorConfig(**checkpoint[CheckpointKey.AUDIO_PROCESSOR])
            self.log_melspec_configs = LogMelSpectrogramConfig(**checkpoint[CheckpointKey.LOG_MELSPECTROGRAM])
            self.hyper_params = W2vBERTConfig(**checkpoint[CheckpointKey.HYPER_PARAMS])
        else:
            self.audio_configs = AudioProcessorConfig(sample_rate=sample_rate)
            self.log_melspec_configs = LogMelSpectrogramConfig(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=pad,
                n_mels=n_mels,
                window_fn=window_fn,
                power=power,
                frame_norm=frame_norm,
                window_norm=window_norm,
                center=center,
                pad_mode=pad_mode,
                slaney_norm=slaney_norm,
                mel_scale=mel_scale
            )
        self.hyper_params = W2vBERTConfig(
            
        )