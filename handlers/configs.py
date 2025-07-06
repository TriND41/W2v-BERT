from dataclasses import dataclass
from typing import Optional

@dataclass
class W2vBERTConfig:
    n_contrastive_blocks: int = 6
    n_mlm_blocks: int = 6
    in_channels: int = 80
    hidden_channels: int = 512