import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class GumbelQuantizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        num_vars: int,
        vq_dim: int,
        temperature: float = 2.0,
        weight_proj_depth: int = 1,
        weight_proj_factor: int = 1,
        activation: str = 'gelu',
        hard: bool = True,
        std: float = 0.0
    ) -> None:
        super().__init__()
        assert vq_dim % num_groups == 0

        self.num_groups = num_groups
        self.num_vars = num_vars
        self.num_samples = num_groups * num_vars
        self.hard = hard
        self.vq_dim = vq_dim

        self.codebook = nn.Parameter(torch.randn(1, self.num_groups, num_vars, vq_dim // num_groups))
        if std == 0:
            nn.init.uniform_(self.codebook)
        else:
            nn.init.normal_(self.codebook)

        if weight_proj_depth > 1:
            blocks = []
            inner_dim = input_dim * weight_proj_factor
            if activation == 'gelu':
                activation_fn = nn.GELU()
            elif activation == 'silu':
                activation_fn = nn.SiLU()
            else:
                activation_fn = nn.ReLU()

            for i in range(weight_proj_depth):
                blocks.append(
                    nn.Sequential(
                        nn.Linear(in_features=input_dim if i == 0 else inner_dim, out_features=inner_dim),
                        activation_fn
                    )
                )
            
            self.weight_proj = nn.Sequential(*blocks, nn.Linear(inner_dim, self.num_samples))
        else:
            self.weight_proj = nn.Linear(in_features=input_dim, out_features=self.num_samples)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        self.max_temperature, self.min_temperature, self.temperature_decay = temperature
        self.current_temperature = self.max_temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.size()

        x = self.weight_proj(x)
        x = x.view([batch_size * length, self.num_groups, self.num_vars])

        avg_prob = F.softmax(x, dim=2).mean(dim=0) # [batch_size * length, num_groups]
        entropy = -torch.sum(avg_prob * torch.log(avg_prob + 1e-7), dim=1)
        prob_perplexity = torch.exp(entropy).sum()
        
        x = F.gumbel_softmax(x, tau=self.current_temperature, hard=self.hard).type(x.dtype)
        x = x.view([batch_size * length, self.num_samples])

        discreted_ids = x.view([batch_size, length, self.num_groups, self.num_vars]).argmax(dim=3).detach()

        x = x.view([batch_size * length, self.num_groups, self.num_vars])
        x = x.unsqueeze(dim=3) * self.codebook
        x = x.sum(dim=2)
        x = x.view([batch_size, length, self.vq_dim])

        return x, discreted_ids, prob_perplexity