import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

class TimeMasking(nn.Module):
    def __init__(self, n_masks: int, mask_param: int, ratio: float, zero_masking: bool = True) -> None:
        super().__init__()
        self.n_masks = n_masks
        self.mask_param = mask_param
        self.ratio = ratio

        self.mask_value: Optional[float] = None
        if zero_masking:
            self.mask_value = 0.0

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, get_indices: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=1, dtype=torch.int32)
        else:
            batch_size, length, _ = x.size()
            lengths = torch.full([batch_size], fill_value=length, dtype=torch.int32, device=x.device)

        mask_value = self.mask_value
        if mask_value is None:
            if attn_mask is None:
                mask_value = x.mean(dim=[1, 2])
            else:
                mask_value = x.masked_fill(attn_mask.unsqueeze(dim=2).logical_not(), value=0.0).sum(dim=[1,2]) / lengths

        max_num_maskings = torch.minimum((lengths * self.ratio).long(), torch.tensor(self.mask_param, device=x.device))

        # Get timestamps
        t = (torch.rand([batch_size, self.n_masks], device=x.device) * max_num_maskings.unsqueeze(1)).long()
        t_0 = (torch.rand([batch_size, self.n_masks], device=x.device) * (lengths - max_num_maskings).unsqueeze(1)).long()
        t_1 = t_0 + t

        # Create Time Mask in time axis
        tmask = torch.arange(end=x.size(1), device=x.device).unsqueeze(1)
        tmask = ((tmask >= t_0.unsqueeze(1)) & (tmask < t_1.unsqueeze(1)))
        tmask = tmask.any(dim=-1)

        return x.masked_fill(tmask.unsqueeze(dim=2), value=mask_value)