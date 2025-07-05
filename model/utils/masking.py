import torch

def sample_mask(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, context_length = mask.size()
    feature_length = features.size(1)

    num_samples = context_length // feature_length
    extra = context_length - num_samples * feature_length
    
    mask = mask[:, :context_length - extra]
    mask = mask.view([batch_size, feature_length, num_samples]).all(dim=2)
    return mask

def generate_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    return lengths.unsqueeze(dim=1) > torch.arange(lengths.max().item()).unsqueeze(dim=0)