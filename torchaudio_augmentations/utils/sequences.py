import torch
from torch.nn.functional import pad


def pad_or_trim(tensor: torch.Tensor, length: int):
    if length > tensor.size(-1):
        return pad(tensor, (0, length - tensor.size(-1)))
    return tensor[..., :length]