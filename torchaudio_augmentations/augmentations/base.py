import abc

import torch
import torch.nn as nn


class BatchRandomDataAugmentation(nn.Module):
    def __init__(self, p: float = 0.5, return_masks: bool = False):
        super(BatchRandomDataAugmentation, self).__init__()
        self.p = p
        self.return_masks = return_masks

    def forward(self, x: torch.Tensor, **kwargs):
        mask = self._compute_mask(x.size(0), x.device)
        augmented_x = self._apply_augmentation(x, mask, **kwargs)
        if self.return_masks:
            return augmented_x, mask
        return augmented_x

    @abc.abstractmethod
    def _apply_augmentation(
            self,
            x: torch.Tensor,
            mask: torch.BoolTensor,
            **kwargs
    ) -> torch.Tensor:
        pass

    def _compute_mask(self, length: int, device: torch.device) -> torch.BoolTensor:
        return torch.rand(length, device=device) < self.p

    @staticmethod
    def uniform_sampling_fn(min_value, max_value):
        def sample_uniform(*size, **kwargs):
            return torch.empty(*size, **kwargs).uniform(min_value, max_value)
        return sample_uniform