import abc
from typing import Tuple

import torch
import torch.nn as nn

from torchaudio_augmentations.batch_augmentations.base import BaseBatchRandomDataAugmentation


class BatchRandomDataAugmentation(BaseBatchRandomDataAugmentation):
    def forward(self, x: torch.Tensor, **kwargs):
        r"""Apply the data augmentation to each sample of a batch with probability `p`
        and eventually return the mask indicating to which samples the augmentation
        has been applied on

        Args:
            x (torch.Tensor): batch of waveforms or spectrograms the augmentation
                should be applied on
            kwargs: Implementation-specific keyword-arguments that are directly passed
                to the `apply_augmentation` method

        Returns:
            torch.Tensor: batch with some random elements augmented
            torch.BoolTensor (eventually): mask indicating which samples have been augmented
        """
        mask = self._compute_mask(x.size(0), x.device)
        indices = torch.argwhere(mask)

        if indices.size(0) == 0:
            if self.return_masks:
                return x, mask
            return x

        augmented_samples, params = self.apply_augmentation(x[mask], **kwargs)
        x.scatter_(
            0,
            indices.expand_as(augmented_samples.view(indices.size(0), -1)).view_as(augmented_samples),
            augmented_samples
        )

        if self.return_params:
            augmented_params = torch.empty_like(mask, dtype=torch.float)
            augmented_params.fill_(self.default_param)
            augmented_params.scatter_(0, indices.squeeze(1), params)
            return x, augmented_params

        if self.return_masks:
            return x, mask

        return x

    @abc.abstractmethod
    def apply_augmentation(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply the data augmentation to the samples of the batch that should be modified according to the mask

        Args:
            x (torch.Tensor): input batch of waveforms or spectrograms, shape (batch_size, *)
            kwargs: Implementation-specific keyword-arguments

        Returns:
            torch.Tensor: batch with randomly transformed samples according to mask
        """
        pass

    @property
    def default_param(self):
        return 1


def BatchRandomApply(module: nn.Module):
    r"""Workaround to enable to directly transform classical data augmentations
    to batch random ones. It only works when the data augmentation is identical
    for all samples it should be applied to.

    Args:
        module (nn.Module): data augmentation to be transformed so that it is
            randomly applied

    Returns:
        BatchRandomDataAugmentation class
    """
    class BatchRandomTransform(BatchRandomDataAugmentation):
        def __init__(self, p: float = 0.5, return_masks: bool = False):
            super(BatchRandomTransform, self).__init__(p=p, return_masks=return_masks)
            self.module = module
            self.__class__.__name__ = "BatchRandom" + module.__class__.__name__

        def apply_augmentation(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.module(x)

    return BatchRandomTransform
