import abc
from typing import Optional

import torch
import torch.nn as nn


class BatchRandomDataAugmentation(nn.Module):
    r"""Base class for data augmentations that should be randomly applied to
    elements of a batch. In order to create new data augmentations, override
    this class and implement `apply_augmentation`.

    Args:
        p (float): probability to apply the augmentation to a sample
        return_masks (bool): whether masks indicating to which samples the
            augmentation has been applied must be returned
    """
    def __init__(self, p: float = 0.5, return_masks: bool = False):
        super(BatchRandomDataAugmentation, self).__init__()
        self.p = p
        self.return_masks = return_masks

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
            return x, mask if self.return_masks else x

        augmented_samples = self.apply_augmentation(x[mask], **kwargs)
        augmented_x = x.scatter(
            0,
            indices.expand_as(augmented_samples.view(indices.size(0), -1)).view_as(augmented_samples),
            augmented_samples
        )

        if self.return_masks:
            return augmented_x, mask

        return augmented_x

    @abc.abstractmethod
    def apply_augmentation(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Apply the data augmentation to the samples of the batch that should be modified according to the mask

        Args:
            x (torch.Tensor): input batch of waveforms or spectrograms, shape (batch_size, *)
            kwargs: Implementation-specific keyword-arguments

        Returns:
            torch.Tensor: batch with randomly transformed samples according to mask
        """
        pass

    def _compute_mask(self, length: int, device: torch.device) -> torch.BoolTensor:
        return torch.rand(length, device=device) < self.p

    @staticmethod
    def expand_right(to_expand: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Expand a tensor so that it has the same shape has a given target

        Args:
            to_expand (torch.Tensor): tensor to expand, shape (B)
            target (torch.Tensor): target tensor, shape (B, *)

        Returns:
            torch.Tensor: expanded tensor, same shape as `target`
        """
        batch_size = to_expand.size(0)
        assert target.size(0) == batch_size, \
            f"Both tensors must have the same batch size, got {to_expand.size()} and {target.size()}."
        return to_expand.unsqueeze(-1).expand_as(target.view(batch_size, -1)).view_as(target)

    @staticmethod
    def randint_sampling_fn(min_value, max_value):
        def sample_randint(*size, **kwargs):
            return torch.randint(min_value, max_value, size, **kwargs)
        return sample_randint

    @staticmethod
    def uniform_sampling_fn(min_value, max_value):
        def sample_uniform(*size, **kwargs):
            return torch.empty(*size, **kwargs).uniform(min_value, max_value)
        return sample_uniform


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

        def apply_augmentation(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.module(x)

    return BatchRandomTransform
