import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseBatchRandomDataAugmentation(nn.Module, metaclass=abc.ABCMeta):
    r"""Base class for data augmentations that should be randomly applied to
    elements of a batch. In order to create new data augmentations, override
    this class and implement `apply_augmentation`.

    Args:
        p (float): probability to apply the augmentation to a sample
        return_masks (bool): whether masks indicating to which samples the
            augmentation has been applied must be returned
    """
    def __init__(self, p: Optional[float] = None, return_params: Optional[bool] = None, return_masks: Optional[bool] = None):
        super(BaseBatchRandomDataAugmentation, self).__init__()
        self.p = p
        self.return_params = return_params
        self.return_masks = False if return_params else return_masks

    def _compute_mask(self, length: int, device: torch.device) -> torch.BoolTensor:
        return torch.rand(length, device=device).lt(self.p)

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
    def expand_mid(to_expand: torch.Tensor, target: torch.Tensor):
        batch_size, num_samples = to_expand.size()
        assert target.size(0) == batch_size, \
            f"Both tensors must have the same batch size, got {to_expand.size()} and {target.size()}."
        assert target.size(-1) == num_samples, \
            f"Both tensors must have the same number of samples, got {to_expand.size()} and {target.size()}."
        return to_expand.unsqueeze(1).expand_as(target.view(batch_size, -1, num_samples)).view_as(target)

    @staticmethod
    def randint_sampling_fn(min_value, max_value):
        def sample_randint(*size, **kwargs):
            return torch.randint(min_value, max_value, size, **kwargs)
        return sample_randint

    @staticmethod
    def uniform_sampling_fn(min_value, max_value):
        def sample_uniform(*size, **kwargs):
            tensor = torch.empty(*size, **kwargs)
            tensor.uniform_(min_value, max_value)
            return tensor
        return sample_uniform


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
        augmented_x, params = self.apply_augmentation(x, mask, **kwargs)

        if self.return_params:
            return augmented_x, params

        if self.return_masks:
            return augmented_x, mask

        return augmented_x

    @abc.abstractmethod
    def apply_augmentation(
            self,
            x: torch.Tensor,
            mask: torch.BoolTensor,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply the data augmentation to the samples of the batch that should be modified according to the mask

        Args:
            x (torch.Tensor): input batch of waveforms or spectrograms, shape (batch_size, *)
            mask (torch.BoolTensor): mask indicating which samples should be transformed, shape (batch_size)
            kwargs: Implementation-specific keyword-arguments

        Returns:
            torch.Tensor: batch with randomly transformed samples according to mask
        """
        pass


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

        def apply_augmentation(
                self,
                x: torch.Tensor,
                mask: torch.BoolTensor,
                **kwargs
        ) -> torch.Tensor:
            return torch.where(self.expand_right(mask, x), self.module(x, **kwargs), x)

    return BatchRandomTransform
