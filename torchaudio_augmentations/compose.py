from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn

from torchaudio_augmentations.batch_augmentations.base import BaseBatchRandomDataAugmentation


class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ComposeMany(Compose):
    """
    Data augmentation module that transforms any given data example randomly
    resulting in N correlated views of the same example
    """

    def __init__(self, transforms, num_augmented_samples):
        self.transforms = transforms
        self.num_augmented_samples = num_augmented_samples

    def __call__(self, x):
        samples = []
        for _ in range(self.num_augmented_samples):
            samples.append(self.transform(x).unsqueeze(dim=0).clone())
        return torch.cat(samples, dim=0)


# TODO: add eventual final transform (log-spec, mel, etc.)
class BatchAudioComposeTransforms(nn.Module):
    def __init__(
            self,
            wave_transforms: Sequence[BaseBatchRandomDataAugmentation] = None,
            spec_transforms: Sequence[BaseBatchRandomDataAugmentation] = None,
            spectrogram_fn: Optional[nn.Module] = None,
            p: float = 0.5,
            return_masks: bool = False
    ):
        super(BatchAudioComposeTransforms, self).__init__()

        self.wave_transforms = nn.ModuleList(wave_transforms)
        self.spec_transforms = nn.ModuleList(spec_transforms)

        for transform in self.wave_transforms + self.spec_transforms:
            if transform.p is None:
                transform.p = p
            if transform.return_masks is None:
                transform.return_masks = return_masks

        self.spectrogram = spectrogram_fn
        if spectrogram_fn is None:
            assert len(spec_transforms) == 0, \
                "You try to apply transformations to a spectrogram but " \
                "no method for changing the waveform into a spectrogram is defined"
            self.spectrogram = nn.Identity()

        # retrieve attributes from spectrogram since spec_transforms should be applied to complex spectrograms
        self.power, self.spectrogram.power = self.spectrogram.power, None
        self.normalized, self.spectrogram.normalized = self.spectrogram.normalized, False

    def forward(self, x: torch.Tensor):
        waveforms = x.clone()
        masks = []

        for transform in self.wave_transforms:
            if transform.return_masks:
                x, mask = transform(x)
                masks.append(mask)
            else:
                x = transform(x)

        specgrams = self.spectrogram(waveforms)
        x = self.spectrogram(x)

        for transform in self.spec_transforms:
            if transform.return_masks:
                x, mask = transform(x)
                masks.append(mask)
            else:
                x = transform(x)

        if self.power is not None:
            specgrams = specgrams.abs().pow(self.power)
            x = x.abs().pow(self.power)
        if self.normalized:
            specgrams = self.normalize(specgrams)
            x = self.normalize(x)

        if masks:
            return specgrams, x, torch.stack(masks, dim=1)
        return specgrams, x

    @staticmethod
    def normalize(specgrams):
        batch_size = specgrams.size(0)
        flat_specgrams = specgrams.view(batch_size, -1)
        max_values = flat_specgrams.max(dim=1, keepdim=True).clip(min=1e-12)
        return specgrams / max_values.expand_as(flat_specgrams).view_as(specgrams)
