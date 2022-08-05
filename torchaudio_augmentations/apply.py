# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Sequence

import torch
import torch.nn as nn

from augmentations.base import BatchRandomDataAugmentation


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# TODO: add eventual final transform (log-spec, mel, etc.)
class BatchAudioComposeTransforms(nn.Module):
    def __init__(
            self,
            wave_transforms: Sequence[BatchRandomDataAugmentation] = None,
            spec_transforms: Sequence[BatchRandomDataAugmentation] = None,
            spectrogram_fn: Optional[nn.Module] = None
    ):
        super(BatchAudioComposeTransforms, self).__init__()
        self.wave_transforms = nn.ModuleList(wave_transforms)
        self.spec_transforms = nn.ModuleList(spec_transforms)

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
            x, mask = transform(x)
            masks.append(mask)

        specgrams = self.spectrogram(waveforms)
        x = self.spectrogram(x)

        for transform in self.spec_transforms:
            x, mask = transform(x)
            masks.append(mask)

        if self.power is not None:
            specgrams = specgrams.abs().pow(self.power)
            x = x.abs().pow(self.power)
        if self.normalized:
            specgrams = self.normalize(specgrams)
            x = self.normalize(x)
        return specgrams, x, torch.stack(masks, dim=1)

    @staticmethod
    def normalize(specgrams):
        batch_size = specgrams.size(0)
        flat_specgrams = specgrams.view(batch_size, -1)
        max_values = flat_specgrams.max(dim=1, keepdim=True).clip(min=1e-12)
        return specgrams / max_values.expand_as(flat_specgrams).view_as(specgrams)

