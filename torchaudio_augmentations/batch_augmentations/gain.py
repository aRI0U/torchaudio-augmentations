from typing import Optional

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomGain(BatchRandomDataAugmentation):
    def __init__(self, min_gain_db: float = -6., max_gain_db: float = 0., p: float = 0.5, return_masks: bool = False):
        super(BatchRandomGain, self).__init__(p=p, return_masks=return_masks)
        self.sample_random_gains = self.uniform_sampling_fn(min_gain_db, max_gain_db)

    def apply_augmentation(self, audio_waveforms: torch.Tensor, gains_db: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = audio_waveforms.size(0)
        device = audio_waveforms.device

        if gains_db is None:
            gains_db = self.sample_random_gains(batch_size, device=device)

        # compute ratios corresponding to gain in dB
        ratios = torch.full((batch_size,), 10, device=device).pow(gains_db / 20)

        return self.expand_right(ratios, audio_waveforms) * audio_waveforms
