from typing import Optional

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomGain(BatchRandomDataAugmentation):
    def __init__(
            self,
            min_gain_db: float = -6.,
            max_gain_db: float = 0.,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomGain, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_gains = self.uniform_sampling_fn(min_gain_db, max_gain_db)

    def apply_augmentation(self, audio_waveforms: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = audio_waveforms.size(0)
        device = audio_waveforms.device

        if params is None:
            params = self.sample_random_gains(batch_size, device=device)

        # compute ratios corresponding to gain in dB
        ratios = torch.full((batch_size,), 10, device=device).pow(params / 20)

        return self.expand_right(ratios, audio_waveforms) * audio_waveforms
