from typing import Optional

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomDelay(BatchRandomDataAugmentation):
    def __init__(
            self,
            sample_rate: int,
            volume_factor: float = 0.5,
            min_delay: int = 200,
            max_delay: int = 500,
            delay_interval: int = 50,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomDelay, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor
        self.delay_interval = delay_interval

        self.sample_random_delays = self.randint_sampling_fn(min_delay // delay_interval,
                                                             max_delay // delay_interval + 1)

    def apply_augmentation(self, audio_waveforms: torch.Tensor, delays: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = audio_waveforms.size(0)
        num_samples = audio_waveforms.size(-1)
        device = audio_waveforms.device

        if delays is None:
            delays = self.delay_interval * self.sample_random_delays(batch_size, device=device)

        offsets = self._calc_offset(delays)
        indices = torch.arange(num_samples, device=device).unsqueeze(0) - offsets.unsqueeze(1)
        invalid_indices = indices < 0
        indices.masked_fill_(invalid_indices, 0)

        delayed_signals = audio_waveforms.gather(-1, self.expand_mid(indices, audio_waveforms))
        delayed_signals.masked_fill_(self.expand_mid(invalid_indices, audio_waveforms), 0)

        return audio_waveforms + self.volume_factor * delayed_signals

    def _calc_offset(self, delay_in_ms):
        return (self.sample_rate * delay_in_ms / 1000).long()
