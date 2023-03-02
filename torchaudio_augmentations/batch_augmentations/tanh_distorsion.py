from typing import Optional, Tuple

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomTanhDistorsion(BatchRandomDataAugmentation):
    r"""See https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/"""

    def __init__(
            self,
            min_distorsion: float = 0.01,
            max_distorsion: float = 0.7,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomTanhDistorsion, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_distorsions = self.uniform_sampling_fn(min_distorsion, max_distorsion)
        assert p == 1, "For now only supports p=1"

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            params=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio_waveforms.size(0)
        device = audio_waveforms.device
        assert audio_waveforms.ndim == 2

        if params is None:
            params = self.sample_random_distorsions(batch_size, device=device)

        q = 1 - 0.99 * params
        threshold = torch.quantile(audio_waveforms.abs(), q, dim=1).diag()
        gain_factor = 0.5 / (threshold + 1e-6)

        # Distort the audio
        distorted_samples = torch.tanh(self.expand_right(gain_factor, audio_waveforms) * audio_waveforms)

        # Scale the output so its loudness matches the input
        rms_before = audio_waveforms.pow(2).mean(dim=-1).sqrt()
        if rms_before > 1e-9:
            rms_after = distorted_samples.pow(2).mean(dim=-1).sqrt()
            post_gain = rms_before / rms_after
            distorted_samples.mul_(self.expand_right(post_gain, distorted_samples))

        return distorted_samples, params

