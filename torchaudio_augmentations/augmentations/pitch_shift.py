from typing import Optional

import torch
from torchaudio.functional import pitch_shift


class PitchShift(torch.nn.Module):
    def __init__(
            self,
            sample_rate: int,
            bins_per_octave: int = 12,
            n_fft: int = 512,
            win_length: Optional[int] = None,
            hop_length: Optional[int] = None,
            window: Optional[torch.Tensor] = None
    ):
        super(PitchShift, self).__init__()
        self.sample_rate = sample_rate
        self.bins_per_octave = bins_per_octave
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window

    def forward(self, x, n_steps):
        return pitch_shift(
            x,
            self.sample_rate,
            n_steps,
            bins_per_octave=self.bins_per_octave,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window
        )
