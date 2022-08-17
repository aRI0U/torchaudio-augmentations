import torch
from torchaudio.functional import pitch_shift


class PitchShift(torch.nn.Module):
    def __init__(self, n_fft, sample_rate):
        super(PitchShift, self).__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate

    def forward(self, x, n_steps):
        return pitch_shift(x, self.sample_rate, n_steps)
