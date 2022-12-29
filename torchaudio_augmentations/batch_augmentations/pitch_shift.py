from typing import Callable, Optional

import torch

from torchaudio_augmentations.utils import phase_vocoder, resample
# from .base import BatchRandomDataAugmentation
from torchaudio_augmentations.batch_augmentations.base import BatchRandomDataAugmentation


class BatchRandomPitchShift(BatchRandomDataAugmentation):
    def __init__(
            self,
            min_steps: int,
            max_steps: int,
            sample_rate: int,
            n_fft: int,
            hop_length: Optional[int] = None,
            win_length: Optional[int] = None,
            window: Optional[Callable] = None,
            bins_per_octave: int = 12,
            p: float = 0.5,
            return_masks: bool = False
    ):
        super(BatchRandomPitchShift, self).__init__(p=p, return_masks=return_masks)
        self.sample_random_steps = self.randint_sampling_fn(min_steps, max_steps)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft

        if window is None:
            window = torch.hann_window(window_length=self.win_length)
        self.register_buffer("window", window, persistent=False)

        self.register_buffer(
            "phase_advance",
            torch.linspace(0, torch.pi * self.hop_length, self.n_fft // 2 + 1).unsqueeze(1),
            persistent=False
        )

        self.bins_per_octave = bins_per_octave

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            n_steps: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.sample_random_steps(audio_waveforms.size(0), device=audio_waveforms.device)

        num_samples = audio_waveforms.size(-1)

        return torch.where(
            self.expand_right(mask, audio_waveforms),
            self.pitch_shift(audio_waveforms, n_steps)[..., :num_samples],
            audio_waveforms
        )

    def pitch_shift(self, waveforms: torch.Tensor, n_steps: torch.LongTensor) -> torch.Tensor:
        # pack batch
        waveforms = waveforms.reshape(-1, waveforms.size(-1))

        rates = 2.0 ** (-n_steps.float().div(self.bins_per_octave))
        specgrams = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        spec_stretch = phase_vocoder(specgrams, rates, self.phase_advance)
        wave_stretch = torch.istft(
            spec_stretch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        sample_rate = torch.full_like(n_steps, self.sample_rate)
        wave_shift = resample(wave_stretch, (sample_rate / rates).long(), sample_rate)
        return wave_shift


if __name__ == "__main__":
    import time
    import torchaudio

    wave1, sr = torchaudio.load("example.wav")
    wave2, _ = torchaudio.load("example2.wav")
    waveform = torch.cat((wave1, wave2))
    print(waveform.size())

    print(torch.stft(waveform, n_fft=512))

    n_fft = 512

    ps = BatchRandomPitchShift(-5, 5, sr, n_fft=n_fft, p=1).to(waveform.device)

    steps = torch.tensor([-2, 0, 4], dtype=torch.long, device=waveform.device)
    t1 = time.time()
    shifted = ps(waveform, n_steps=steps)
    t2 = time.time()
    print(t2 - t1)

