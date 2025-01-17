from typing import Callable, Optional, Tuple

import torch

from torchaudio_augmentations.utils import pad_or_trim, phase_vocoder, resample
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
            lowpass_filter_width: int = 6,
            rolloff: float = 0.99,
            resampling_method: str = "sinc_interpolation",
            minimal_gcd: int = 1,
            resampling_dtype: torch.dtype = torch.float32,
            p: Optional[float] = None,
            return_masks: Optional[bool] = None
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
        self.register_buffer("minimal_gcd", torch.tensor(minimal_gcd), persistent=False)
        self.bins_per_octave = bins_per_octave

        self.resampling_kwargs = dict(
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            resampling_method=resampling_method,
            dtype=resampling_dtype
        )

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            n_steps: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_steps is None:
            n_steps = self.sample_random_steps(audio_waveforms.size(0), device=audio_waveforms.device)

        return torch.where(
            self.expand_right(mask, audio_waveforms),
            self.pitch_shift(audio_waveforms, n_steps),
            audio_waveforms
        ), 2 ** (n_steps / self.bins_per_octave)

    def pitch_shift(self, waveforms: torch.Tensor, n_steps: torch.LongTensor) -> torch.Tensor:
        r"""Performs batch-wise pitch-shift

        Args:
            waveforms (torch.Tensor): batch of audio waveforms, shape (batch_size, *, num_samples)
            n_steps (torch.LongTensor): number of steps to pitch-shift for each sample, shape (batch_size)

        Returns:
            torch.Tensor: batch of pitch-shifted tensors, same shape as `waveforms`
        """
        original_shape = waveforms.size()

        # pack batch
        waveforms = waveforms.reshape(-1, waveforms.size(-1))
        n_steps = n_steps.repeat_interleave(waveforms.size(0) // n_steps.size(0))

        # time-stretch in Fourier domain
        rates = 2.0 ** (-n_steps.float().div(self.bins_per_octave))

        wave_stretch = self._stretch_waveform(waveforms, rates)

        # resample time-stretched waveforms
        sample_rate = torch.full_like(n_steps, self.sample_rate)
        wave_shift = resample(
            wave_stretch,
            (sample_rate / rates).long(),
            sample_rate,
            minimal_gcd=self.minimal_gcd,
            **self.resampling_kwargs
        )
        # print(wave_shift.size())
        return pad_or_trim(wave_shift, waveforms.size(-1)).view(original_shape)

    def _stretch_waveform(self, waveforms: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
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
        return torch.istft(
            spec_stretch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window  # WARNING: len_stretch not defined
        )


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
