from typing import Callable, Optional, Tuple

import torch

from torchaudio_augmentations.utils import pad_or_trim, phase_vocoder, resample
from .base import BatchRandomDataAugmentation


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
            n_chunks: int = 1,
            autoscale_n_chunks: float = 1.,
            print_oom_warnings: bool = True,
            p: Optional[float] = None,
            uniform: bool = True,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        r"""

        Args:
            ...  # TODO: finish this
            minimal_gcd (int): increase this value for reducing memory usage, at the cost of possibly
                less precise computations.
            n_chunks (int): number of chunks per batch, increase this value for reducing memory usage
            autoscale_n_chunks: deprecated
            print_oom_warnings (bool): whether user should be warned when there is an OOM
            p (float): probability to apply the transform. If `None`, use the global value instead
            uniform (bool): whether number of steps should be sampled uniformly between `min_steps` and `max_steps`.
                Otherwise, it is sampled from a gaussian
            return_params (bool): whether parameters (frequency ratios) of the transform should be returned
            return_masks (bool): whether mask indicating to which sample the transform has been applied
                should be returned
        """
        super(BatchRandomPitchShift, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_steps = self.randint_sampling_fn(min_steps, max_steps) \
            if uniform else self.gaussint_sampling_fn(min_steps, max_steps)

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

        # OOM handling
        self.oom_counter = 0
        self.steps_counter = 0
        self.n_chunks = n_chunks
        self.autoscale_n_chunks = autoscale_n_chunks
        self.print_oom_warnings = print_oom_warnings

    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None):
        r"""See the parent class for the detailed documentation. In order to save some computation time,
        we sample the number of steps directly in the forward pass and mask inputs whose selected `n_steps`
        is 0.
        """
        self.steps_counter += 1

        mask = self._compute_mask(x.size(0), x.device)

        if params is None:
            params = self.sample_random_steps(x.size(0), device=x.device)
            mask[params == 0] = False
        if not torch.is_floating_point(params):
            params = 2 ** (params.float() / self.bins_per_octave)
        params[~mask] = self.default_param

        if mask.sum() == 0:
            if self.return_params:
                return x, params
            if self.return_masks:
                return x, mask
            return x

        indices = torch.argwhere(mask)

        augmented_samples, params = self.apply_augmentation(x[mask], params[mask])
        x.scatter_(
            0,
            indices.expand_as(augmented_samples.view(indices.size(0), -1)).view_as(augmented_samples),
            augmented_samples
        )

        if self.return_params:
            augmented_params = torch.empty_like(mask, dtype=torch.float)
            augmented_params.fill_(self.default_param)
            augmented_params.scatter_(0, indices.squeeze(1), params)
            return x, augmented_params

        if self.return_masks:
            return x, mask

        return x

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            params: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pitch_shift(audio_waveforms, params), params

    def pitch_shift(self, waveforms: torch.Tensor, freq_ratios: torch.Tensor) -> torch.Tensor:
        r"""Performs batch-wise pitch-shift

        Args:
            waveforms (torch.Tensor): batch of audio waveforms, shape (batch_size, *, num_samples)
            freq_ratios (torch.Tensor): frequency ratios to pitch-shift for each sample, shape (batch_size)

        Returns:
            torch.Tensor: batch of pitch-shifted tensors, same shape as `waveforms`
        """
        original_shape = waveforms.size()

        # pack batch
        waveforms = waveforms.reshape(-1, waveforms.size(-1))
        freq_ratios = freq_ratios.repeat_interleave(waveforms.size(0) // freq_ratios.size(0))

        # time-stretch in Fourier domain
        # rates = 2.0 ** (-n_steps.float().div(self.bins_per_octave))

        wave_stretch = self._stretch_waveform(waveforms, 1 / freq_ratios)
        # print(wave_stretch.size(), original_shape[-1] / rates)
        # resample time-stretched waveforms
        sample_rate = torch.full_like(freq_ratios, self.sample_rate, dtype=torch.long)
        orig_freq = (sample_rate * freq_ratios).long()
        try:
            wave_shift = self._resample(
                wave_stretch,
                orig_freq,
                sample_rate,
                pad_size=waveforms.size(-1),
                n_chunks=self.n_chunks
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                # report error
                self.oom_counter += 1
                if self.print_oom_warnings:
                    print(f"Warning: caught a CUDA OOM ({self.oom_counter}/{self.steps_counter}). "
                          f"If this message appears too often, you should consider reducing the batch size "
                          f"or the `autoscale_n_chunks` parameter.")
                torch.cuda.empty_cache()

                # resample with more chunks
                wave_shift = self._resample(
                    wave_stretch,
                    orig_freq,
                    sample_rate,
                    pad_size=waveforms.size(-1),
                    n_chunks=self.n_chunks+1
                )
            else:
                raise e

        return wave_shift.reshape(original_shape)

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

    def _resample(
            self,
            waveforms: torch.Tensor,
            orig_freq: torch.Tensor,
            new_freq: torch.Tensor,
            pad_size: int,
            n_chunks: int = 1
    ) -> torch.Tensor:
        r"""Calls the `resample` method and then pad/trim the signal to the requested shape.
        Eventually split the inputs into chunks and perform the operation sequentially to avoid GPU OOM.

        Args:
            waveforms (torch.Tensor): audio signals to resample, shape (batch_size, num_samples)
            orig_freq (torch.Tensor): original sample rate, shape (batch_size)
            new_freq (torch.Tensor): target sample rate, shape (batch_size)
            pad_size (int): number of samples for the output
            n_chunks (int): number of chunks to split the inputs in. Increasing this number reduces GPU memory usage
                but increases computation time since inputs are smaller but processed sequentially

        Returns:
            torch.Tensor: resampled audio signals, shape (batch_size, pad_size)
        """
        if n_chunks == 1:
            return pad_or_trim(
                resample(waveforms, orig_freq, new_freq, minimal_gcd=self.minimal_gcd, **self.resampling_kwargs),
                pad_size
            )

        outputs = []
        for w, of, nf in zip(
            torch.chunk(waveforms, n_chunks),
            torch.chunk(orig_freq, n_chunks),
            torch.chunk(new_freq, n_chunks)
        ):
            outputs.append(pad_or_trim(
                resample(w, of, nf, minimal_gcd=self.minimal_gcd, **self.resampling_kwargs),
                pad_size
            ))
        return torch.cat(outputs)

