from typing import Optional, Union

import torch

from .base import BatchRandomDataAugmentation


def batch_phase_vocoder(
        complex_specgrams: torch.Tensor,
        rate: Union[float, torch.Tensor],
        phase_advance: torch.Tensor,
        pad_mode: Union[int,str] = "same"
) -> torch.Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.

    Args:
        complex_specgrams (Tensor):
            A tensor of dimension `(batch_size, freq, num_frame)` with complex dtype.
        rate (torch.Tensor): Per-sample speed-up factor, shape (batch_size)
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of `(freq, 1)`

    Returns:
        Tensor:
            Stretched spectrogram. The resulting tensor is of the same dtype as the input
            spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.

    Example
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time)
        >>> complex_specgrams = torch.randn(2, freq, 300, dtype=torch.cfloat)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, torch.pi * hop_length, freq)[..., None]
        >>> x = batch_phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231])
    """
    # TODO: handle multi-channel spectrograms properly
    batch_size = complex_specgrams.size(0)
    num_freqs = complex_specgrams.size(-2)
    num_timesteps = complex_specgrams.size(-1)
    device = complex_specgrams.device

    if isinstance(rate, float):
        rate = torch.empty(batch_size, device=device).fill(rate)

    if torch.allclose(rate, torch.ones(batch_size, device=device)):
        return complex_specgrams

    # compute the actual length of each time-stretched sample
    lengths = torch.ceil(num_timesteps / rate)
    if pad_mode == "same":
        lengths = lengths.clip(max=num_timesteps)
    else:
        raise NotImplementedError

    # Figures out the corresponding real dtype, i.e. complex128 -> float64, complex64 -> float32
    # Note torch.real is a view, so it does not incur any memory copy.
    real_dtype = torch.real(complex_specgrams).dtype
    indices = torch.arange(0, lengths.max(), device=device, dtype=real_dtype)
    time_steps = indices.repeat(batch_size, 1) * rate.unsqueeze(-1)

    # mask invalid time steps
    invalid_time_steps = time_steps >= num_timesteps
    time_steps[invalid_time_steps] = 0

    time_steps = time_steps.unsqueeze(1)  # shape: (batch_size, 1, num_timesteps / min_rate)

    alphas = time_steps % 1.0
    phase_0 = complex_specgrams[..., :1].angle()

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 2])

    # (new_bins, freq, 2)
    # complex_specgrams_0 = complex_specgrams.index_select(-1, time_steps.long())
    # complex_specgrams_1 = complex_specgrams.index_select(-1, (time_steps + 1).long())
    time_steps = time_steps.long().expand(-1, num_freqs, -1)
    complex_specgrams_0 = complex_specgrams.gather(-1, time_steps)
    complex_specgrams_1 = complex_specgrams.gather(-1, time_steps + 1)

    angle_0 = complex_specgrams_0.angle()
    angle_1 = complex_specgrams_1.angle()

    norm_0 = complex_specgrams_0.abs()
    norm_1 = complex_specgrams_1.abs()

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * torch.pi * torch.round(phase / (2 * torch.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    complex_specgrams_stretch = torch.polar(mag, phase_acc)

    invalid_time_steps = invalid_time_steps.unsqueeze(1).expand_as(complex_specgrams_stretch)
    complex_specgrams_stretch[invalid_time_steps] = 0

    return complex_specgrams_stretch


class BatchRandomTimeStretch(BatchRandomDataAugmentation):
    def __init__(
            self,
            r_min: float,
            r_max: float,
            n_fft: int,
            hop_length: Optional[int] = None,
            p: float = 0.5,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomTimeStretch, self).__init__(p=p, return_masks=return_masks)

        hop_length = hop_length if hop_length is not None else n_fft // 2
        n_freq = n_fft // 2 + 1
        self.register_buffer("phase_advance", torch.linspace(0, torch.pi * hop_length, n_freq).unsqueeze(1))

    def apply_augmentation(self, complex_specgrams: torch.Tensor, rates: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = len(complex_specgrams)
        if rates is not None:
            rates = self.sample_random_rates(batch_size)

        return batch_phase_vocoder(complex_specgrams, rates, self.phase_advance)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import traceback

    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    from torch.testing import assert_allclose, make_tensor
    from torchaudio.transforms import TimeStretch


    def test_batches_augmentations(module: torch.nn.Module, batch_module: torch.nn.Module, gpu=0):
        batch_size = 5
        devices = [torch.device("cpu")]
        if gpu is not None:
            devices.append(torch.device(f"cuda:{gpu:d}"))

        for device in devices:
            module.eval()
            batch_module.eval()
            module = module.to(device)
            batch_module = batch_module.to(device)

            # load data
            batch = make_tensor((batch_size, 129, 192), device=device, dtype=torch.complex128).pow(2)
            rates = torch.tensor([1.1296, 1.1464, 0.6710, 0.8516, 0.5487], device=device)
            # rates = torch.tensor([1., 1., 1., 0.7, 1.4], device=device)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch, rates=rates)

            # pass through standard module
            # augmented_samples = torch.nn.utils.rnn.pad_sequence(
            #     [module(sample, rate).T if m else sample for sample, rate, m in zip(batch, rates, mask)],
            #     batch_first=True
            # )
            samples_list = [module(sample, rate).T if m else sample.T for sample, rate, m in zip(batch, rates, mask)]
            augmented_samples = pad_sequence(samples_list, batch_first=True).transpose(1, 2)
            max_timesteps = augmented_samples.size(-1)
            if max_timesteps < 192:
                augmented_samples = F.pad(augmented_samples, (0, (192 - max_timesteps)))
            else:
                augmented_samples = augmented_samples[..., :192]

            # compare results
            try:
                assert_allclose(augmented_samples, augmented_batch)
            except AssertionError:
                traceback.print_exc()
                diff = (augmented_batch - augmented_samples).abs()
                for d, r in zip(diff, rates):
                    plt.imshow(d, cmap="inferno")
                    plt.title(f"rate = {r.item():.3f}")
                    plt.colorbar()
                    plt.show()
                return
            print(device, "OK.")
        print("Passed.")


    sample_module = TimeStretch(n_freq=129)
    batch_module = BatchRandomTimeStretch(p=0.8, r_min=0.5, r_max=1.5, n_fft=256)

    test_batches_augmentations(sample_module, batch_module)
