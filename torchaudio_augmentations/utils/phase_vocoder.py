from typing import Optional

import torch

from torchaudio_augmentations.utils.sequences import pad_or_trim


def expand_mid(to_expand: torch.Tensor, target: torch.Tensor):
    batch_size, num_samples = to_expand.size()
    assert target.size(0) == batch_size, \
        f"Both tensors must have the same batch size, got {to_expand.size()} and {target.size()}."
    assert target.size(-1) == num_samples, \
        f"Both tensors must have the same number of samples, got {to_expand.size()} and {target.size()}."
    return to_expand.unsqueeze(1).expand_as(target.view(batch_size, -1, num_samples)).view_as(target)


def phase_vocoder_v1(
        complex_specgrams: torch.Tensor,
        rate: torch.Tensor,
        phase_advance: torch.Tensor
) -> torch.Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.

    Args:
        complex_specgrams (Tensor):
            A tensor of dimension `(batch_size, *, freq, num_frame)` with complex dtype.
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
    num_timesteps = complex_specgrams.size(-1)
    device = complex_specgrams.device

    # Figures out the corresponding real dtype, i.e. complex128 -> float64, complex64 -> float32
    # Note torch.real is a view, so it does not incur any memory copy.
    real_dtype = torch.real(complex_specgrams).dtype
    indices = torch.arange(0, num_timesteps, device=device, dtype=real_dtype)
    time_steps = indices.repeat(batch_size, 1) * rate.unsqueeze(-1)

    # mask invalid time steps
    invalid_time_steps = time_steps >= num_timesteps
    time_steps.masked_fill_(invalid_time_steps, 0)

    time_steps = expand_mid(time_steps, complex_specgrams)

    alphas = time_steps % 1.0
    phase_0 = complex_specgrams[..., :1].angle()

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 2])

    # (new_bins, freq, 2)
    # complex_specgrams_0 = complex_specgrams.index_select(-1, time_steps.long())
    # complex_specgrams_1 = complex_specgrams.index_select(-1, (time_steps + 1).long())
    time_steps = time_steps.long()
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
    complex_specgrams_stretch.masked_fill_(
        expand_mid(invalid_time_steps, complex_specgrams_stretch),
        0
    )

    return complex_specgrams_stretch


def phase_vocoder(
        complex_specgrams: torch.Tensor,
        rate: torch.Tensor,
        phase_advance: torch.Tensor,
        pad_length: Optional[int] = None
) -> torch.Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.

    Args:
        complex_specgrams (Tensor):
            A tensor of dimension `(batch_size, *, freq, num_frame)` with complex dtype.
        rate (torch.Tensor): Per-sample speed-up factor, shape (batch_size)
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of `(freq, 1)`
        pad_length (int): Length to eventually pad or trim the signal, left unchanged if `None` is given.
            Default: `None`

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
    num_samples = complex_specgrams.size(-1)
    num_timesteps = torch.ceil(num_samples / rate)
    device = complex_specgrams.device

    # Figures out the corresponding real dtype, i.e. complex128 -> float64, complex64 -> float32
    # Note torch.real is a view, so it does not incur any memory copy.
    real_dtype = torch.real(complex_specgrams).dtype
    indices = torch.arange(0, num_timesteps.max(), device=device, dtype=real_dtype)
    time_steps = indices.unsqueeze(0) * rate.unsqueeze(1)  # time_steps[i, j] = rate[i] * indices[j]

    # mask invalid time steps
    invalid_time_steps = time_steps.ge(num_samples)
    time_steps.masked_fill_(invalid_time_steps, 0)

    time_steps = time_steps.unsqueeze(1)\
        .expand(batch_size, num_freqs, time_steps.size(-1))\
        .view(*complex_specgrams.shape[:-1], time_steps.size(-1))

    alphas = time_steps % 1.0
    phase_0 = complex_specgrams[..., :1].angle()

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 2])

    # (new_bins, freq, 2)
    # complex_specgrams_0 = complex_specgrams.index_select(-1, time_steps.long())
    # complex_specgrams_1 = complex_specgrams.index_select(-1, (time_steps + 1).long())
    time_steps = time_steps.long()
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
    complex_specgrams_stretch.masked_fill_(
        expand_mid(invalid_time_steps, complex_specgrams_stretch),
        0
    )

    if pad_length is not None:
        return pad_or_trim(complex_specgrams_stretch, length=pad_length)
    return complex_specgrams_stretch


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    from torchvision.utils import make_grid

    def plottable(batch: torch.Tensor):
        return make_grid(batch.abs()).permute(1, 2, 0).cpu().numpy()

    bs = 8
    n_freq = 19
    n_sample = 11

    specgrams = torch.randn(bs, 1, n_freq, n_sample, dtype=torch.complex64).cuda()

    rates = torch.rand(bs, device=specgrams.device) + 0.5
    phase = torch.linspace(0, torch.pi * n_freq - 1, n_freq, device=specgrams.device).unsqueeze(1)

    t1 = time.time()
    a = phase_vocoder_v1(specgrams, rates, phase)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    b = phase_vocoder(specgrams, rates, phase)
    t2 = time.time()
    print(t2 - t1)

    print(torch.allclose(a, b[..., :n_sample]))

    plt.imshow(plottable(specgrams))
    plt.title("specgrams")
    plt.show()

    plt.imshow(plottable(a))
    plt.title("v1")
    plt.show()

    plt.imshow(plottable(b))
    plt.title("v2")
    plt.show()




