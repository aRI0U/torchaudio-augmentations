from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence


def resample_v1(
        audio_waveforms: torch.Tensor,
        orig_freq: torch.LongTensor,
        new_freq: torch.LongTensor,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interpolation",
        beta: Optional[float] = None,
        dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    gcd = torch.gcd(orig_freq, new_freq)

    orig_freq.div_(gcd, rounding_mode="floor")
    new_freq.div_(gcd, rounding_mode="floor")

    assert lowpass_filter_width > 0
    resampled_list = []

    base_freq = torch.minimum(orig_freq, new_freq).to(dtype).mul(rolloff)

    width = torch.ceil(lowpass_filter_width * orig_freq / base_freq).long()

    for wave, w, of, nf, bf in zip(audio_waveforms, width, orig_freq, new_freq, base_freq):
        kernels = []
        idx = torch.arange(-w, w + of, device=audio_waveforms.device, dtype=dtype)

        for i in range(nf):
            t = (-i / nf + idx / of) * bf
            t.clamp_(-lowpass_filter_width, lowpass_filter_width)

            if resampling_method == "sinc_interpolation":
                t.mul_(torch.pi)
                window = torch.cos(t / (2 * lowpass_filter_width)).pow(2)
            else:
                raise NotImplementedError

            kernel = torch.where(t != 0, torch.sin(t) / t, 1.)
            kernel.mul_(window)
            kernels.append(kernel)

        scale = bf / of
        kernels = torch.stack(kernels).view(nf, 1, -1).mul_(scale).to(audio_waveforms.dtype)

        # pack batch
        shape = wave.size()
        wave = wave.view(-1, shape[-1])

        num_wavs, length = wave.shape
        wave = torch.nn.functional.pad(wave, (w, w + of))
        resampled = torch.nn.functional.conv1d(wave[:, None], kernels, stride=of.item())
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        target_length = torch.ceil(nf * length / of).long()
        resampled = resampled[..., :target_length]

        # unpack batch
        resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
        resampled_list.append(resampled)

    return pad_sequence(resampled_list, batch_first=True)


def resample(
        audio_waveforms: torch.Tensor,
        orig_freq: torch.LongTensor,
        new_freq: torch.LongTensor,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interpolation",
        beta: Optional[float] = None,
        dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    device = audio_waveforms.device

    gcd = torch.gcd(orig_freq, new_freq)

    orig_freq.div_(gcd, rounding_mode="floor")
    new_freq.div_(gcd, rounding_mode="floor")

    assert lowpass_filter_width > 0
    resampled_list = []

    base_freq = torch.minimum(orig_freq, new_freq).to(dtype).mul(rolloff)

    width = torch.ceil(lowpass_filter_width * orig_freq / base_freq).long()

    for wave, w, of, nf, bf in zip(audio_waveforms, width, orig_freq, new_freq, base_freq):
        inv_of = torch.arange(-w, w + of, device=device, dtype=dtype).div(of)
        inv_nf = torch.arange(nf, device=device, dtype=dtype).div(nf)  # -i/nf

        t = inv_of.unsqueeze(0) - inv_nf.unsqueeze(1)
        t.mul_(bf).clamp_(-lowpass_filter_width, lowpass_filter_width)

        if resampling_method == "sinc_interpolation":
            t.mul_(torch.pi)
            window = torch.cos(t / (2 * lowpass_filter_width)).pow(2)
        else:
            raise NotImplementedError

        kernels = torch.where(t != 0, torch.sin(t) / t, 1.)
        kernels.mul_(window)

        scale = bf / of
        kernels = kernels.unsqueeze(1).mul_(scale).to(audio_waveforms.dtype)

        # pack batch
        shape = wave.size()
        wave = wave.view(-1, shape[-1])

        num_wavs, length = wave.shape
        wave = torch.nn.functional.pad(wave, (w, w + of))
        resampled = torch.nn.functional.conv1d(wave[:, None], kernels, stride=of.item())
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        target_length = torch.ceil(nf * length / of).long()
        resampled = resampled[..., :target_length]

        # unpack batch
        resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
        resampled_list.append(resampled)
    print(*[x.size() for x in resampled_list])
    return pad_sequence(resampled_list, batch_first=True)


if __name__ == "__main__":
    import time

    waveform = torch.randn(3, 5).cuda()

    orig_freq = torch.tensor([65, 80, 66], device=waveform.device)
    new_freq = torch.tensor([45, 72, 99], device=waveform.device)

    t1 = time.time()
    k1 = resample_v1(waveform, orig_freq, new_freq)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    k2 = resample(waveform, orig_freq, new_freq)
    t2 = time.time()
    print(t2 - t1)

    for a, b in zip(k1, k2):
        print(a.size(), b.size())
        ok = torch.allclose(a, b)
        if not ok:
            diff = (a - b).squeeze().abs()
            print(a.squeeze())
            print(diff.mean(), diff.max())
            x = 1
