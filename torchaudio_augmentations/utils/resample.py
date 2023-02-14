import torch
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence
from torch.nn.functional import pad
from torchaudio.functional import resample as torchaudio_resample

from torchaudio_augmentations.utils.sequences import unpack_sequence_it


def resample_v1(
        audio_waveforms: torch.Tensor,
        orig_freq: torch.LongTensor,
        new_freq: torch.LongTensor,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interpolation",
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


def resample_v5(
        audio_waveforms: torch.Tensor,
        orig_freq: torch.LongTensor,
        new_freq: torch.LongTensor,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interpolation",
        minimal_gcd: torch.Tensor = torch.tensor(1),
        dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    device = audio_waveforms.device

    gcd = torch.maximum(torch.gcd(orig_freq, new_freq), minimal_gcd)
    orig_freq.div_(gcd, rounding_mode="floor")
    new_freq.div_(gcd, rounding_mode="floor")

    assert lowpass_filter_width > 0
    resampled_list = []

    base_freq = torch.minimum(orig_freq, new_freq).to(dtype).mul(rolloff)

    width = torch.ceil(lowpass_filter_width * orig_freq / base_freq).long()

    w_max = width.max()
    of_max = orig_freq.max()
    nf_max = new_freq.max()

    # we first pack the timesteps as a PackedSequence object to avoid OOM induced by padding
    # TODO: instead of heuristic use the most efficient packing
    lengths = (orig_freq + 2 * width).cpu()
    timesteps = pack_padded_sequence(
        torch.arange(of_max + 2*w_max, device=device, dtype=dtype).unsqueeze(1).sub(width).div_(orig_freq).t(),
        lengths=lengths,
        batch_first=True,
        enforce_sorted=False
    )
    indices = pad(timesteps.batch_sizes[:-1].cumsum(dim=0).neg(), (1, 0)).repeat_interleave(timesteps.batch_sizes)
    indices.add_(torch.arange(timesteps.data.size(0)))

    times_data = new_freq.to(dtype).pow(-1)[timesteps.sorted_indices][indices]
    times_data = -times_data.unsqueeze(1) * torch.arange(nf_max, device=device, dtype=dtype)

    times_data.add_(timesteps.data.unsqueeze(1)).mul_(base_freq[timesteps.sorted_indices][indices].unsqueeze(1))

    times_data.clamp_(-lowpass_filter_width, lowpass_filter_width)

    if resampling_method == "sinc_interpolation":
        windows = torch.cos(times_data.mul(0.5 * torch.pi / lowpass_filter_width)).pow(2)
    else:
        raise NotImplementedError

    # at this point, we don't need timesteps anymore
    kernels = PackedSequence(
        torch.sinc(times_data).mul_(windows),
        batch_sizes=timesteps.batch_sizes,
        sorted_indices=timesteps.sorted_indices,
        unsorted_indices=timesteps.unsorted_indices
    )

    for wave, w, of, nf, bf, kernel in zip(
            audio_waveforms,
            width,
            orig_freq,
            new_freq,
            base_freq,
            unpack_sequence_it(kernels, lengths)
    ):
        # compute kernel
        kernel = kernel[:, :nf].mul(bf / of).to(wave.dtype).t().unsqueeze(1)

        # pack batch
        shape = wave.size()
        wave = wave.view(-1, shape[-1])

        num_wavs, length = wave.shape
        wave = torch.nn.functional.pad(wave, (w, w + of))
        resampled = torch.nn.functional.conv1d(wave[:, None], kernel, stride=of.item())
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        target_length = torch.ceil(nf * length / of).long()
        resampled = resampled[..., :target_length]

        # unpack batch
        resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
        resampled_list.append(resampled)

    return pad_sequence(resampled_list, batch_first=True)


resample = resample_v5


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    bs = 7
    waveform = torch.randn(bs, 16000).clip(-1, 1).cuda()

    # orig_freq = torch.tensor([65, 80, 66, 64, 81, 31, 46, 60], device=waveform.device)
    # new_freq = torch.tensor([45, 72, 99, 60, 72, 76, 69, 48], device=waveform.device)
    orig_freq = torch.randint(785, 810, (bs,), device=waveform.device)
    new_freq = torch.randint(800, 805, (bs,), device=waveform.device)
    print(orig_freq)
    print(new_freq)

    t1 = time.time()
    k1 = resample_v1(waveform, orig_freq.clone(), new_freq.clone())
    t2 = time.time()
    print(t2 - t1)
    print(k1.size())

    t1 = time.time()
    k2 = pad_sequence([
        torchaudio_resample(wave, of.item(), nf.item())
        for wave, of, nf in zip(waveform, orig_freq, new_freq)
        ],
        batch_first=True
    )
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    k3 = resample_v5(waveform, orig_freq.clone(), new_freq.clone())
    t2 = time.time()
    print(t2 - t1)

    for a, b in zip(k3, k2):
        print(a.size(), b.size())
        if a.size(0) > b.size(0):
            a = a[:b.size(0)]
        ok = torch.allclose(a, b)
        if not ok:
            diff = (a - b).squeeze().abs()
            print(a.squeeze())
            print(b.squeeze())
            print(diff.mean(), diff.max())
            plt.plot(diff.cpu().numpy())
            plt.yscale("log")
            plt.show()
            x = 1
