import traceback

import matplotlib.pyplot as plt

import torch
from torchaudio.functional.functional import _stretch_waveform


from torchaudio_augmentations import BatchRandomPitchShift


aug = BatchRandomPitchShift(-4, 5, 16000, n_fft=256, hop_length=64, win_length=256, p=1, return_masks=True)
bs = 1

wave = torch.randn(bs, 64000).clamp_(-1, 1)
n_steps = aug.sample_random_steps(bs)
rates = 2.0 ** (-n_steps.float().div(aug.bins_per_octave))

s1 = []
for w, n in zip(wave, n_steps):
    s = _stretch_waveform(
        w,
        n_steps=n.item(),
        bins_per_octave=aug.bins_per_octave,
        n_fft=aug.n_fft,
        win_length=aug.win_length,
        hop_length=aug.hop_length,
        window=aug.window
    )
    s1.append(s)


s2 = aug._stretch_waveform(wave, rates).unsqueeze(0)

for a, b in zip(s1, s2):
    num_samples = min(a.size(-1), b.size(-1))
    try:
        torch.testing.assert_close(a[..., :num_samples], b[..., :num_samples])
    except AssertionError:
        traceback.print_exc()
        print(a.size(), b.size())
        diff = (a[..., :num_samples] - b[..., :num_samples]).squeeze().numpy()
        plt.plot(diff)
        plt.show()
        plt.plot(a[..., num_samples:].squeeze().numpy())
        plt.plot(b[..., num_samples:].squeeze().numpy())
        plt.show()
