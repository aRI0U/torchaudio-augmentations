from typing import Optional, Tuple

import torch

from .base import BatchRandomDataAugmentation
from torchaudio_augmentations.utils import phase_vocoder


class BatchRandomTimeStretch(BatchRandomDataAugmentation):
    def __init__(
            self,
            r_min: float,
            r_max: float,
            n_fft: int,
            hop_length: Optional[int] = None,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomTimeStretch, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_rates = self.uniform_sampling_fn(r_min, r_max)

        hop_length = hop_length if hop_length is not None else n_fft // 2
        n_freq = n_fft // 2 + 1
        self.register_buffer("phase_advance", torch.linspace(0, torch.pi * hop_length, n_freq).unsqueeze(1))

    def apply_augmentation(
            self,
            complex_specgrams: torch.Tensor,
            rates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rates is None:
            rates = self.sample_random_rates(complex_specgrams.size(0), device=complex_specgrams.device)

        return phase_vocoder(
            complex_specgrams, rates, self.phase_advance, pad_length=complex_specgrams.size(-1)
        ).contiguous(), rates


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
                for i, r in enumerate(rates):
                    plt.subplot(1, 3, 1)
                    plt.imshow(augmented_batch[i])
                    plt.subplot(1, 3, 2)
                    plt.imshow(augmented_samples[i])
                    plt.subplot(1, 3, 3)
                    plt.imshow(diff[i], cmap="inferno")
                    plt.title(f"rate = {r.item():.3f}")
                    plt.colorbar()
                    plt.show()
                return
            print(device, "OK.")
        print("Passed.")


    sample_module = TimeStretch(n_freq=129)
    batch_module = BatchRandomTimeStretch(p=0.8, r_min=0.5, r_max=1.5, n_fft=256)

    test_batches_augmentations(sample_module, batch_module)
