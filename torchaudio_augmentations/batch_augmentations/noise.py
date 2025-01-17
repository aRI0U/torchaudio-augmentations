from typing import Optional, Tuple

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomNoise(BatchRandomDataAugmentation):
    def __init__(
            self,
            min_snr: float = 0.0001,
            max_snr: float = 0.01,
            p: Optional[float] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomNoise, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_snr = self.uniform_sampling_fn(min_snr, max_snr)

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            snr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio_waveforms.size(0)
        device = audio_waveforms.device

        if snr is None:
            snr = self.sample_random_snr(batch_size, device=device)

        noise_std = snr * audio_waveforms.view(batch_size, -1).std(dim=-1)

        # compute ratios corresponding to gain in dB
        noise = self.expand_right(noise_std, audio_waveforms) * torch.randn_like(audio_waveforms)

        return torch.where(
            self.expand_right(mask, audio_waveforms),
            audio_waveforms + noise,
            audio_waveforms
        ), noise_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import traceback

    import torch
    from torch.testing import assert_allclose


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
            batch = torch.randn((batch_size, 48000), device=device, dtype=torch.float32).clamp(-1, 1)
            snr = torch.rand((batch_size,), device=device)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch, snr=snr)
            print(mask)

            # pass through standard module
            samples_list = [module(sample, s.item()) if m else sample for sample, s, m in zip(batch, snr, mask)]
            augmented_samples = torch.stack(samples_list, dim=0)

            # compare results
            try:
                assert_allclose(augmented_samples, augmented_batch)
            except AssertionError:
                traceback.print_exc()
                diff = (augmented_batch - augmented_samples).abs().cpu()
                for d, r in zip(diff, snr):
                    plt.plot(d)
                    plt.title(f"snr = {r.item():.4f}")
                    plt.show()
                return
            print(device, "OK.")
        print("Passed.")


    # sample_module = Noise()
    batch_module = BatchRandomNoise(p=0.8, return_masks=True)

    # test_batches_augmentations(sample_module, batch_module, gpu=5)

