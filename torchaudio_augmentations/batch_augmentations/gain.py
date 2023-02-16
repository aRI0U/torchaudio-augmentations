from typing import Optional, Tuple

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomGain(BatchRandomDataAugmentation):
    def __init__(
            self,
            min_gain_db: float = -6.,
            max_gain_db: float = 0.,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomGain, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_gains = self.uniform_sampling_fn(min_gain_db, max_gain_db)

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            gains_db: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio_waveforms.size(0)
        device = audio_waveforms.device

        if gains_db is None:
            gains_db = self.sample_random_gains(batch_size, device=device)

        # compute ratios corresponding to gain in dB
        ratios = torch.full((batch_size,), 10, device=device).pow(gains_db / 20)

        return torch.where(
            self.expand_right(mask, audio_waveforms),
            self.expand_right(ratios, audio_waveforms) * audio_waveforms,
            audio_waveforms
        ), gains_db


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
            gains = torch.tensor([-1.3, -2.5, 0., -3., -5.7], device=device)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch, gains_db=gains)

            # pass through standard module
            samples_list = [module(sample, delay.item()) if m else sample for sample, delay, m in zip(batch, gains, mask)]
            augmented_samples = torch.stack(samples_list, dim=0)

            # compare results
            try:
                assert_allclose(augmented_samples, augmented_batch)
            except AssertionError:
                traceback.print_exc()
                diff = (augmented_batch - augmented_samples).abs().cpu()
                for d, r in zip(diff, gains):
                    plt.plot(d)
                    plt.title(f"gain = {r.item():.1f} dB")
                    plt.show()
                return
            print(device, "OK.")
        print("Passed.")


    sample_module = Gain()
    batch_module = BatchRandomGain(p=0.8, return_masks=True)

    test_batches_augmentations(sample_module, batch_module, gpu=5)
