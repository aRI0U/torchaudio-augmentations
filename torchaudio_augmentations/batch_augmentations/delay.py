from typing import Optional

import torch

from .base import BatchRandomDataAugmentation


class BatchRandomDelay(BatchRandomDataAugmentation):
    def __init__(
            self,
            sample_rate: int,
            volume_factor: float = 0.5,
            min_delay: int = 200,
            max_delay: int = 500,
            delay_interval: int = 50,
            p: float = 0.5,
            return_masks: bool = False
    ):
        super(BatchRandomDelay, self).__init__(p=p, return_masks=return_masks)
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor
        self.delay_interval = delay_interval

        self.sample_random_delays = self.randint_sampling_fn(min_delay // delay_interval,
                                                             max_delay // delay_interval + 1)

    def apply_augmentation(
            self,
            audio_waveforms: torch.Tensor,
            mask: torch.BoolTensor,
            delays: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # TODO: make it work for inputs with more than 2 dims, e.g. stereo signals
        batch_size = audio_waveforms.size(0)
        num_samples = audio_waveforms.size(-1)
        device = audio_waveforms.device

        if delays is None:
            delays = self.delay_interval * self.sample_random_delays(batch_size, device=device)

        offsets = self._calc_offset(delays)
        indices = torch.arange(num_samples, device=device).unsqueeze(0) - offsets.unsqueeze(1)
        invalid_indices = torch.logical_or(~mask.unsqueeze(1), indices < 0)
        indices[invalid_indices] = 0

        delayed_signals = audio_waveforms.gather(-1, indices)
        delayed_signals[invalid_indices] = 0

        return audio_waveforms + self.volume_factor * delayed_signals

    def _calc_offset(self, delay_in_ms):
        return (self.sample_rate * delay_in_ms / 1000).long()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import traceback

    import torch
    from torch.testing import assert_allclose, make_tensor


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
            batch = make_tensor((batch_size, 48000), device=device, dtype=torch.float32)
            delays = torch.tensor([276., 120., 468., 321., 2.], device=device)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch, delays=delays)

            # pass through standard module
            # augmented_samples = torch.nn.utils.rnn.pad_sequence(
            #     [module(sample, rate).T if m else sample for sample, rate, m in zip(batch, rates, mask)],
            #     batch_first=True
            # )
            samples_list = [module(sample, delay.item()) if m else sample for sample, delay, m in zip(batch, delays, mask)]
            augmented_samples = torch.stack(samples_list, dim=0)

            # compare results
            try:
                assert_allclose(augmented_samples, augmented_batch)
            except AssertionError:
                traceback.print_exc()
                diff = (augmented_batch - augmented_samples).abs()
                for d, r in zip(diff, delays):
                    plt.plot(d)
                    plt.title(f"delay = {r.item():.1f}")
                    plt.show()
                return
            print(device, "OK.")
        print("Passed.")


    sample_module = Delay(16000)
    batch_module = BatchRandomDelay(16000, p=0.8, return_masks=True)

    test_batches_augmentations(sample_module, batch_module, gpu=5)
