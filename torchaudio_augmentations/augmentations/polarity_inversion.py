import torch


class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio = torch.neg(audio)
        return audio


class BatchRandomPolarityInversion(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(BatchRandomPolarityInversion, self).__init__()
        self.p = p

    def forward(self, audio):
        r"""

        Args:
            audio (torch.Tensor): batch of audio samples, shape (batch_size, *)

        Returns:
            torch.Tensor: batch of augmented audio, samples, same shape as `audio`
            torch.BoolTensor: mask indicating on which elements augmentations were applied,
                shape (batch_size)
        """
        batch_size = len(audio)
        mask = torch.rand(batch_size, device=audio.device) < self.p
        coeffs = torch.where(mask, -1, 1)
        coeffs = coeffs.unsqueeze(1).expand_as(audio.view(batch_size, -1)).view_as(audio)
        return coeffs * audio, mask


if __name__ == "__main__":
    import torch
    from torch.testing import assert_allclose, make_tensor


    def test_batches_augmentations(module: torch.nn.Module, batch_module: torch.nn.Module, gpu=0):
        devices = [torch.device("cpu")]
        if gpu is not None:
            devices.append(torch.device(f"cuda:{gpu:d}"))

        for device in devices:
            module.eval()
            batch_module.eval()
            module = module.to(device)
            batch_module = batch_module.to(device)

            # load data
            batch = make_tensor((5, 19), device=device, dtype=torch.float)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch)

            # pass through standard module
            augmented_samples = torch.stack(
                [module(sample) if m else sample for sample, m in zip(batch, mask)],
                dim=0
            )

            # compare results
            assert_allclose(augmented_samples, augmented_batch)
        print("Passed.")


    sample_module = PolarityInversion()
    batch_module = BatchRandomPolarityInversion(p=0.3)

    test_batches_augmentations(sample_module, batch_module)
