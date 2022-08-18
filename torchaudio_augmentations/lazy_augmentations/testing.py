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
        print(device, "OK")
