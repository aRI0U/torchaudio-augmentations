from functools import partial
import time
import timeit
import traceback

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torchaudio_augmentations.augmentations.delay import Delay
from torchaudio_augmentations.augmentations.gain import Gain
from torchaudio_augmentations.augmentations.polarity_inversion import PolarityInversion
from torchaudio_augmentations.augmentations.reverse import Reverse
from torchaudio.transforms import TimeStretch

from torchaudio_augmentations import BatchRandomDelay, BatchRandomGain, \
    BatchRandomPolarityInversion, BatchRandomReverse, BatchRandomTimeStretch

def sample_uniform(min_value, max_value, size, **kwargs):
    tensor = torch.empty(*size, **kwargs)
    tensor.uniform_(min_value, max_value)
    return tensor


def test_batches_augmentations(
        module: torch.nn.Module,
        batch_module: torch.nn.Module,
        input_shape: tuple = None,
        dtype: torch.dtype = torch.float32,
        gpu: int = 0,
        **kwargs
):
    print(module.__class__.__name__)
    devices = [torch.device("cpu")]
    if gpu is not None:
        devices.append(torch.device(f"cuda:{gpu:d}"))

    for device in devices:
        module.eval()
        batch_module.eval()
        module = module.to(device)
        batch_module = batch_module.to(device)

        # load data
        if input_shape is None:
            input_shape = (batch_size, 2, 48000)
        batch = torch.randn(input_shape, device=device, dtype=dtype)
        if batch.isreal().all():
            batch.clamp_(-1, 1)

        if kwargs == {}:
            t0 = time.time()
            augmented_batch, mask = batch_module(batch)
            t1 = time.time()
            samples_list = [module(sample) if m else sample for sample, m in zip(batch, mask)]
        else:  # TODO: handle values as **kwargs
            for k, v in kwargs.items():
                kwargs[k] = v.to(device)

            # pass through batch_module
            t0 = time.time()
            augmented_batch, mask = batch_module(batch, **kwargs)
            t1 = time.time()

            values = next(iter(kwargs.values()))
            samples_list = [module(sample, value.item()) if m else sample
                            for sample, value, m in zip(batch, values, mask)]

        if batch.ndim >= 3:
            samples_list = [sample.transpose(0, -1) for sample in samples_list]
        augmented_samples = torch.nn.utils.rnn.pad_sequence(samples_list, batch_first=True)
        if batch.ndim >= 3:
            augmented_samples = augmented_samples.transpose(1, -1)

        if augmented_samples.size(-1) >= batch.size(-1):
            augmented_samples = augmented_samples[..., :batch.size(-1)]
        else:
            augmented_samples = F.pad(augmented_samples, (0, batch.size(-1) - augmented_samples.size(-1)))

        # compare results
        try:
            assert_close(augmented_samples, augmented_batch)
        except AssertionError:
            traceback.print_exc()
            diff = (augmented_batch - augmented_samples).abs()
            if kwargs == {}:
                for d in diff:
                    plt.plot(d)
                    plt.show()
            else:
                key, values = next(iter(kwargs.items()))
                # for d, v in zip(diff, values):
                #     plt.plot(d)
                #     plt.title(f"{key} = {v.item():.1f}")
                #     plt.show()
            return
        print(device, f"OK ({t1 - t0:.3f}s).")
    print("Passed.\n")


def benchmark(batch_module, input_shape=(2, 48000), dtype=torch.float32, device=None):
    print(batch_module.__class__.__name__)
    print(f"device: {device}\tlazy: {lazy}")
    batch_sizes = [8, 16, 32, 64, 128, 256]
    probs = [0, 0.25, 0.5, 0.75, 1]

    batch_module = batch_module.to(device)

    results = np.zeros((len(batch_sizes), len(probs)))

    for i, batch_size in enumerate(batch_sizes):
        for j, p in enumerate(probs):
            print(f"batch size: {batch_size}\tp: {p:.02f}", end='\r')
            x = torch.randn(batch_size, *input_shape, device=device, dtype=dtype)
            batch_module.p = p
            if not dtype.is_complex:
                x.clamp_(-1, 1)
            try:
                results[i, j] = timeit.timeit("batch_module(x)", number=100, globals=dict(x=x, batch_module=batch_module)) / 100
            except RuntimeError:
                results[i, j] = np.inf

    return results


batch_size = 67
gpu = 7 if torch.cuda.is_available() else None

# test_batches_augmentations(
#     Delay(16000),
#     BatchRandomDelay(16000, p=0.8, return_masks=True),
#     delays=sample_uniform(200, 500, (batch_size,)),
#     gpu=gpu
# )
#
# test_batches_augmentations(
#     Gain(),
#     BatchRandomGain(p=0.9, return_masks=True),
#     gains_db=sample_uniform(-6., 0., (batch_size,)),
#     gpu=gpu
# )
#
# test_batches_augmentations(
#     PolarityInversion(),
#     BatchRandomPolarityInversion(p=0.8, return_masks=True),
#     gpu=gpu
# )
#
# test_batches_augmentations(
#     Reverse(),
#     BatchRandomReverse(p=0.8, return_masks=True),
#     gpu=gpu
# )
#
# test_batches_augmentations(
#     TimeStretch(n_freq=129),
#     BatchRandomTimeStretch(r_min=0.5, r_max=1.5, n_fft=256, p=0.8, return_masks=True),
#     rates=sample_uniform(0.5, 1.5, (batch_size,)),
#     input_shape=(batch_size, 2, 129, 192),
#     dtype=torch.complex128,
#     gpu=gpu
# )

module_list = [
    partial(BatchRandomDelay, 16000, p=0.8, return_masks=True),
    partial(BatchRandomGain, p=0.9, return_masks=True),
    partial(BatchRandomPolarityInversion, p=0.8, return_masks=True),
    partial(BatchRandomReverse, p=0.8, return_masks=True)
]

for d in ["cpu", "cuda"]:
    device = torch.device(d)
    for lazy in [True, False]:
        for m in module_list:
            module = m(lazy=lazy)
            arr = benchmark(module, device=device)
            np.save(f"time/{'lazy' if lazy else 'normal'}/{module.__class__.__name__}_{device}", arr)
            print(f"    mean: {arr.mean():.3f}"
                  f"    std: {arr.std():.3f}"
                  f"    min: {arr.min():.3f}"
                  f"    max: {arr.max():.3f}"
            )
            print("                           ")

        arr = benchmark(
            BatchRandomTimeStretch(r_min=0.5, r_max=1.5, n_fft=256, p=0.8, return_masks=True, lazy=lazy),
            input_shape=(1, 129, 192),
            dtype=torch.complex64,
            device=device
        )
        np.save(f"time/{'lazy' if lazy else 'normal'}/BatchRandomTimeStretch_{device}", arr)
        print(f"    mean: {arr.mean():.3f}"
              f"    std: {arr.std():.3f}"
              f"    min: {arr.min():.3f}"
              f"    max: {arr.max():.3f}"
        )
        print("                           ")
