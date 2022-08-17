import matplotlib.pyplot as plt
import traceback

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torchaudio_augmentations.augmentations.delay import Delay
from torchaudio_augmentations.augmentations.gain import Gain
from torchaudio_augmentations.augmentations.polarity_inversion import PolarityInversion
from torchaudio_augmentations.augmentations.reverse import Reverse
from torchaudio.transforms import TimeStretch

from torchaudio_augmentations.batch_augmentations.delay import BatchRandomDelay
from torchaudio_augmentations.batch_augmentations.gain import BatchRandomGain
from torchaudio_augmentations.batch_augmentations.misc import BatchRandomPolarityInversion, BatchRandomReverse
from torchaudio_augmentations.batch_augmentations.time_stretch import BatchRandomTimeStretch


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
            augmented_batch, mask = batch_module(batch)
            samples_list = [module(sample) if m else sample for sample, m in zip(batch, mask)]
        else:  # TODO: handle values as **kwargs
            for k, v in kwargs.items():
                kwargs[k] = v.to(device)

            # pass through batch_module
            augmented_batch, mask = batch_module(batch, **kwargs)

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
        print(device, "OK.")
    print("Passed.\n")


batch_size = 11
gpu = 0 if torch.cuda.is_available() else None

test_batches_augmentations(
    Delay(16000),
    BatchRandomDelay(16000, p=0.8, return_masks=True),
    delays=sample_uniform(200, 500, (batch_size,)),
    gpu=gpu
)

test_batches_augmentations(
    Gain(),
    BatchRandomGain(p=0.9, return_masks=True),
    gains_db=sample_uniform(-6., 0., (batch_size,)),
    gpu=gpu
)

test_batches_augmentations(
    PolarityInversion(),
    BatchRandomPolarityInversion(p=0.8, return_masks=True),
    gpu=gpu
)

test_batches_augmentations(
    Reverse(),
    BatchRandomReverse(p=0.8, return_masks=True),
    gpu=gpu
)

test_batches_augmentations(
    TimeStretch(n_freq=129),
    BatchRandomTimeStretch(r_min=0.5, r_max=1.5, n_fft=256, p=0.8, return_masks=True),
    rates=sample_uniform(0.5, 1.5, (batch_size,)),
    input_shape=(batch_size, 2, 129, 192),
    dtype=torch.complex64,
    gpu=gpu
)
