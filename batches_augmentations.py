if __name__ == "__main__":
    from torchaudio_augmentations.augmentations.polarity_inversion import PolarityInversion
    from torchaudio_augmentations.batch_augmentations.basic import batch_random2
    from torchaudio_augmentations.batch_augmentations.testing import test_batches_augmentations

    RandomBatchPolarityInversion = batch_random2(PolarityInversion())
    inv = RandomBatchPolarityInversion(p=0.8, return_masks=True)

    assert isinstance(inv, RandomBatchPolarityInversion)

    test_batches_augmentations(PolarityInversion(), inv)
