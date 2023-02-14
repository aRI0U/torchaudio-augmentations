import importlib
from .batch_augmentations.base import BaseBatchRandomDataAugmentation


class import_batch_random_data_augmentation(BaseBatchRandomDataAugmentation):
    def __init__(self, augmentation_name, module_name):
        super(import_batch_random_data_augmentation, self).__init__()
        self.augmentation_name = augmentation_name
        self.module_name = module_name

    def __call__(self, *args, **kwargs) -> BaseBatchRandomDataAugmentation:
        lazy = kwargs.pop("lazy", False)
        package_name = "lazy" if lazy else "batch"
        augmentation_module = importlib.import_module(f"torchaudio_augmentations.{package_name}_augmentations.{self.module_name}")
        augmentation = getattr(augmentation_module, self.augmentation_name)
        return augmentation(*args, **kwargs)


BatchRandomDelay = import_batch_random_data_augmentation("BatchRandomDelay", "delay")
BatchRandomGain = import_batch_random_data_augmentation("BatchRandomGain", "gain")
BatchRandomNoise = import_batch_random_data_augmentation("BatchRandomNoise", "noise")
BatchRandomPitchShift = import_batch_random_data_augmentation("BatchRandomPitchShift", "pitch_shift")
BatchRandomTimeStretch = import_batch_random_data_augmentation("BatchRandomTimeStretch", "time_stretch")
BatchRandomPolarityInversion = import_batch_random_data_augmentation("BatchRandomPolarityInversion", "misc")
BatchRandomReverse = import_batch_random_data_augmentation("BatchRandomReverse", "misc")
