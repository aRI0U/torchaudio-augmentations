from torchaudio_augmentations.batch_augmentations.base import BatchRandomApply
from torchaudio_augmentations.augmentations.polarity_inversion import PolarityInversion
from torchaudio_augmentations.augmentations.reverse import Reverse


BatchRandomPolarityInversion = BatchRandomApply(PolarityInversion())
BatchRandomReverse = BatchRandomApply(Reverse())
