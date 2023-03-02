from typing import Optional

from .base import BatchRandomDataAugmentation


class BatchRandomTanhDistorsion(BatchRandomDataAugmentation):
    r"""See https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/"""
    def __init__(
            self,
            min_distorsion: float = 0.01,
            max_distorsion: float = 0.7,
            p: Optional[bool] = None,
            return_params: Optional[bool] = None,
            return_masks: Optional[bool] = None
    ):
        super(BatchRandomTanhDistorsion, self).__init__(p=p, return_params=return_params, return_masks=return_masks)
        self.sample_random_distorsions = self.uniform_sampling_fn(min_distorsion, max_distorsion)
        raise NotImplementedError
