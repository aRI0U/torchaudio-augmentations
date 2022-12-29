import importlib


def import_batch_random_data_augmentation(augmentation_name: str, module_name: str):
    def data_augmentation(lazy, *args, **kwargs):
        augmentations_package = "lazy_augmentations" if lazy else "batch_augmentations"
        augmentation_module = importlib.import_module(augmentation_name,augmentations_package + '.' + module_name)
        augmentation = getattr(augmentation_module, augmentation_name)
        return augmentation(*args, **kwargs)
    return data_augmentation
