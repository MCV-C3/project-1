from typing import Optional
from kornia import augmentation as aug


def get_augmentation(name: Optional[str]):
    """
    Returns a SINGLE augmentation module.
    This ensures causal interpretation of augmentation effects.
    """

    if name is None:
        return None

    if name == "flip":
        return aug.AugmentationSequential(
            aug.RandomHorizontalFlip(p=0.5)
        )

    if name == "rotation":
        return aug.AugmentationSequential(
            aug.RandomRotation(degrees=15)
        )

    if name == "color":
        return aug.AugmentationSequential(
            aug.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        )

    if name == "blur":
        return aug.AugmentationSequential(
            aug.RandomGaussianBlur(
                kernel_size=5,
                sigma=(0.1, 0.6)
            )
        )

    raise ValueError(f"Unknown augmentation type: {name}")
