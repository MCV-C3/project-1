from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F


def get_datasets(
    base_path: str,
    split: int,
    image_size: int = 224
):
    """
    Returns training and test ImageFolder datasets.

    Args:
        base_path (str): Base dataset directory
        split (int): Dataset split index
        image_size (int): Target image resolution

    Returns:
        data_train, data_test
    """

    transform = F.Compose([
        F.ToImage(),
        F.ToDtype(dtype=float, scale=True),
        F.Resize(size=(image_size, image_size)),
    ])

    train_path = f"{base_path}/MIT_small_train_{split}/train"
    test_path = f"{base_path}/MIT_small_train_{split}/test"

    data_train = ImageFolder(train_path, transform=transform)
    data_test = ImageFolder(test_path, transform=transform)

    return data_train, data_test
