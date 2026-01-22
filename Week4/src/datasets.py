import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, img_size=128, batch_size=32, augment=False):
    train_tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
    if augment:
        train_tfms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ]

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                    transform=transforms.Compose(train_tfms))
    # Use test as validation since no val folder exists
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                  transform=val_tfms)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                   transform=val_tfms)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        train_ds.classes
    )
