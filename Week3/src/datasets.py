import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

def get_dataloaders(data_dir, batch_size):

    tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    ds = ImageFolder(data_dir, transform=tf)
    y = [l for _,l in ds]

    train_idx, val_idx = train_test_split(
        range(len(ds)),
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        ds.classes
    )
