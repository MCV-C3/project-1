import torchvision.transforms as T

def get_train_aug():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.ToTensor(),
        T.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
