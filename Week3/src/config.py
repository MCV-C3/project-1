import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "seed": 42,
    "model": "resnet18"
}
