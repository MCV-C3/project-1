import wandb
from src.datasets import get_dataloaders
from src.models import build_model
from src.train_core import train_model
from src.config import DEVICE, BASE_CONFIG
from src.utils import set_seed
import matplotlib.pyplot as plt
import os

def main():

    wandb.init(project="cnn_sweep_project")

    cfg = BASE_CONFIG.copy()
    cfg.update(dict(wandb.config))

    set_seed(cfg["seed"])

    train_loader, val_loader, classes = get_dataloaders(
        "data/MIT_small_train_1/train",
        cfg["batch_size"]
    )

    model = build_model(cfg["model"], len(classes), cfg["dropout"])

    model, hist, best_acc = train_model(
        model, train_loader, val_loader, cfg, DEVICE
    )

    wandb.log({"best_val_acc":best_acc})

    for e in range(len(hist["val_acc"])):
        wandb.log({
            "epoch":e,
            "train_loss":hist["train_loss"][e],
            "val_loss":hist["val_loss"][e],
            "val_acc":hist["val_acc"][e]
        })



if __name__ == "__main__":
    main()
