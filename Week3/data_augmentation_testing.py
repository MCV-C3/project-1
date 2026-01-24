import copy
from enum import auto
import time
from typing import *
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, WraperModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
from kornia import augmentation as aug


import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

import wandb

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from main import load_data_on_gpu, test, train,train_run, unfreeze_layers


def main():

    wandb.login()

    project = "C3-Week3"


    config = {
        'epochs' : 2_000,
        'lr' : 0.0001,
        'batch_size' : 256,
        'unfreeze' : "All",
        'weight_decay': 0.0001,
    }
        

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    base_path = "/home/msiau/data/tmp/jventosa/2425"

    choosen_split = 1

    data_train = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/train", transform=transformation)
    data_test = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/test", transform=transformation) 

    train_loader = load_data_on_gpu(data_train,device=device,batch_size=config["batch_size"])
    test_loader = load_data_on_gpu(data_test,device=device,batch_size=128)

    C, H, W = np.array(data_train[0][0]).shape

    

    model = WraperModel(num_classes=8, feature_extraction=True,batch_norm=False,dropout=True,dropout_prob=0.5,classifier_type="FCN")#SimpleModel(input_d=C*H*W, hidden_d=300, output_d=8)
    

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    num_epochs = config["epochs"]


    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_accuracy = 0.0

    epochs_since_improvement = 0

    best_epoch = 0
    
    run_name = f"Agumentations"



    with wandb.init(project=project) as run:

        augmentations = aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=run.config.hor_flip),
        aug.RandomRotation(run.config.ran_rot),
        aug.RandomVerticalFlip(p=run.config.ver_flip),
        aug.RandomGrayscale(p=run.config.gray_scale),
        aug.RandomResizedCrop(
        size=(224, 224),       
        scale=(0.8, 1),
        ratio=(1, 1)),
        aug.ColorJitter(
        brightness=run.config.cj_bright,
        contrast=run.config.cj_con,
        saturation=run.config.cj_sat,
        hue=run.config.cj_hue),
        aug.RandomGaussianBlur(kernel_size=run.config.gb_kernel, sigma=(run.config.gb_sigma_min, run.config.gb_sigma_max))
        )


            
        best_test_accuracy = 0.0

        best_model = model.state_dict()

        epochs_since_improvement = 0

        best_epoch = 0

        freeze_count = 1
        fire_count = 0
        best_unfreeze = 0
        best_train_accuracy = 0.0

        epoch = 0

        pbar = tqdm.tqdm(total=num_epochs, desc="TRAINING THE MODEL")

        continue_train = True

        while continue_train:


            epochs_since_improvement += 1


            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device,augmentations=augmentations)
            test_loss, test_accuracy = test(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            run.log({"train_loss": train_loss, "train_ accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "epoch": epoch,})

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_train_accuracy = train_accuracy
                best_model = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
                best_unfreeze = fire_count
                best_epoch = epoch + 1 
                

            if epochs_since_improvement >= 75:
                if config["unfreeze"] == "None":
                    print(f"Early stopping at epoch {best_epoch}.")
                    continue_train = False
                    
                elif config["unfreeze"] == "Last":
                    model.load_state_dict(best_model)
                    print("Unfreezing last layer")
                    if fire_count < 1:
                        last_fire = model.backbone.features[-1]
                        
                        for param in last_fire.parameters():
                            param.requires_grad = True
                        fire_count += 1
                        epochs_since_improvement = 0

                    else:
                        print(f"Early stopping at epoch {best_epoch}.")
                        continue_train = False
                    
                elif config["unfreeze"] == "All":
                    
                    if fire_count < 8:
                        model.load_state_dict(best_model)
                        freeze_count = unfreeze_layers(model, freeze_count)
                        fire_count += 1
                        epochs_since_improvement = 0
                    else:
                        print(f"Early stopping at epoch {best_epoch}.")
                        continue_train = False
                        
                else: 
                    if freeze_count <= config["unfreeze"] :
                        model.load_state_dict(best_model)
                        freeze_count = unfreeze_layers(model, freeze_count)
                        fire_count += 1
                        epochs_since_improvement = 0
                    else:
                        print(f"Early stopping at epoch {best_epoch}.")
                        continue_train = False
                

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            epoch += 1
            pbar.update(1)

            if epoch == num_epochs:
                continue_train = False 
        run_id = str(time.time())
        os.makedirs(f"saved_models/OG",exist_ok=True)
        torch.save(best_model, f"saved_models/OG/{run_id}.pth")
        wandb.log({"best_test_accuracy": best_test_accuracy, "best_train_accuracy": best_train_accuracy, "best_epoch": best_epoch, "best_unfreeze": best_unfreeze, "run_id": run_id})
        
        with open(f"saved_models/OG/run_accuracies.txt", "a") as f:
            f.write(f"{run_id} : {best_test_accuracy:.4f} : {best_train_accuracy:.4f} : {best_epoch} : {best_unfreeze}\n")


        run.log({"accuracy": best_test_accuracy})

if __name__ == '__main__':
        
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "hor_flip": {"max": 0.5, "min": 0.0},
            "ver_flip": {"max": 0.5, "min": 0.0},
            "gray_scale": {"max": 0.5, "min": 0.0},
            "ran_rot": {"max": 45, "min": 0},
            "cj_bright": {"max": 0.5, "min": 0.0},
            "cj_con": {"max": 0.5, "min": 0.0},
            "cj_sat": {"max": 0.5, "min": 0.0},
            "cj_hue": {"max": 0.5, "min": 0.0},
            "gb_kernel": {"values":[1, 3,5, 7,9]},
            "gb_sigma_min": {"max": 1.0, "min": 0.0},
            "gb_sigma_max": {"max": 3.0, "min": 1.01},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="C3-Week3")

    wandb.agent(sweep_id, function=main, count=300)