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

    with wandb.init(project=project) as run:

        config = {
            'epochs' : 2000,
            'lr' : run.config.learn_rate,
            'batch_size' : run.config.batch_size,
            'unfreeze' : "All",
            'weight_decay': 0.0001,
            'optimizer': run.config.optimizer,
            'momentum': run.config.momentum,
            'batch_norm': run.config.batch_norm,
            'dropout': run.config.dropout,
            'dropout_prob': run.config.dropout_prob if run.config.dropout else 0.0,
        }

        best_augmentations = {"cj_bright":
     0.495837262449209,
"cj_con":
     0.21104613670053696,
"cj_hue":
     0.017212853392425453,
"cj_sat":
     0.2391866946177022,
"gb_kernel":
     9,
"gb_sigma_max":
     2.741465766783409,
"gb_sigma_min":
     0.029591630066822416,
"gray_scale":
     0.018206674887420504,
"hor_flip":
     0.25070926882889294,
"ran_rot":
     10,
"ver_flip":
     0.02239975089933588,
                    }
            

        # torch.manual_seed(42)

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

        

        model = WraperModel(
            num_classes=8, 
            feature_extraction=True,
            batch_norm=config['batch_norm'],
            dropout=config['dropout'],
            dropout_prob=config['dropout_prob'],
            classifier_type="FCN"
        )
        

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Select optimizer based on config
        if config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'])
        elif config['optimizer'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=config["lr"], momentum=config['momentum'])
        elif config['optimizer'] == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=config["lr"])
        elif config['optimizer'] == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])
        elif config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        elif config['optimizer'] == 'Adamax':
            optimizer = optim.Adamax(model.parameters(), lr=config["lr"])
        elif config['optimizer'] == 'Nadam':
            optimizer = optim.NAdam(model.parameters(), lr=config["lr"])
        
        num_epochs = config["epochs"]


        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        best_test_accuracy = 0.0

        epochs_since_improvement = 0

        best_epoch = 0
        
        run_name = f"Hyperparameter_Sweep"

        aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=best_augmentations["hor_flip"]),
        aug.RandomRotation(best_augmentations["ran_rot"]),
        aug.RandomVerticalFlip(p=best_augmentations["ver_flip"]),
        aug.RandomGrayscale(p=best_augmentations["gray_scale"]),
        aug.RandomResizedCrop(
            size=(224, 224),       
            scale=(0.8, 1),
            ratio=(1, 1)),
        aug.ColorJitter(
            brightness=best_augmentations["cj_bright"],
            contrast=best_augmentations["cj_con"],
            saturation=best_augmentations["cj_sat"],
            hue=best_augmentations["cj_hue"]),
        aug.RandomGaussianBlur(kernel_size=best_augmentations["gb_kernel"], sigma=(best_augmentations["gb_sigma_min"], best_augmentations["gb_sigma_max"]))
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

            run.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "epoch": epoch,})

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
            "batch_size": {"values": [16, 32, 64, 128, 256, 512]},
            "optimizer": {"values": ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']},
            "learn_rate": {"values": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]},
            "momentum": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]},
            
            # Batch normalization and dropout
            "batch_norm": {"values": [True, False]},
            "dropout": {"values": [True, False]},
            "dropout_prob": {"values": [0.2, 0.3, 0.4, 0.5, 0.6]},
            
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="C3-Week3")

    wandb.agent(sweep_id, function=main, count=300)