from email.policy import strict
from pathlib import Path
import torch
import copy
from main import test

from enum import auto
import time
from typing import *
from networkx import freeze
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

from torchvision.models.squeezenet import Fire


import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

import wandb

import os

from main import load_data_on_gpu

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def average_state_dicts(state_dicts):
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        for sd in state_dicts[1:]:
            avg[k] += sd[k]
        avg[k] /= len(state_dicts)
    return avg

def evaluate_state_dict(state_dict, model, test_loader, criterion, device):
    model.load_state_dict(state_dict,strict=False)
    model.to(device)
    _, acc = test(model, test_loader, criterion, device)
    return acc


def load_runs(summary_file):
    runs = []
    with open(summary_file, "r") as f:
        for line in f:
            run_id, test_acc, train_acc, epoch, unfreeze = line.strip().split(" : ")
            runs.append({
                "run_id": run_id,
                "test_acc": float(test_acc),
                "train_acc": float(train_acc),
                "epoch": int(epoch),
                "unfreeze": int(unfreeze),
            })
    return runs

def greedy_soup(
    runs,
    model,
    test_loader,
    criterion,
    device,
    weights_dir,
):
    # Find the best model by actually evaluating all of them
    print("Finding best starting model...")
    best_acc = -1
    best_idx = 0
    best_sd = None
    
    for idx, run in enumerate(runs):
        path = Path(weights_dir) / f"{run['run_id']}.pth"
        sd = torch.load(path, map_location="cpu",)
        acc = evaluate_state_dict(sd, model, test_loader, criterion, device)
        
        print(f"Run {run['run_id']}: reported={run['test_acc']:.4f}, actual={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_idx = idx
            best_sd = sd
    
    print(f"\nBest model: {runs[best_idx]['run_id']} with accuracy {best_acc:.4f}")
    
    # Start soup with best model
    soup_state_dicts = [best_sd]
    soup_acc = best_acc
    
    # Try adding other models
    for idx, run in enumerate(runs):
        if idx == best_idx:
            continue  # Skip the model we already added
        
        path = Path(weights_dir) / f"{run['run_id']}.pth"
        candidate_sd = torch.load(path, map_location="cpu")

        candidate_avg = average_state_dicts(
            soup_state_dicts + [candidate_sd]
        )

        candidate_acc = evaluate_state_dict(
            candidate_avg, model, test_loader, criterion, device
        )

        print(
            f"Trying {run['run_id']} | "
            f"single={run['test_acc']:.4f} | "
            f"soup={candidate_acc:.4f}"
        )

        if candidate_acc > soup_acc:
            soup_state_dicts.append(candidate_sd)
            soup_acc = candidate_acc
            print("  ✔ Accepted into soup")
        else:
            print("  ✘ Rejected")

    final_soup = average_state_dicts(soup_state_dicts)
    return final_soup, soup_acc

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

train_loader = load_data_on_gpu(data_train,device=device,batch_size=256)
test_loader = load_data_on_gpu(data_test,device=device,batch_size=128)

C, H, W = np.array(data_train[0][0]).shape




model = WraperModel(num_classes=8, feature_extraction=True,batch_norm=False,)#SimpleModel(input_d=C*H*W, hidden_d=300, output_d=8)



summary_file = "saved_models/OG/run_accuracies.txt"
weights_dir = "saved_models/OG"

runs = load_runs(summary_file)

criterion = nn.CrossEntropyLoss()

final_soup_sd, final_acc = greedy_soup(
    runs=runs,
    model=model,                     
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    weights_dir=weights_dir,
)

torch.save(final_soup_sd, "saved_models/OG/greedy_soup.pth")
print(f"Final Greedy Soup accuracy: {final_acc:.4f}")