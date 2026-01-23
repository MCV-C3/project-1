from enum import auto
import json
import time
from typing import *
from networkx import freeze
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
from model import ModularCNN
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
from kornia import augmentation as aug
import copy
import os



import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

import wandb


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_rank():
    """Get the rank of the current process"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get the total number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0



# Train function
def train(model, dataloader, criterion, optimizer, device, augmentations=None, sampler=None, epoch=None):
    model.train()
    
    # Set epoch for DistributedSampler to ensure different shuffle each epoch
    if sampler is not None and epoch is not None:
        sampler.set_epoch(epoch)
    
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        if augmentations is not None:
            inputs = augmentations(inputs)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Aggregate metrics across all GPUs
    if dist.is_initialized():
        metrics = torch.tensor([train_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        train_loss, correct, total = metrics.tolist()
    
    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Aggregate metrics across all GPUs
    if dist.is_initialized():
        metrics = torch.tensor([test_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        test_loss, correct, total = metrics.tolist()
    
    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = "loss.png" if metric_name.lower() == "loss" else "metrics.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory

# Data augmentation example
def get_data_transforms():
    """
    Returns a Compose object with data augmentation transformations.
    """
    return Compose([
        RandomResizedCrop(size=224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")


def create_dataloader(dataset, batch_size, is_train=True, num_workers=4):
    """Create DataLoader with optional DistributedSampler for DDP"""
    sampler = None
    shuffle = is_train
    
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=is_train)
        shuffle = False  # Sampler handles shuffling
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, sampler




def train_run(train_loader, test_loader, model, criterion, optimizer, device, num_epochs, config, run_name, augmentations, train_sampler=None):
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_accuracy = 0.0
    best_model = None
    epochs_since_improvement = 0
    best_epoch = 0
    best_train_accuracy = 0.0

    # Only initialize wandb on main process
    if is_main_process():
        wandb.init(project=project, config=config, name=run_name)
        pbar = tqdm.tqdm(total=num_epochs, desc="TRAINING THE MODEL")
    
    epoch = 0
    continue_train = True

    while continue_train:
        epochs_since_improvement += 1

        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, augmentations=augmentations, sampler=train_sampler, epoch=epoch)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Only log and save on main process
        if is_main_process():
            wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "epoch": epoch})

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_train_accuracy = train_accuracy
                # Get state dict from DDP model
                if isinstance(model, DDP):
                    best_model = copy.deepcopy(model.module.state_dict())
                else:
                    best_model = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
                best_epoch = epoch + 1

            if epochs_since_improvement >= 75:
                print(f"Early stopping at epoch {best_epoch}.")
                continue_train = False

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            pbar.update(1)

        epoch += 1
        if epoch == num_epochs:
            continue_train = False
        
        # Synchronize early stopping across all processes
        if dist.is_initialized():
            continue_tensor = torch.tensor([1 if continue_train else 0], device=device)
            dist.broadcast(continue_tensor, src=0)
            continue_train = bool(continue_tensor.item())

    # Only save and log final results on main process
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        run_id = str(time.time())
        os.makedirs(f"saved_models/{config['model_name']}", exist_ok=True)
        torch.save(best_model, f"saved_models/{config['model_name']}/{run_id}.pth")
        wandb.log({"best_test_accuracy": best_test_accuracy, "best_train_accuracy": best_train_accuracy, 
                   "total_parameters": total_params, "total_learnable_parameters": total_learnable_params,
                   "best_epoch": best_epoch, "run_id": run_id})
        
        with open(f"saved_models/{config['model_name']}/run_accuracies.txt", "a") as f:
            f.write(f"{run_id} : {best_test_accuracy:.4f} : {best_train_accuracy:.4f} : {best_epoch}\n")
        
        wandb.finish()
        pbar.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", required=False, type=int, default=2000)
    parser.add_argument("--lr", required=False, type=int, default=0.0001)
    parser.add_argument("--batch", required=False, type=int, default=128)
    parser.add_argument("--run_name", required=False, type=str, default="")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.0001)
    parser.add_argument("--optimizer", required=False, type=str, choices=['adam', 'sgd'], default="adam") 
    parser.add_argument("--learning_rate", required=False, type=float, default=0.0001)
    parser.add_argument("--model_name", required=False, type=str, default="OG")
    parser.add_argument("--model_config", required=False, type=json.loads, default="")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Initialize distributed training if enabled
    if args.distributed:
        setup_distributed()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    you_are_jordi = True

    if you_are_jordi:
        base_path = "/home/msiau/data/tmp/jventosa/2425"
    else:
        base_path = "valentin_data"

    # Only login to wandb on main process
    if is_main_process():
        wandb.login()

    project = "C3-Week4"


    config = {
        'model_name': args.model_name,
        'epochs' : args.epochs,
        'lr' : args.lr,
        'batch_size' : args.batch,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'model_config': args.model_config
    }


    if args.run_name == "":
        run_name = str(time.time())
    else:
        run_name = args.run_name
    
    torch.manual_seed(42)
    if dist.is_initialized():
        # Set different seed per GPU for better randomness
        torch.manual_seed(42 + get_rank())

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])

    choosen_split = 1

    data_train = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/train", transform=transformation)
    data_test = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/test", transform=transformation)

    # Create dataloaders with distributed samplers
    train_loader, train_sampler = create_dataloader(data_train, batch_size=config["batch_size"], is_train=True, num_workers=4)
    test_loader, _ = create_dataloader(data_test, batch_size=128, is_train=False, num_workers=4)

    C, H, W = np.array(data_train[0][0]).shape

    model = ModularCNN(num_classes=8, input_channels=3, config=config["model_config"])
    model = model.to(device)

    # Wrap model with DDP if distributed training is enabled
    if args.distributed:
        model = DDP(model, device_ids=[device.index])

    criterion = nn.CrossEntropyLoss()
    
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9
        )
    
    num_epochs = config["epochs"]

    best_augmentations = {
        "cj_bright": 0.495837262449209,
        "cj_con": 0.21104613670053696,
        "cj_hue": 0.017212853392425453,
        "cj_sat": 0.2391866946177022,
        "gb_kernel": 9,
        "gb_sigma_max": 2.741465766783409,
        "gb_sigma_min": 0.029591630066822416,
        "gray_scale": 0.018206674887420504,
        "hor_flip": 0.25070926882889294,
        "ran_rot": 10,
        "ver_flip": 0.02239975089933588,
    }

    augmentations = aug.AugmentationSequential(
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
        aug.RandomGaussianBlur(kernel_size=best_augmentations["gb_kernel"], 
                               sigma=(best_augmentations["gb_sigma_min"], best_augmentations["gb_sigma_max"]))
    )

    try:
        train_run(train_loader, test_loader, model, criterion, optimizer, device, 
                  num_epochs, config, run_name, augmentations=augmentations, train_sampler=train_sampler)
    finally:
        # Clean up distributed training
        if args.distributed:
            cleanup_distributed()

    