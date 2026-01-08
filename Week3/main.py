from enum import auto
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Train function
def train(model, dataloader, criterion, optimizer, device,augmentations=None):
    model.train()
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


def load_data_on_gpu(data,device,batch_size=256):

    
    data_images = []
    data_labels = []
    for img, label in data:
        data_images.append(img)
        data_labels.append(label)

    data_images = torch.stack(data_images).to(device=device)
    data_labels = torch.tensor(data_labels, device=device)

    dataset_gpu = TensorDataset(data_images, data_labels)

    return DataLoader(dataset_gpu, batch_size=256, shuffle=True, num_workers=0)


def unfreeze_layers(model, freeze_count):
    freezed = True
    while freezed:
        last_fire = model.backbone.features[-freeze_count]
        if isinstance(last_fire, Fire):
            for param in last_fire.parameters():
                param.requires_grad = True
            freezed = False
            freeze_count += 1
        else:
            freeze_count += 1
    return freeze_count


def train_run(train_loader,test_loader,model, criterion, optimizer, device,num_epochs,config,run_name,augmentations):
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_accuracy = 0.0

    best_model = model.state_dict()

    epochs_since_improvement = 0

    best_epoch = 0

    freeze_count = 1
    fire_count = 0

    with wandb.init(project=project, config=config,name=run_name) as run:
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
                best_model = model.state_dict()
                epochs_since_improvement = 0
                best_epoch = epoch + 1 
                

            if epochs_since_improvement >= 50:
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
        
    torch.save(best_model, "./saved_model.pt")

    run.log({"BestEpoch": best_epoch, "BestTestAccuracy": best_test_accuracy})

def unfreeze_arg(string):
    if string not in ['None', 'Last', 'All']:
        return int(string)
    else:
        return string



if __name__ == "__main__":

    


    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", required=False, type=int,default=1000)
    parser.add_argument("--lr", required=False, type=int,default=0.001)
    parser.add_argument("--batch", required=False, type=int,default=256)
    parser.add_argument("--run_name", required=False, type=str,default="")
    parser.add_argument("--unfreeze", required=False, type=unfreeze_arg,default="None")
    parser.add_argument("--weight_decay", required=False, type=float,default=0.0001)
    parser.add_argument("--optimizer", required=False, type=str,choices=['adam', 'sgd'],default="adam") 
    parser.add_argument("--learning_rate", required=False, type=float,default=0.001)
    parser.add_argument("--batch_normalization", required=False, type=bool,default=False)
    parser.add_argument("--dropout", required=False, type=bool,default=True)
    parser.add_argument("--dropout_prob", required=False, type=float,default=0.5)
    parser.add_argument("--squeeze_excite", required=False, type=bool,default=False)
    parser.add_argument("--reduction", required=False, type=int,default=16)
    parser.add_argument("--classifier_type", required=False, type=str,choices=['FCN', 'MLP', 'Attention'],default="FCN")




    args = parser.parse_args()

    base_name = args.run_name


    wandb.login()

    project = "C3-Week3"


    config = {
        'epochs' : args.epochs,
        'lr' : args.lr,
        'batch_size' : args.batch,
        'unfreeze' : args.unfreeze,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'batch_normalization': args.batch_normalization,
        'dropout': args.dropout,
        'dropout_prob': args.dropout_prob,
        'add_squeeze_excite': args.squeeze_excite,
        'reduction': args.reduction,
        'classifier_type': args.classifier_type,
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

    


    model = WraperModel(num_classes=8, feature_extraction=True,batch_norm=config["batch_normalization"],dropout=config["dropout"],dropout_prob=config["dropout_prob"])#SimpleModel(input_d=C*H*W, hidden_d=300, output_d=8)

    if config["add_squeeze_excite"]:
        model.add_squeeze_and_excite(reduction=config["reduction"])

    model.delete_last_n_modules(n=3)
    print(model.backbone.features)
    # model.add_fire_modules(n=2, sq_channels=64, exp_channels=256)
    
    model = model.to(device)

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
    
    run_name = f"{base_name}_{config['epochs']}_{config['lr']}_{config['batch_size']}_{config['unfreeze']}"
    
    augmentations = aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=0.5),
        aug.RandomRotation(9),
        aug.RandomVerticalFlip(p=0.05),
        aug.RandomGrayscale(p=0.1),
        aug.RandomResizedCrop(
        size=(224, 224),       
        scale=(0.8, 1),
        ratio=(1, 1)),
        aug.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05),
        aug.RandomGaussianBlur(kernel_size=5, sigma=(0.1, 0.6))
        
    )

    """ augmentations = aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=0),
    ) """

    
    train_run(train_loader,test_loader,model, criterion, optimizer, device,num_epochs,config,run_name,augmentations=augmentations)

    