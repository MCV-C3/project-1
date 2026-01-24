import code
from enum import auto
import json
import time
from typing import *
from cv2 import threshold
from networkx import freeze
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import ModularCNN
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
from kornia import augmentation as aug
import copy



import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

import wandb



# Train function
def train(model, dataloader, criterion, optimizer, device,augmentations=None,masks = None):
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
        if masks != None:
            for name, param in model.named_parameters():
                if name in masks:
                    param.grad *= masks[name]


        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def quantized_train(model, dataloader, criterion, optimizer,cb_optimizer, device,augmentations=None,codebooks = None, indices_map= None,masks = None):
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
        cb_optimizer.zero_grad()
        
        update_codebooks(model, codebooks, indices_map, masks)

        cb_optimizer.step()

        # Reconstruct weights from updated centroids
        reconstruct_weights(model, codebooks, indices_map, masks)


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

def prune_model(model, sensitivity):
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = torch.std(param.data) * sensitivity
            masks[name] = torch.abs(param.data) > threshold
            param.data *= masks[name]
    
    return masks

def count_pruned_from_masks(masks):
    pruned = 0

    for mask in masks.values():
        pruned += (~mask).sum().item()

    return pruned

@torch.no_grad()
def init_codebooks(model, masks, bits=2):
    """
    Initializes codebooks (centroids) and index maps for each weight tensor.
    """
    codebooks = {}
    indices_map = {}

    n_clusters = 2 ** bits

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        weight = param.data
        mask = masks[name]

        # Extract non-zero (unpruned) weights
        sparse_weights = weight[mask].view(-1, 1).cpu().numpy()

        if sparse_weights.shape[0] < n_clusters:
            continue  # too small to cluster safely

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(sparse_weights)

        # Store centroids as trainable torch parameters
        codebooks[name] = torch.nn.Parameter(
            torch.from_numpy(kmeans.cluster_centers_).float().to(weight.device)
        )

        # Assign each weight to nearest centroid
        full_indices = torch.full_like(weight, -1, dtype=torch.long)
        assigned = torch.from_numpy(
            kmeans.predict(weight[mask].view(-1, 1).cpu().numpy())
            ).long().to(weight.device)


        full_indices[mask] = assigned
        indices_map[name] = full_indices

        # Replace weights with centroid values
        param.data[mask] = codebooks[name][assigned].view(-1)

    return codebooks, indices_map

def update_codebooks(model, codebooks, indices_map, masks):
    """
    Aggregate weight gradients into centroid gradients.
    """
    for name, param in model.named_parameters():
        if name not in codebooks:
            continue

        grad = param.grad
        if grad is None:
            continue

        indices = indices_map[name]
        mask = masks[name]

        cb = codebooks[name]
        cb_grad = torch.zeros_like(cb)

        for k in range(cb.shape[0]):
            sel = (indices == k) & mask
            if sel.any():
                cb_grad[k] = grad[sel].mean()

        cb.grad = cb_grad

@torch.no_grad()
def reconstruct_weights(model, codebooks, indices_map, masks):
    for name, param in model.named_parameters():
        if name not in codebooks:
            continue

        indices = indices_map[name]
        mask = masks[name]

        param.data[mask] = codebooks[name][indices[mask]].view(-1)


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




def train_run(train_loader,test_loader,model, criterion, optimizer, device,num_epochs,config,run_name,augmentations):
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_accuracy = 0.0

    best_model = model.state_dict()

    epochs_since_improvement = 0

    best_epoch = 0

    best_train_accuracy = 0.0

    masks = None

    codebooks = None

    indices_map = None

    pruning_improved = False


    optimal_pruning_steps = 0
    cb_optimizer = None
    pruned_parameters = 0

    with wandb.init(project=project, config=config,name=run_name) as run:
        epoch = 0

        pbar = tqdm.tqdm(total=num_epochs, desc="TRAINING THE MODEL")

        continue_train = True

        while continue_train:


            epochs_since_improvement += 1


            if codebooks == None:
                train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device,augmentations=augmentations,masks=masks)
            else:
                train_loss, train_accuracy = quantized_train(model, train_loader, criterion, optimizer,cb_optimizer, device,augmentations=augmentations,masks=masks,codebooks=codebooks,indices_map=indices_map)
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
                best_epoch = epoch + 1 
                pruning_improved = True

                

            if epochs_since_improvement >= 75:
                if config["pruning"]:
                    if pruning_improved:
                        print("pruning...")
                        model.load_state_dict(best_model)
                        masks = prune_model(model, 0.8)
                        optimal_pruning_steps += 1
                        epochs_since_improvement = 0
                        pruning_improved = False
                    else:
                        if config["quantization"] and codebooks == None:
                            print("quantizing...")
                            codebooks, indices_map = init_codebooks(model, masks, bits=config["quantization_bits"])
                            cb_optimizer = torch.optim.Adam(codebooks.values(), lr=1e-3)
                            epochs_since_improvement = 0
                        else:
                            print(f"Early stopping at epoch {best_epoch}.")
                            continue_train = False
                else:
                    if config["quantization"] and codebooks == None:
                            print("quantizing...")
                            codebooks, indices_map = init_codebooks(model, masks, bits=config["quantization_bits"])
                            cb_optimizer = torch.optim.Adam(codebooks.values(), lr=1e-3)
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

        total_params = sum(p.numel() for p in model.parameters())
        
        total_learnable_params = sum(p.numel() for p in model.parameters())
        if masks != None:
            pruned_parameters = count_pruned_from_masks(masks)
        else:
            pruned_parameters = 0




        run_id = str(time.time())
        os.makedirs(f"saved_models/{config['model_name']}",exist_ok=True)
        torch.save(best_model, f"saved_models/{config['model_name']}/{run_id}.pth")
        wandb.log({"best_test_accuracy": best_test_accuracy, "best_train_accuracy": best_train_accuracy, "total_parameters": total_params,"total_learnable_parameters": total_learnable_params,"pruned_parameters": pruned_parameters,"pruning_steps":optimal_pruning_steps,"best_epoch": best_epoch,"run_id": run_id})
        
        with open(f"saved_models/{config['model_name']}/run_accuracies.txt", "a") as f:
            f.write(f"{run_id} : {best_test_accuracy:.4f} : {best_train_accuracy:.4f} : {best_epoch}\n")



if __name__ == "__main__":

    


    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", required=False, type=int,default=10_000)
    parser.add_argument("--lr", required=False, type=float,default=0.0001)
    parser.add_argument("--batch", required=False, type=int,default=128)
    parser.add_argument("--run_name", required=False, type=str,default="")
    parser.add_argument("--weight_decay", required=False, type=float,default=0.0001)
    parser.add_argument("--optimizer", required=False, type=str,choices=['adam', 'sgd'],default="adam") 
    parser.add_argument("--learning_rate", required=False, type=float,default=0.0001)
    parser.add_argument("--model_name", required=False, type=str,default="OG")
    parser.add_argument("--gpu_index", required=False, type=str,default="1")
    parser.add_argument("--model_config", required=False, type=json.loads,default="")
    parser.add_argument("--head_type", required=False, type=str,default="gap")
    parser.add_argument("--head_params", required=False, type=json.loads,default={})
    parser.add_argument("--pruning", required=False, type=bool,default=False)
    parser.add_argument("--quantization", required=False, type=bool,default=False)
    parser.add_argument("--quantization_bits", required=False, type=int,default=2)

    args = parser.parse_args()

    you_are_jordi = True

    if you_are_jordi:
        import os

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_index
        base_path = "/home/msiau/data/tmp/jventosa/2425"
    else:
        base_path = "valentin_data"

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
        'model_config': args.model_config,
        'head_type': args.head_type,
        'head_params': args.head_params,
        'pruning': args.pruning,
        'quantization': args.quantization,
        'quantization_bits': args.quantization_bits
    }


    if args.run_name == "":
        run_name = str(time.time())
    else:
        run_name = args.run_name
    
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    

    choosen_split = 1

    data_train = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/train", transform=transformation)
    data_test = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/test", transform=transformation) 

    train_loader = load_data_on_gpu(data_train,device=device,batch_size=config["batch_size"])
    test_loader = load_data_on_gpu(data_test,device=device,batch_size=128)

    C, H, W = np.array(data_train[0][0]).shape

    model = ModularCNN(num_classes=8,input_channels = 3,config = config["model_config"],head_type=config["head_type"],head_params=config["head_params"])
    

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
        aug.RandomGaussianBlur(kernel_size=best_augmentations["gb_kernel"], sigma=(best_augmentations["gb_sigma_min"], best_augmentations["gb_sigma_max"]))
        )
    


    train_run(train_loader,test_loader,model, criterion, optimizer, device,num_epochs,config,run_name,augmentations=augmentations)

    