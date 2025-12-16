from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel
import torchvision.transforms.v2 as F
from kornia import augmentation as aug
from torchviz import make_dot
import tqdm
import wandb
import json

from sklearn.svm import LinearSVC
from main import test,extract_features,train_with_patches,test_patches,extract_patch_features
from fisher_vector import neural_based_fisher
from IPython.display import clear_output
from torch.utils.data import TensorDataset, DataLoader

from kornia import augmentation as aug


def train(model, train_loader, criterion, optimizer, device,augmentation = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    

    for inputs, labels in train_loader:
        # Data is already on device, so augmentations will be on GPU
        inputs = augmentation(inputs)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['JOBLIB_TEMP_FOLDER'] = '/home/msiau/workspace/jventosa/PostTFG/Master/project-1/Week2/joblib'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Initialize WandB
wandb.init(
    project="image-classification-hyperparameter-search",
    config={
        "architecture": "SimpleModel",
        "dataset": "places_reduced",
        "epochs": 100,
        "optimizer": "Adam",
        "learning_rate": 0.001,
    }
)

def train_simple_model(model, model_name, train_loader, test_loader, search_type=None, param_value=None, augmentation= None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 150

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_loss = 10_000_000
    train_loss_at_best = 10_000_000
    best_test_accuracy = 0
    best_epoch = 0
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device,augmentation=augmentation)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Log metrics to WandB
        log_dict = {
            f"{search_type}/train_loss" if search_type else "train_loss": train_loss,
            f"{search_type}/train_accuracy" if search_type else "train_accuracy": train_accuracy,
            f"{search_type}/test_loss" if search_type else "test_loss": test_loss,
            f"{search_type}/test_accuracy" if search_type else "test_accuracy": test_accuracy,
            f"{search_type}/epoch" if search_type else "epoch": epoch + 1,
        }
        
        if param_value is not None:
            log_dict[f"{search_type}/param_value"] = param_value
            
        wandb.log(log_dict)

        if  best_test_accuracy <  test_accuracy:
            best_test_loss = test_loss
            train_loss_at_best = train_accuracy
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            model_dict = model.state_dict()
            
            # Log best model to WandB
            if search_type:
                wandb.log({
                    f"{search_type}/best_test_loss": best_test_loss,
                    f"{search_type}/best_test_accuracy": best_test_accuracy,
                    f"{search_type}/best_epoch": best_epoch,
                })
        if epoch % 10 == 0:
            torch.save(model_dict, f"SimpleModel/{model_name}.pth")
            
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    torch.save(model_dict, f"SimpleModel/{model_name}.pth")
    return best_test_accuracy, train_loss_at_best, best_epoch


# # ==================== IMAGE SIZE SEARCH ====================
# wandb.log({"search_phase": "image_size_search"})
# image_search_results = {}
# best_accuracy = 100000

# for image_size in [256, 128, 64, 32, 16, 8,]:
#     print(f"\n{'='*50}")
#     print(f"Testing image size: {image_size}x{image_size}")
#     print(f"{'='*50}\n")
    
#     transformation = F.Compose([
#         F.ToImage(),
#         F.ToDtype(torch.float32, scale=True),
#         F.Resize(size=(image_size, image_size)),
#     ])


#     augmentations = aug.AugmentationSequential(
#         aug.RandomHorizontalFlip(p=0.5),
#         aug.RandomRotation(9),
#         aug.RandomVerticalFlip(p=0.05),
#         aug.RandomGrayscale(p=0.1),
#         aug.RandomResizedCrop(
#         size=(image_size, image_size),       
#         scale=(0.8, 1),
#         ratio=(1, 1)),
#         aug.ColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.05),
#         aug.RandomGaussianBlur(kernel_size=5, sigma=(0.1, 0.6))
        
#     )

#     data_train = ImageFolder("../places_reduced/train", transform=transformation)
#     data_test = ImageFolder("../places_reduced/val", transform=transformation)

#     train_images = []
#     train_labels = []
#     for img, label in data_train:
#         train_images.append(img)
#         train_labels.append(label)

#     train_images = torch.stack(train_images).to(device=device)
#     train_labels = torch.tensor(train_labels, device=device)

#     print("Loading test data to VRAM...")
#     test_images = []
#     test_labels = []
#     for img, label in data_test:
#         test_images.append(img)
#         test_labels.append(label)

#     test_images = torch.stack(test_images).to(device=device)
#     test_labels = torch.tensor(test_labels, device=device)

#     train_dataset_gpu = TensorDataset(train_images, train_labels)
#     test_dataset_gpu = TensorDataset(test_images, test_labels)

#     train_loader = DataLoader(train_dataset_gpu, batch_size=256, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset_gpu, batch_size=128, shuffle=False, num_workers=0)
    
#     C, H, W = np.array(data_train[0][0]).shape
#     input_size = C * H * W
#     hidden_layers_n = 2
#     hidden_dim = 300

#     model = SimpleModel(input_d=C*H*W, hidden_layers_n=hidden_layers_n, hidden_d=hidden_dim, output_d=11)
#     model_name = f"{input_size}_input_{hidden_layers_n}_layers_{hidden_dim}_dimension"
    
#     image_search_results[input_size] = train_simple_model(
#         model, model_name, train_loader, test_loader, 
#         search_type=f"image_size_search_{image_size}", param_value=image_size,augmentation=augmentations
#     )
    
#     if image_search_results[input_size][0] < best_accuracy:
#         best_accuracy = image_search_results[input_size][0]
#         best_input_size = input_size
#         wandb.log({
#             "image_size_search/best_image_size": image_size,
#             "image_size_search/best_input_size": best_input_size,
            
#             "image_size_search/best_overall_accuracy": best_accuracy,
#         })

# with open('image_search_results.json', 'w') as f:
#     json.dump(image_search_results, f, indent=4)

# wandb.log({"image_size_search/results": wandb.Table(
#     columns=["input_size", "best_accuracy", "train_accuracy", "best_epoch"],
#     data=[[k, v[0], v[1], v[2]] for k, v in image_search_results.items()]
# )})

# # ==================== LAYER SEARCH ====================
# wandb.log({"search_phase": "layer_search"})
optimal_image_size = 16

augmentations = aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=0.5),
        aug.RandomRotation(9),
        aug.RandomVerticalFlip(p=0.05),
        aug.RandomGrayscale(p=0.1),
        aug.RandomResizedCrop(
        size=(optimal_image_size, optimal_image_size),       
        scale=(0.8, 1),
        ratio=(1, 1)),
        aug.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05),
        aug.RandomGaussianBlur(kernel_size=5, sigma=(0.1, 0.6))
        
    )

transformation = F.Compose([
    F.ToImage(),
    F.ToDtype(torch.float32, scale=True),
    F.Resize(size=(optimal_image_size, optimal_image_size)),
])

data_train = ImageFolder("../places_reduced/train", transform=transformation)
data_test = ImageFolder("../places_reduced/val", transform=transformation)

train_images = []
train_labels = []
for img, label in data_train:
    train_images.append(img)
    train_labels.append(label)

train_images = torch.stack(train_images).to(device=device)
train_labels = torch.tensor(train_labels, device=device)

print("Loading test data to VRAM...")
test_images = []
test_labels = []
for img, label in data_test:
    test_images.append(img)
    test_labels.append(label)

test_images = torch.stack(test_images).to(device=device)
test_labels = torch.tensor(test_labels, device=device)

train_dataset_gpu = TensorDataset(train_images, train_labels)
test_dataset_gpu = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset_gpu, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset_gpu, batch_size=128, shuffle=False, num_workers=0)

# layer_search_results = {}
# best_accuracy = 100000

# for hidden_layers_n in [1, 2, 3, 4, 5, 6, 7, 8]:
#     print(f"\n{'='*50}")
#     print(f"Testing hidden layers: {hidden_layers_n}")
#     print(f"{'='*50}\n")
    
#     C, H, W = np.array(data_train[0][0]).shape
#     input_size = C * H * W
#     hidden_dim = 300

#     model = SimpleModel(input_d=C*H*W, hidden_layers_n=hidden_layers_n, hidden_d=hidden_dim, output_d=11)
#     model_name = f"{input_size}_input_{hidden_layers_n}_layers_{hidden_dim}_dimension"
    
#     layer_search_results[hidden_layers_n] = train_simple_model(
#         model, model_name, train_loader, test_loader,
#         search_type=f"layer_search_{hidden_layers_n}", param_value=hidden_layers_n,augmentation=augmentations
#     )
    
#     if layer_search_results[hidden_layers_n][0] < best_accuracy:
#         best_accuracy = layer_search_results[hidden_layers_n][0]
#         best_hidden_layers_n = hidden_layers_n
#         wandb.log({
#             "layer_search/best_layer_count": best_hidden_layers_n,
#             "layer_search/best_overall_accuracy": best_accuracy,
#         })

# with open('layer_search_results.json', 'w') as f:
#     json.dump(layer_search_results, f, indent=4)

# wandb.log({"layer_search/results": wandb.Table(
#     columns=["hidden_layers", "best_accuracy", "train_accuracy", "best_epoch"],
#     data=[[k, v[0], v[1], v[2]] for k, v in layer_search_results.items()]
# )})

# ==================== DIMENSION SEARCH ====================
wandb.log({"search_phase": "dimension_search"})
best_accuracy = 100000
dimension_search_results = {}
optimal_layer_n = 1
best_hidden_layers_n = 1
hidden_layers_n = 1

# for hidden_dim in [32, 64, 128, 256, 512, 300]:
#     print(f"\n{'='*50}")
#     print(f"Testing hidden dimension: {hidden_dim}")
#     print(f"{'='*50}\n")
    
#     C, H, W = np.array(data_train[0][0]).shape
#     input_size = C * H * W

#     model = SimpleModel(input_d=C*H*W, hidden_layers_n=best_hidden_layers_n, hidden_d=hidden_dim, output_d=11)
#     model_name = f"{input_size}_input_{hidden_layers_n}_layers_{hidden_dim}_dimension"
    
#     dimension_search_results[hidden_dim] = train_simple_model(
#         model, model_name, train_loader, test_loader,
#         search_type=f"dimension_search_{hidden_dim}", param_value=hidden_dim,augmentation=augmentations
#     )
    
#     if dimension_search_results[hidden_dim][0] < best_accuracy:
#         best_accuracy = dimension_search_results[hidden_dim][0]
#         best_hidden_dim = hidden_dim
#         wandb.log({
#             "dimension_search/best_hidden_dim": best_hidden_dim,
#             "dimension_search/best_overall_accuracy": best_accuracy,
#         })

# with open('dimension_search_results.json', 'w') as f:
#     json.dump(dimension_search_results, f, indent=4)

# wandb.log({"dimension_search/results": wandb.Table(
#     columns=["hidden_dim", "best_accuracy", "train_accuracy", "best_epoch"],
#     data=[[k, v[0], v[1], v[2]] for k, v in dimension_search_results.items()]
# )})

# ==================== PATCH SEARCH ====================
wandb.log({"search_phase": "patch_search"})
patches_search_results = {}


augmentations = aug.AugmentationSequential(
        aug.RandomHorizontalFlip(p=0.5),
        aug.RandomRotation(9),
        aug.RandomVerticalFlip(p=0.05),
        aug.RandomGrayscale(p=0.1),
        aug.RandomResizedCrop(
        size=(256, 256),       
        scale=(0.8, 1),
        ratio=(1, 1)),
        aug.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05),
        aug.RandomGaussianBlur(kernel_size=5, sigma=(0.1, 0.6))
        
    )

transformation = F.Compose([
    F.ToImage(),
    F.ToDtype(torch.float32, scale=True),
    F.Resize(size=(256, 256)),
])

data_train = ImageFolder("../places_reduced/train", transform=transformation)
data_test = ImageFolder("../places_reduced/val", transform=transformation)

train_images = []
train_labels = []
for img, label in data_train:
    train_images.append(img)
    train_labels.append(label)

train_images = torch.stack(train_images).to(device=device)
train_labels = torch.tensor(train_labels, device=device)

print("Loading test data to VRAM...")
test_images = []
test_labels = []
for img, label in data_test:
    test_images.append(img)
    test_labels.append(label)

test_images = torch.stack(test_images).to(device=device)
test_labels = torch.tensor(test_labels, device=device)

train_dataset_gpu = TensorDataset(train_images, train_labels)
test_dataset_gpu = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset_gpu, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset_gpu, batch_size=128, shuffle=False, num_workers=0)

best_hidden_dim = 256

hidden_dim = 256


for patch_size in [4,8,16, 32, 64,]:
    print(f"\n{'='*50}")
    print(f"Testing patch size: {patch_size}x{patch_size}")
    print(f"{'='*50}\n")
    
    C, H, W = np.array(data_train[0][0]).shape
    optimal_layer_n = 1

    model = SimpleModel(input_d=C*patch_size*patch_size, hidden_layers_n=optimal_layer_n, 
                       hidden_d=best_hidden_dim, output_d=11)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 150

    model_name = f"{patch_size}_patchsize_{hidden_layers_n}_layers_{hidden_dim}_dimension"
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_train = 0
    best_epoch = 0
    best_test_loss = 10_000_000
    best_accuracy = 0
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE PATCH BASED MODEL"):
        train_loss, train_accuracy = train_with_patches(model, train_loader, criterion, optimizer, device, patch_size)
        test_loss, test_accuracy = test_patches(model, test_loader, criterion, device, patch_size)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Log patch training metrics
        wandb.log({
            f"patch_search_{patch_size}/train_loss": train_loss,
            f"patch_search_{patch_size}/train_accuracy": train_accuracy,
            f"patch_search_{patch_size}/test_loss": test_loss,
            f"patch_search_{patch_size}/test_accuracy": test_accuracy,
            f"patch_search_{patch_size}/epoch": epoch + 1,
        })
        
        if best_accuracy < test_accuracy:
            best_test_loss = test_loss
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            best_train = train_accuracy
            os.makedirs(f"PatchModel", exist_ok=True)
            model_dict = model.state_dict()
            
            wandb.log({
                f"patch_search/best_test_loss": best_test_loss,
                f"patch_search/best_accuracy": best_accuracy,
                f"patch_search/best_epoch": best_epoch,
            })
        if epoch % 10 == 0:
            torch.save(model_dict, f"PatchModel/{model_name}.pth")
    
    torch.save(model_dict, f"SimpleModel/{model_name}.pth")
    patches_search_results[patch_size] = [best_accuracy, train_accuracy, best_epoch]

with open('patches_search_results.json', 'w') as f:
    json.dump(patches_search_results, f, indent=4)

wandb.log({"patch_search/results": wandb.Table(
    columns=["patch_size", "best_accuracy", "train_accuracy", "best_epoch"],
    data=[[k, v[0], v[1], v[2]] for k, v in patches_search_results.items()]
)})

# Log final summary
wandb.log({
    "final/best_image_size": best_input_size,
    "final/best_hidden_layers": best_hidden_layers_n,
    "final/best_hidden_dim": best_hidden_dim,
})

print("\n" + "="*50)
print("HYPERPARAMETER SEARCH COMPLETE")
print("="*50)
print(f"Best image size: {best_input_size}")
print(f"Best hidden layers: {best_hidden_layers_n}")
print(f"Best hidden dimension: {best_hidden_dim}")
print(f"View results at: {wandb.run.get_url()}")

wandb.finish()