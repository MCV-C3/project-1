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

            
            # Save immediately when we find a better model
            torch.save(model_dict, f"SimpleModel/{model_name}.pth")
            print(f"New best model saved! Epoch {best_epoch}, Accuracy: {best_test_accuracy:.4f}")
            
            # Log best model to WandB
            if search_type:
                wandb.log({
                    f"{search_type}/best_test_loss": best_test_loss,
                    f"{search_type}/best_test_accuracy": best_test_accuracy,
                    f"{search_type}/best_epoch": best_epoch,
                })

            
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return best_test_accuracy, train_loss_at_best, best_epoch


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


BEST_IMAGE_SIZE = 16
BEST_HIDDEN_LAYERS = 1
BEST_HIDDEN_DIM = 256
BEST_PATCH_SIZE = 8

optimizer_results = {}

optimizers = {
    "Adam": lambda p: torch.optim.Adam(p, lr=1e-3),
    "SGD": lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
    "AdamW": lambda p: torch.optim.AdamW(p, lr=1e-3)
}

for name, opt_fn in optimizers.items():
    print(f"Optimizer: {name}")

    model = SimpleModel(
        input_d=3*16*16,
        hidden_d=256,
        hidden_layers_n=1,
        output_d=11
    )
    model.to(device)

    optimizer = opt_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()




    optimizer_results[name], _,_ = train_simple_model(model, name, train_loader, test_loader, search_type="w", param_value=3, augmentation= augmentations)

plt.figure(figsize=(5,4))
plt.bar(optimizer_results.keys(), optimizer_results.values())
plt.ylabel("Accuracy")
plt.title("Optimizer (Best Config)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig('raitest.png', dpi=300, bbox_inches='tight')
plt.show()

# activation_results = {}

# activations = {
#     "ReLU": nn.ReLU(),
#     "LeakyReLU": nn.LeakyReLU(0.1),
#     "Tanh": nn.Tanh()
# }

# for name, act in activations.items():
#     print(f"Activation: {name}")

#     model = SimpleModel(
#         input_d=3*16*16,
#         hidden_d=256,
#         hidden_layers_n=1,
#         output_d=11
#     )
#     model.activation = act
#     model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()


#     activation_results[name], _,_ = train_simple_model(model, name, train_loader, test_loader, search_type="w", param_value=3, augmentation= augmentations)

# plt.figure(figsize=(5,4))
# plt.bar(activation_results.keys(), activation_results.values())
# plt.ylabel("Accuracy")
# plt.title("Activation Function (Best Config)")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()

# plt.savefig('raitest.png', dpi=300, bbox_inches='tight')
# plt.show()