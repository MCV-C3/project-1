from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Retrieve Layer
        #layer_2 = model.recover_layer(inputs,2)

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

def train_with_patches(model, dataloader, criterion, optimizer, device, patch_size, stride=None):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0
    
    if stride is None:
        stride = patch_size

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size, channels, height, width = inputs.shape
        
        patches = []
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = inputs[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  
        num_patches = patches.shape[1]
        patches = patches.view(-1, channels, patch_size, patch_size) 
        
        labels_expanded = labels.unsqueeze(1).expand(-1, num_patches).reshape(-1)
        
        outputs = model(patches)
        loss = criterion(outputs, labels_expanded)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size  
        _, predicted = outputs.max(1)
        correct += (predicted == labels_expanded).sum().item()
        total += labels_expanded.size(0)

    avg_loss = train_loss / len(dataloader.dataset)
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

def test_patches(model, dataloader, criterion, device, patch_size, stride=None, aggregation='mean'):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    
    if stride is None:
        stride = patch_size

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size, channels, height, width = inputs.shape
            
            # Extract patches from each image in the batch
            patches = []
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    patch = inputs[:, :, i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
            
            # Stack patches: (batch, num_patches, 3, patch_size, patch_size)
            patches = torch.stack(patches, dim=1)
            num_patches = patches.shape[1]
            patches = patches.view(-1, channels, patch_size, patch_size)
            
            # Forward pass on all patches
            outputs = model(patches)  # (batch * num_patches, num_classes)
            
            # Reshape outputs back to (batch, num_patches, num_classes)
            num_classes = outputs.shape[1]
            outputs = outputs.view(batch_size, num_patches, num_classes)
            
            # Aggregate predictions across patches for each image
            if aggregation == 'mean':
                # Average the logits/probabilities across patches
                aggregated_outputs = outputs.mean(dim=1)  # (batch, num_classes)
            elif aggregation == 'vote':
                # Majority voting: each patch votes for a class
                patch_predictions = outputs.argmax(dim=2)  # (batch, num_patches)
                aggregated_outputs = torch.zeros(batch_size, num_classes, device=device)
                for b in range(batch_size):
                    for pred in patch_predictions[b]:
                        aggregated_outputs[b, pred] += 1
            elif aggregation == 'max':
                # Max pooling: take maximum confidence across patches
                aggregated_outputs = outputs.max(dim=1)[0]  # (batch, num_classes)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            # Calculate loss on aggregated outputs
            loss = criterion(aggregated_outputs, labels)
            
            # Track loss and accuracy
            test_loss += loss.item() * batch_size
            _, predicted = aggregated_outputs.max(1)
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

def extract_features(model, dataloader, device, layer_id):
    model.eval()
    feats, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            f = model.recover_layer(imgs, layer_id)
            feats.append(f.cpu().numpy())
            labels_list.append(labels.numpy())

    feats = np.vstack(feats)
    labels = np.hstack(labels_list)
    return feats, labels

def extract_patch_features(model, dataloader, device, layer_id, patch_size, stride=None):
    model.eval()
    feats, labels_list = [], []
    
    if stride is None:
        stride = patch_size

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            batch_size, channels, height, width = imgs.shape
            
            # Extract patches from each image in the batch
            patches = []
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    patch = imgs[:, :, i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
            
            # Stack patches: (batch, num_patches, 3, patch_size, patch_size)
            patches = torch.stack(patches, dim=1)
            num_patches = patches.shape[1]
            patches = patches.view(-1, channels, patch_size, patch_size)
            
            # Extract features from all patches
            patch_features = model.recover_layer(patches, layer_id)  # (batch * num_patches, feature_dim)
            
            # Reshape to (batch, num_patches, feature_dim)
            feature_dim = patch_features.shape[1]
            patch_features = patch_features.view(batch_size, num_patches, feature_dim)
            
            feats.append(patch_features.cpu().numpy())
            labels_list.append(labels.numpy())

    feats = np.vstack(feats)  # (total_images, num_patches, feature_dim)
    labels = np.hstack(labels_list)
    return feats, labels

if __name__ == "__main__":

    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder("../places_reduced/train", transform=transformation)
    data_test = ImageFolder("../places_reduced/val", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = SimpleModel(input_d=C*H*W,hidden_layers_n=2, hidden_d=300, output_d=11)
    # plot_computational_graph(model, input_size=(1, C*H*W))  # Batch size of 1, input_dim=10

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")

    from sklearn.svm import LinearSVC

    # Choose which layer to use (0, 1, 2, ...)
    layer_id = 1

    train_feats, train_labels = extract_features(model, train_loader, device, layer_id)
    test_feats, test_labels = extract_features(model, test_loader, device, layer_id)

    svm = LinearSVC(C=1.0)
    svm.fit(train_feats, train_labels)
    svm_acc = svm.score(test_feats, test_labels)

    print(f"SVM accuracy using layer {layer_id}: {svm_acc:.4f}")
