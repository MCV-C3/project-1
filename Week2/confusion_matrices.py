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

from sklearn.svm import LinearSVC
from main import train,test,extract_features,train_with_patches,test_patches,extract_patch_features
from fisher_vector import neural_based_fisher
from IPython.display import clear_output
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="2"

os.environ['JOBLIB_TEMP_FOLDER'] = '/home/msiau/workspace/jventosa/PostTFG/Master/project-1/Week2/joblib'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

optimal_image_size = 16
transformation  = F.Compose([
                                F.ToImage(),
                                F.ToDtype(torch.float32, scale=True),
                                F.Resize(size=(optimal_image_size, optimal_image_size)),
                            ])
data_train = ImageFolder("../places_reduced/train", transform=transformation)
data_test = ImageFolder("../places_reduced/val", transform=transformation) 
train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

# model = SimpleModel(input_d=3*16*16, hidden_layers_n=1, hidden_d=256, output_d=11)
# model = model.to(device)
# model.load_state_dict(torch.load(f"SimpleModel/768_input_1_layers_256_dimension.pth", weights_only=True))

# train_feats, train_labels = extract_features(model, train_loader, device, 1)
# test_feats, test_labels = extract_features(model, test_loader, device, 1)

# svm = LinearSVC(C=1.0)
# svm.fit(train_feats, train_labels)
# svm_acc = svm.score(test_feats, test_labels)
# print(f"SVM accuracy using layer {1}: {svm_acc:.4f}")

# # Generate predictions for confusion matrix
# test_predictions = svm.predict(test_feats)

# # Compute confusion matrix
# cm = confusion_matrix(test_labels, test_predictions)

# # Get class names from the dataset
# class_names = data_test.classes

# # Display confusion matrix
# fig, ax = plt.subplots(figsize=(12, 10))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(ax=ax, cmap='Blues', values_format='d')
# plt.title(f'Confusion Matrix - SVM (Accuracy: {svm_acc:.4f})')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Print per-class accuracy
# print("\nPer-class accuracy:")
# for i, class_name in enumerate(class_names):
#     class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
#     print(f"{class_name}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    

# model = SimpleModel(input_d=3*16*16, hidden_layers_n=1, hidden_d=256, output_d=11)
# model = model.to(device)
# model.load_state_dict(torch.load(f"SimpleModel/768_input_1_layers_256_dimension.pth", weights_only=True))

# print(f"Model training mode: {model.training}")

# test_predictions = []
# test_labels = []

# model.eval()  # Ensure evaluation mode
# with torch.no_grad():  # Move this outside the loop for efficiency
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # Forward pass
#         outputs = model(inputs)
        
#         # Get predicted class (argmax of logits)
#         _, predicted = torch.max(outputs, 1)
        
#         test_predictions.append(predicted.cpu().numpy())
#         test_labels.append(labels.cpu().numpy())

# test_predictions = np.concatenate(test_predictions, axis=0)
# test_labels = np.concatenate(test_labels, axis=0)

# print(f"Predictions shape: {test_predictions.shape}")
# print(f"Labels shape: {test_labels.shape}")
# print(f"Unique predictions: {np.unique(test_predictions)}")
# print(f"Unique labels: {np.unique(test_labels)}")

# # Calculate accuracy (should match your test function)
# mlp_acc = (test_predictions == test_labels).mean()
# print(f"MLP accuracy: {mlp_acc:.4f}")

# # Also verify using your original test function
# criterion = torch.nn.CrossEntropyLoss()  # Make sure this matches your training
# test_loss, test_accuracy = test(model, test_loader, criterion, device)
# print(f"Test function accuracy: {test_accuracy:.4f}")
# print(f"Test function loss: {test_loss:.4f}")

# # Compute confusion matrix
# cm = confusion_matrix(test_labels, test_predictions)

# # Get class names from the dataset
# class_names = data_test.classes

# # Display confusion matrix
# fig, ax = plt.subplots(figsize=(12, 10))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(ax=ax, cmap='Blues', values_format='d')
# plt.title(f'Confusion Matrix - MLP (Accuracy: {mlp_acc:.4f})')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('confusion_matrix_mlp.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print per-class accuracy
# print("\nPer-class accuracy:")
# for i, class_name in enumerate(class_names):
#     class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
#     print(f"{class_name}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

transformation  = F.Compose([
                                F.ToImage(),
                                F.ToDtype(torch.float32, scale=True),
                                F.Resize(size=(256, 256)),
                            ])

data_train = ImageFolder("../places_reduced/train", transform=transformation)
data_test = ImageFolder("../places_reduced/val", transform=transformation) 

train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

# Patch parameters (make sure these match your training!)
patch_size = 8  # Adjust this to match your training
stride = 8      # Adjust this to match your training (None means stride=patch_size)
aggregation = 'mean'  # Options: 'mean', 'vote', 'max'

if stride is None:
    stride = patch_size
model = SimpleModel(input_d=3*8*8, hidden_layers_n=1, hidden_d=256, output_d=11)
test_predictions = []
test_labels_list = []
model.load_state_dict(torch.load(f"SimpleModel/8_patchsize_1_layers_256_dimension.pth", weights_only=True))

model.to(device)
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
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
            aggregated_outputs = outputs.mean(dim=1)
        elif aggregation == 'vote':
            patch_predictions = outputs.argmax(dim=2)
            aggregated_outputs = torch.zeros(batch_size, num_classes, device=device)
            for b in range(batch_size):
                for pred in patch_predictions[b]:
                    aggregated_outputs[b, pred] += 1
        elif aggregation == 'max':
            aggregated_outputs = outputs.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Get final predictions
        _, predicted = aggregated_outputs.max(1)
        
        test_predictions.append(predicted.cpu().numpy())
        test_labels_list.append(labels.cpu().numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_labels = np.concatenate(test_labels_list, axis=0)

print(f"Predictions shape: {test_predictions.shape}")
print(f"Labels shape: {test_labels.shape}")

# Calculate accuracy
mlp_acc = (test_predictions == test_labels).mean()
print(f"MLP accuracy (patch-based): {mlp_acc:.4f}")

# Verify with your test function
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_accuracy = test_patches(model, test_loader, criterion, device, 
                                        patch_size=patch_size, stride=stride, 
                                        aggregation=aggregation)
print(f"Test function accuracy: {test_accuracy:.4f}")
print(f"Test function loss: {test_loss:.4f}")

# Compute confusion matrix
cm = confusion_matrix(test_labels, test_predictions)

# Get class names from the dataset
class_names = data_test.classes

# Display confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix - MLP Patch-based (Accuracy: {mlp_acc:.4f})\n'
          f'Patch size: {patch_size}, Stride: {stride}, Aggregation: {aggregation}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'confusion_matrix_mlp_patch_{aggregation}.png', dpi=300, bbox_inches='tight')
plt.show()

# Print per-class accuracy
print("\nPer-class accuracy:")
for i, class_name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"{class_name}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
