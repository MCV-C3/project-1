import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from sklearn.metrics import confusion_matrix, classification_report
from models import WraperModel
import argparse
from collections import defaultdict
import os

def load_model(model_path, num_classes=8, device='cuda'):
    """Load a saved model from path"""
    model = WraperModel(
        num_classes=num_classes,
        feature_extraction=False,
        batch_norm=False,
        dropout=True,
        dropout_prob=0.5,
        classifier_type="FCN"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataset, device, max_misclassified_per_class=5):
    """
    Evaluate model and collect predictions with file paths
    
    Returns:
        y_true: true labels
        y_pred: predicted labels
        misclassified: dict mapping (true_class, pred_class) -> list of file paths
    """
    y_true = []
    y_pred = []
    misclassified = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            img_path = dataset.imgs[idx][0]
            
            # Add batch dimension and move to device
            img = img.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(img)
            _, predicted = output.max(1)
            pred_label = predicted.item()
            
            y_true.append(label)
            y_pred.append(pred_label)
            
            # Track misclassified samples
            if label != pred_label:
                key = (label, pred_label)
                if len(misclassified[key]) < max_misclassified_per_class:
                    misclassified[key].append(img_path)
    
    return np.array(y_true), np.array(y_pred), dict(misclassified)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm

def print_misclassified_samples(misclassified, class_names):
    """Print misclassified samples in a readable format"""
    print("\n" + "="*80)
    print("MISCLASSIFIED SAMPLES")
    print("="*80)
    
    for (true_label, pred_label), file_paths in sorted(misclassified.items()):
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        
        print(f"\n{true_class} → {pred_class} ({len(file_paths)} samples shown):")
        print("-" * 80)
        
        for i, path in enumerate(file_paths, 1):
            filename = os.path.basename(path)
            print(f"  {i}. {filename}")
            print(f"     Path: {path}")

def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model and generate confusion matrix')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test dataset folder')
    parser.add_argument('--gpu_index', type=str, default='0',
                        help='GPU index to use')
    parser.add_argument('--max_misclassified', type=int, default=5,
                        help='Maximum misclassified samples to show per class pair')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading dataset from {args.data_path}...")
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])
    
    dataset = ImageFolder(args.data_path, transform=transformation)
    class_names = dataset.classes
    print(f"Found {len(dataset)} images in {len(class_names)} classes: {class_names}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, num_classes=len(class_names), device=device)
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, misclassified = evaluate_model(
        model, dataset, device, 
        max_misclassified_per_class=args.max_misclassified
    )
    
    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({(y_true == y_pred).sum()}/{len(y_true)})")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    cm = plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    
    # Print classification report
    print_classification_report(y_true, y_pred, class_names)
    
    # Print misclassified samples
    print_misclassified_samples(misclassified, class_names)
    
    # Save misclassified samples to file
    misclassified_path = os.path.join(args.output_dir, 'misclassified_samples.txt')
    with open(misclassified_path, 'w') as f:
        f.write("MISCLASSIFIED SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for (true_label, pred_label), file_paths in sorted(misclassified.items()):
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            
            f.write(f"{true_class} → {pred_class} ({len(file_paths)} samples):\n")
            f.write("-" * 80 + "\n")
            
            for i, path in enumerate(file_paths, 1):
                f.write(f"  {i}. {os.path.basename(path)}\n")
                f.write(f"     {path}\n")
            f.write("\n")
    
    print(f"\nMisclassified samples saved to {misclassified_path}")
    print(f"All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()