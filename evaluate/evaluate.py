"""
Waste Classification Model Evaluation Script.

This script evaluates any trained WasteClassifier on the test set, generating:
- Confusion matrices (PNG)
- Classification reports (TXT)
- GradCAM visualizations (optional)

Usage:
    python evaluate.py                          # Evaluate best_waste_model.pth
    python evaluate.py --model full             # Evaluate ablation_full model
    python evaluate.py --model baseline         # Evaluate ablation_baseline model
    python evaluate.py --all                    # Evaluate all ablation models
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import argparse

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import WasteClassifier

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model configurations
MODELS = {
    'best': ('weights/best_waste_model.pth', 'Best Model'),
    'baseline': ('weights/ablation_baseline/best_model.pth', 'Baseline'),
    'no_weights': ('weights/ablation_no_weights/best_model.pth', 'No Weights'),
    'no_ls': ('weights/ablation_no_ls/best_model.pth', 'No Label Smoothing'),
    'no_mixup': ('weights/ablation_no_mixup/best_model.pth', 'No Mixup'),
    'full': ('weights/ablation_full/best_model.pth', 'Full'),
}


class TransformedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies transforms to a subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_test_loader(data_dir, batch_size=32):
    """Load test set using stratified split (matching training)."""
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    base_dataset = ImageFolder(root=data_dir, transform=None)
    
    # Stratified split (replicate training logic)
    class_indices = {i: [] for i in range(len(base_dataset.classes))}
    for idx, (_, label) in enumerate(base_dataset.samples):
        class_indices[label].append(idx)

    test_indices = []
    for label, indices in class_indices.items():
        train_idx, temp_idx = train_test_split(
            indices, train_size=0.8, random_state=42, stratify=[label]*len(indices)
        )
        _, test_idx = train_test_split(
            temp_idx, train_size=0.5, random_state=42, stratify=[label]*len(temp_idx)
        )
        test_indices.extend(test_idx)

    test_dataset = TransformedDataset(Subset(base_dataset, test_indices), val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return test_loader, base_dataset.classes


def evaluate_model(model, loader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def generate_confusion_matrix(y_true, y_pred, classes, save_path, title):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_classification_report(y_true, y_pred, classes, save_path):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, target_names=classes)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {save_path}")
    return report


def evaluate_single_model(model_key, test_loader, classes, device, output_dir='evaluate'):
    """Evaluate a single model and save results."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available models: {list(MODELS.keys())}")
        return None
    
    weights_path, model_name = MODELS[model_key]
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return None
    
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"Weights: {weights_path}")
    print(f"{'='*50}")
    
    # Load model
    model = WasteClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    
    # Evaluate
    y_true, y_pred = evaluate_model(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_key}.png')
    generate_confusion_matrix(y_true, y_pred, classes, cm_path, 
                              f'Confusion Matrix - {model_name}')
    
    # Save classification report
    report_path = os.path.join(output_dir, f'classification_report_{model_key}.txt')
    report = generate_classification_report(y_true, y_pred, classes, report_path)
    
    print("\nClassification Report:")
    print(report)
    
    return {
        'model': model_key,
        'name': model_name,
        'test_acc': float(test_acc)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Waste Classification Models")
    parser.add_argument('--model', type=str, default='best',
                        choices=list(MODELS.keys()),
                        help='Model to evaluate (default: best)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all ablation models')
    parser.add_argument('--output', type=str, default='evaluate',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(os.path.join(data_dir, 'Metal')):
        if os.path.exists(os.path.join(data_dir, 'RealWaste', 'Metal')):
            data_dir = os.path.join(data_dir, 'RealWaste')
    
    # Load test data
    test_loader, classes = get_test_loader(data_dir)
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Classes: {classes}")
    
    results = []
    
    if args.all:
        # Evaluate all models
        for model_key in MODELS.keys():
            result = evaluate_single_model(model_key, test_loader, classes, device, args.output)
            if result:
                results.append(result)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Test Accuracy':<15}")
        print("-"*60)
        for r in results:
            print(f"{r['name']:<20} {r['test_acc']*100:.2f}%")
        print("="*60)
        
        # Save summary
        summary_path = os.path.join(args.output, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved summary: {summary_path}")
    else:
        # Evaluate single model
        evaluate_single_model(args.model, test_loader, classes, device, args.output)


if __name__ == "__main__":
    main()
