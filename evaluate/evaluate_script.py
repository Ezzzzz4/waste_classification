import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import random
import sys

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fix path to import model from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 1. Model Definition
from model import WasteClassifier
from utils.gradcam import GradCAM


def main():
    # Initialize model
    model = WasteClassifier(num_classes=9)
    model = model.to(device)

    # Load weights 
    weights_path = os.path.join(parent_dir, 'weights', 'best_waste_model.pth')
    

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Model weights loaded successfully from {weights_path}")
    else:
        print(f"Error: {weights_path} not found.")
        return
        
    model.eval()

    # 2. Dataset and Transforms
    data_dir = os.path.join(parent_dir, "dataset") # Adjusted path assumption based on previous context

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Classes found: {classes}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    full_dataset = ImageFolder(root=data_dir, transform=val_transforms)
    
    # Stratified Split (80% Train, 10% Val, 10% Test) - Matching train_model.ipynb
    from sklearn.model_selection import train_test_split
    
    # Get indices per class
    class_indices = {i: [] for i in range(len(full_dataset.classes))}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    test_indices = []

    # Replicate the split logic
    for label, indices in class_indices.items():
        # First split: 80% Train, 20% Temp (Val + Test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=0.8, random_state=42, stratify=[label]*len(indices)
        )
        # Second split: Split Temp into 50% Val, 50% Test (which matches 10% Val, 10% Test of total)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=0.5, random_state=42, stratify=[label]*len(temp_idx)
        )
        test_indices.extend(test_idx)

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Test set size: {len(test_indices)}")
    
    # Evaluate on the dedicated test set
    eval_dataset = Subset(full_dataset, test_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # 3. Visual Predictions
    print("Generating prediction visualization...")
    visualize_predictions(full_dataset, model, classes, IMAGENET_MEAN, IMAGENET_STD)
    
    # 4. Quantitative Evaluation
    print("Running quantitative evaluation...")
    y_true, y_pred = evaluate_model(model, eval_loader)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(current_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Saved {cm_path}")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    report_path = os.path.join(current_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved {report_path}")

def denormalize(tensor, mean, std):
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

def visualize_predictions(dataset, model, classes, mean, std, num_images=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    indices = random.sample(range(len(dataset)), num_images)
    
    # Initialize GradCAM
    # Target layer for EfficientNet-B0 is usually features[-1] (the last conv block)
    target_layer = model.backbone.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Prepare input
        input_tensor = image.unsqueeze(0).to(device)
        
        # Generate heatmap
        cam = grad_cam.generate_cam(input_tensor)
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_idx = predicted.item()
            
        ax = axes[i]
        
        # Convert image for display (denormalize)
        img_display_np = denormalize(image, mean, std) # Returns numpy (H, W, 3)
        img_display_pil = Image.fromarray((img_display_np * 255).astype(np.uint8))
        
        # Overlay Heatmap
        overlayed_img = grad_cam.overlay_heatmap(img_display_pil, cam, alpha=0.5)
        
        ax.imshow(overlayed_img)
        
        color = 'green' if predicted_idx == label else 'red'
        ax.set_title(f"True: {classes[label]}\nPred: {classes[predicted_idx]}", color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    pred_path = os.path.join(current_dir, 'prediction_gradcam.png')
    plt.savefig(pred_path)
    print(f"Saved {pred_path}")


def evaluate_model(model, loader):
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

if __name__ == "__main__":
    main()
