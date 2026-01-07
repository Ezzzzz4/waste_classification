import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import WasteClassifier

# Constants from original training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
BATCH_SIZE = 32
RANDOM_SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --- Data Augmentation Definitions ---
def get_transforms(mode='full'):
    """
    Returns (train_transforms, val_transforms) based on mode.
    mode: 'full' (composite), 'baseline' (basic)
    """
    if mode == 'full':
        train_t = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.8, 1.2)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else: # baseline
        train_t = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224), # Standard simple augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return train_t, val_t

# --- Dataset Wrapper ---
class TransformedDataset(torch.utils.data.Dataset):
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

# --- Mixup Utils ---
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Experiment Runner ---
def run_experiment(exp_name, epochs, use_wandb):
    print(f"\n{'='*40}\nRunning Experiment: {exp_name}\n{'='*40}")
    
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)
    
    # Feature Flags based on Experiment Name
    # Experiments: 'full', 'no_mixup', 'no_ls', 'no_weights', 'baseline'
    
    use_mixup = exp_name in ['full', 'no_ls', 'no_weights']
    use_ls = exp_name in ['full', 'no_mixup', 'no_weights']
    use_weights = exp_name in ['full', 'no_mixup', 'no_ls']
    use_composite_augs = exp_name != 'baseline'
    
    # Baseline has NO mixup, NO ls, NO weights, BASIC augs
    if exp_name == 'baseline':
        use_mixup = False
        use_ls = False
        use_weights = False
        use_composite_augs = False

    print(f"Config: Mixup={use_mixup}, LabelSmoothing={use_ls}, ClassWeights={use_weights}, CompositeAugs={use_composite_augs}")

    # 2. Data Loading (Stratified Split)
    data_dir = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(os.path.join(data_dir, 'Metal')):
         if os.path.exists(os.path.join(data_dir, 'RealWaste', 'Metal')):
            data_dir = os.path.join(data_dir, 'RealWaste')
            
    base_dataset = ImageFolder(root=data_dir, transform=None)
    
    # Stratified Split (Replicating exact logic from train_model.ipynb)
    class_indices = {i: [] for i in range(len(base_dataset.classes))}
    for idx, (_, label) in enumerate(base_dataset.samples):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        train_idx, temp_idx = train_test_split(
            indices, train_size=0.8, random_state=42, stratify=[label]*len(indices)
        )
        val_idx, _ = train_test_split(
            temp_idx, train_size=0.5, random_state=42, stratify=[label]*len(temp_idx)
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    # Transforms
    train_transform, val_transform = get_transforms('full' if use_composite_augs else 'baseline')
    
    train_dataset = TransformedDataset(Subset(base_dataset, train_indices), train_transform)
    val_dataset = TransformedDataset(Subset(base_dataset, val_indices), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Model & Optimization
    model = WasteClassifier(num_classes=len(base_dataset.classes))
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function Config
    weight_tensor = None
    if use_weights:
        class_counts = [len(indices) for indices in class_indices.values()]
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        weight_tensor = torch.FloatTensor(class_weights).to(device)
    
    ls_val = 0.1 if use_ls else 0.0
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=ls_val)

    # WandB Init
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="waste-classification-ablation", name=exp_name, reinit=True, config={
            "experiment": exp_name,
            "mixup": use_mixup,
            "label_smoothing": use_ls,
            "class_weights": use_weights,
            "composite_augs": use_composite_augs,
            "epochs": epochs
        })

    # 4. Training Loop
    save_dir = os.path.join('weights', f'ablation_{exp_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"[{exp_name}] Epoch {epoch+1}/{epochs} Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                # Accuracy calculation for mixup
                _, predicted = torch.max(outputs.data, 1)
                mixup_correct = (lam * predicted.eq(targets_a.data).cpu().sum().float() + 
                                  (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                train_correct += mixup_correct.item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0) # Accumulate scaled loss
            total_train += images.size(0)
            
        train_loss = train_loss / total_train
        train_acc = train_correct / total_train
        scheduler.step()

        # VALIDATION
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total_val += images.size(0)
        
        val_loss = val_loss / total_val
        val_acc = val_correct / total_val
        
        # Logging
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "epoch": epoch+1
            })

        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
    # Save Metrics & Plots
    plot_results(history, save_dir)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f)
        
    print(f"Experiment {exp_name} Finished. Best Val Acc: {best_val_acc:.4f}")
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

def plot_results(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study Experiments")
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'full', 'no_mixup', 'no_ls', 'no_weights', 'baseline'],
                        help='Experiment to run')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs per experiment')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    
    experiments = []
    if args.experiment == 'all':
        experiments = ['baseline', 'no_weights', 'no_ls', 'no_mixup', 'full']
    else:
        experiments = [args.experiment]
        
    use_wandb = not args.no_wandb
    
    for exp in experiments:
        run_experiment(exp, args.epochs, use_wandb)
