import torch.nn as nn
from torchvision import models

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=9, weights=None):
        super().__init__()
        # Use weights argument for modern torchvision compatibility
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
