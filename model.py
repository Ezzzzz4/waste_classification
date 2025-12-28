import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # Initialize EfficientNet-B0 with ImageNet weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # EfficientNet's classifier is a Sequential block. 
        # The last layer is '1' (Dropout is '0').
        original_in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(original_in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
