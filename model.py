"""
Waste Classification Model Definition.

This module defines the WasteClassifier, a CNN-based image classifier
for the RealWaste dataset using transfer learning from EfficientNet-B0.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class WasteClassifier(nn.Module):
    """
    Waste image classifier using EfficientNet-B0 as backbone.

    This model leverages transfer learning from ImageNet-pretrained
    EfficientNet-B0 and replaces the final classifier layer to output
    predictions for 9 waste categories.

    Attributes:
        backbone (nn.Module): EfficientNet-B0 model with modified classifier.

    Args:
        num_classes (int): Number of output classes. Default is 9 for RealWaste dataset.
    """

    def __init__(self, num_classes: int = 9):
        """Initialize the WasteClassifier with pretrained EfficientNet-B0."""
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # Replace the final classification layer
        original_in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(original_in_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (N, num_classes).
        """
        return self.backbone(x)
