"""
Gradient-weighted Class Activation Mapping (GradCAM) Implementation.

This module provides a GradCAM implementation for visualizing which regions
of an input image are most important for the model's predictions.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).

    Generates visual explanations for CNN predictions by computing
    weighted activations based on gradients flowing into the target layer.

    Attributes:
        model (nn.Module): The neural network model.
        target_layer (nn.Module): The convolutional layer to visualize.
        gradients (torch.Tensor): Cached gradients from backward pass.
        activations (torch.Tensor): Cached activations from forward pass.

    Args:
        model (nn.Module): A trained CNN model.
        target_layer (nn.Module): Target convolutional layer (e.g., last conv block).
    """

    def __init__(self, model, target_layer):
        """Initialize GradCAM with model and target layer, registering hooks."""
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map for the input image.

        Args:
            input_image (torch.Tensor): Input tensor of shape (1, 3, H, W).
            target_class (int, optional): Target class index. If None, uses predicted class.

        Returns:
            np.ndarray: CAM heatmap normalized to [0, 1], shape (h, w).
        """
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients (weights)
        # gradients: (1, C, H, W) -> weights: (1, C, 1, 1) -> (C,)
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        # Weighted sum of activations
        # activations: (1, C, H, W) -> (C, H, W)
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
            
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()

    def overlay_heatmap(self, image_pil, cam, alpha=0.5, colormap='jet'):
        """
        Overlay CAM heatmap on the original image.

        Args:
            image_pil (PIL.Image): Original image as PIL Image.
            cam (np.ndarray): CAM heatmap array of shape (h, w).
            alpha (float): Blending factor for heatmap. Default 0.5.
            colormap (str): Matplotlib colormap name. Default 'jet'.

        Returns:
            PIL.Image: Image with heatmap overlay.
        """
        cam_pil = Image.fromarray(cam).convert('F')
        cam_pil = cam_pil.resize(image_pil.size, resample=Image.BICUBIC)
        
        heatmap = np.array(cam_pil)
        
        # Apply Colormap
        # matplotlib cm returns (N, M, 4) floats
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap) # RGBA
        
        # Drop alpha channel and convert to 0-255 uint8
        heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # Overlay
        image_np = np.array(image_pil)
        
        # Ensure sizes match (PIL resize should handle it, but safety check or just trust)
        overlayed = (alpha * heatmap_colored + (1 - alpha) * image_np).astype(np.uint8)
        
        return Image.fromarray(overlayed)
