import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are typically tuple (grad_output,), we want the tensor
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        # input_image: (1, 3, H, W)
        
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
        # image_pil: PIL Image (H, W, 3)
        # cam: numpy array (h, w)
        
        # Resize CAM to image size using PIL
        # Convert cam to PIL image for resizing
        # We need to map 0-1 float to 0-255 uint8 first for visualization if we use PIL directly,
        # but here we want to resize the float map first then colormap it.
        # Actually simplest to resize with PIL, but PIL resize expects uint8 or similar usually for images,
        # but supports F (float).
        
        # Ensure cam is float32
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
