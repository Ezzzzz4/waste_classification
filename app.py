"""
Waste Classification Web Application.

A Gradio-based interface for real-time waste classification with GradCAM visualization.

Usage:
    python app.py
"""

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os

from model import WasteClassifier
from utils.gradcam import GradCAM


# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CLASSES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 
           'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Available models for selection
AVAILABLE_MODELS = {
    'Best Model (Baseline)': 'weights/best_waste_model.pth',
    'Ablation: Baseline': 'weights/ablation_baseline/best_model.pth',
    'Ablation: Full': 'weights/ablation_full/best_model.pth',
    'Ablation: No Mixup': 'weights/ablation_no_mixup/best_model.pth',
    'Ablation: No Label Smoothing': 'weights/ablation_no_ls/best_model.pth',
    'Ablation: No Weights': 'weights/ablation_no_weights/best_model.pth',
}

# Default model
DEFAULT_MODEL = 'Best Model (Baseline)'

# Global model state
current_model = None
current_model_name = None
grad_cam = None


def load_model(model_name):
    """Load a model by name."""
    global current_model, current_model_name, grad_cam
    
    if model_name == current_model_name and current_model is not None:
        return f"Model already loaded: {model_name}"
    
    weights_path = AVAILABLE_MODELS.get(model_name)
    if not weights_path or not os.path.exists(weights_path):
        return f"Model not found: {weights_path}"
    
    model = WasteClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    current_model = model
    current_model_name = model_name
    
    # Initialize GradCAM
    target_layer = model.backbone.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    print(f"Loaded model: {model_name}")
    return f"✅ Loaded: {model_name}"


# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


def predict_waste(image, model_name):
    """Classify waste image and generate GradCAM heatmap."""
    global current_model, grad_cam
    
    if image is None:
        return None, None, "Please upload an image."
    
    # Load model if needed
    if model_name != current_model_name or current_model is None:
        load_model(model_name)
    
    if current_model is None:
        return None, None, "Failed to load model."
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = current_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Format output
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    # Generate GradCAM heatmap
    cam = grad_cam.generate_cam(input_tensor)
    overlayed_img = grad_cam.overlay_heatmap(image, cam, alpha=0.5)
    
    return confidences, overlayed_img, f"Model: {model_name}"


def get_classification_report():
    """Read classification report from file."""
    report_path = os.path.join('evaluate', 'classification_report_best.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return f.read()
    # Fallback to old path
    report_path = os.path.join('evaluate', 'classification_report.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return f.read()
    return "Report not generated. Run: python evaluate/evaluate.py"


def get_confusion_matrix():
    """Get confusion matrix image path."""
    cm_path = os.path.join('evaluate', 'confusion_matrix_best.png')
    if os.path.exists(cm_path):
        return cm_path
    # Fallback
    cm_path = os.path.join('evaluate', 'confusion_matrix.png')
    if os.path.exists(cm_path):
        return cm_path
    return None


# --- Build Interface ---
with gr.Blocks(title="Waste AI Classifier", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ♻️ Waste AI Classifier")
    gr.Markdown("Upload an image of waste to classify it into one of 9 categories.")
    
    with gr.Tabs():
        # TAB 1: Classification
        with gr.Tab("Classify Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Waste Image")
                    model_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL,
                        label="Select Model"
                    )
                    classify_btn = gr.Button("Classify", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    label_output = gr.Label(num_top_classes=3, label="Predictions")
                    heatmap_output = gr.Image(label="Grad-CAM Heatmap")
            
            classify_btn.click(
                fn=predict_waste, 
                inputs=[input_image, model_dropdown], 
                outputs=[label_output, heatmap_output, status_text]
            )

        # TAB 2: Model Performance
        with gr.Tab("Model Performance"):
            gr.Markdown("## Evaluation Metrics")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Classification Report")
                    report_text = gr.Code(value=get_classification_report, language="markdown", label="Metrics")
                with gr.Column():
                    gr.Markdown("### Confusion Matrix")
                    cm_image = gr.Image(value=get_confusion_matrix, label="Confusion Matrix", show_label=False)

# Load default model on startup
load_model(DEFAULT_MODEL)

if __name__ == "__main__":
    app.launch()
