import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json

# --- 1. Model Definition (Must match training) ---
from model import WasteClassifier

# --- 2. Setup Device & Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define Classes (Alphabetical as per ImageFolder)
# Hardcoding these for consistency with the training data structure
CLASSES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 
           'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

model = WasteClassifier(num_classes=len(CLASSES))
weights_path = 'best_waste_model.pth'

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

model.to(device)
model.eval()

# --- 3. Preprocessing ---
# Same statistics used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),         # Resize shortest side to 256
    transforms.CenterCrop(224),     # Crop center 224x224
    transforms.ToTensor(),          # Convert to Tensor (0-1)
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) # Normalize
])

# --- 4. Predictive Function ---
def predict_waste(image):
    if image is None:
        return None
    
    # 1. Transform Image
    # Gradio passes a PIL Image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 2. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # 3. Format Output
    # Gradio expects a dictionary {label: confidence}
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    return confidences

# --- 5. Helper to Read Reports ---
def get_classification_report():
    if os.path.exists('classification_report.txt'):
        with open('classification_report.txt', 'r') as f:
            return f.read()
    return "Report not generated yet."

def get_confusion_matrix():
    if os.path.exists('confusion_matrix.png'):
        return 'confusion_matrix.png'
    return None

# --- 6. Build Gradio Interface ---
with gr.Blocks(title="RealWaste AI Classifier", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ♻️ RealWaste AI Classifier")
    gr.Markdown("Upload an image of waste to classify it into one of 9 categories.")
    
    with gr.Tabs():
        # TAB 1: Prediction
        with gr.Tab("Classify Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Waste Image")
                    classify_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column():
                    label_output = gr.Label(num_top_classes=3, label="Predictions")
            
            classify_btn.click(fn=predict_waste, inputs=input_image, outputs=label_output)
            
            # Examples (if any exist in datasets, picking a few random ones would be nice, but optional)
            # gr.Examples(examples=["dataset/RealWaste/Glass/glass1.jpg"], inputs=input_image)

        # TAB 2: Model Stats
        with gr.Tab("Model Performance"):
            gr.Markdown("## Evaluation Metrics")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Classification Report")
                    report_text = gr.Code(value=get_classification_report, language="markdown", label="Metrics")
                with gr.Column():
                    gr.Markdown("### Confusion Matrix")
                    cm_image = gr.Image(value=get_confusion_matrix, label="Confusion Matrix", show_label=False)

if __name__ == "__main__":
    app.launch()
