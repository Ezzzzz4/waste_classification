# RealWaste: Deep Learning for Landfill Waste Classification

## 🌍 Problem Statement
Efficient waste management is one of the most pressing environmental challenges of our time. Traditional waste sorting methods are often manual, labor-intensive, and prone to error, leading to low recycling rates and increased landfill usage. Automated waste classification systems can significantly improve the efficiency of recycling plants by identifying and sorting waste materials accurately and at scale.

This project uses deep learning to automate the classification of waste into 9 distinct categories, leveraging the authentic **RealWaste** dataset. The goal is to demonstrate how computer vision can be applied to real-world environmental problems, moving towards a more sustainable future.

## 📊 Dataset
This project uses the **RealWaste** dataset, a collection of authentic waste images captured at the point of reception in a landfill environment. Unlike many synthetic datasets, RealWaste contains images of waste in its real, unadulterated form, presenting a realistic challenge for computer vision models.

-   **Source**: [UCI Machine Learning Repository - RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)
-   **Size**: 4,752 images
-   **Classes**: 9 (Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation)
-   **Resolution**: 524x524 (resized for training)

## 🧠 Methodology
To achieve high accuracy on this complex real-world task, I employed a Transfer Learning approach:

1.  **Architecture**: **ResNet18**, a deep convolutional neural network, was chosen for its balance between performance and computational efficiency.
2.  **Transfer Learning**: The model was initialized with weights pre-trained on ImageNet. This allows the model to leverage learned feature extractors (edges, textures, shapes) and adapt them to the specific domain of waste classification.
3.  **Data Augmentation**: To improve generalization and prevent overfitting, I applied extensive augmentation techniques during training, including:
    -   Random Crops
    -   Horizontal Flips
    -   Color Jitter (brightness, contrast, saturation)
4.  **Handling Class Imbalance**: The dataset has varying numbers of images per class. I implemented a **Weighted Cross-Entropy Loss** function. Classes with fewer samples were assigned higher weights, ensuring the model penalizes mistakes on minority classes more heavily.

## 🚀 Results
The model was trained for 25 epochs and achieved the following performance metrics:

-   **Accuracy**: ~87% on the validation set.
-   **Comparison**: This performance is competitive with state-of-the-art benchmarks (e.g., InceptionV3 achieving ~89% on similar tasks), demonstrating the effectiveness of the ResNet18 + Weighted Loss approach.

## 🛠️ Installation & Usage

### Prerequisites
-   Python 3.8+
-   CUDA-enabled GPU (recommended for training)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/Ezzzzz4/waste_classification_resnet
    cd RealWaste
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training
To retrain the model from scratch, open `realwaste.ipynb` (or `realwaste_fixed.ipynb`) in Jupyter Notebook or Google Colab and run all cells.

### Evaluation
To evaluate the model and generate a classification report:
```bash
python evaluate_script.py
```

### Web Interface
To interact with the model using a user-friendly Gradio interface:
```bash
python app.py
```

## 📚 References & Citations

1.  **RealWaste Dataset**:
    > Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G.
2.  **Introductory Paper**:
    > Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information*, 14(12), 633. https://www.mdpi.com/2078-2489/14/12/633
3.  **UCI Machine Learning Repository**: https://archive.ics.uci.edu

## 📜 License
This project uses the RealWaste dataset, which is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
