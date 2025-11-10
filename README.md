Plant Disease Detection using CNN (MobileNetV2 + PyTorch)

A deep learning project that automatically detects plant diseases from leaf images using a Convolutional Neural Network (CNN) based on MobileNetV2 architecture.
This system helps farmers, researchers, and agricultural professionals quickly identify diseases and take preventive measures — all through image-based diagnosis.

Project Overview

This AI model classifies plant leaf images into healthy or diseased categories using transfer learning on MobileNetV2, a lightweight and efficient CNN model optimized for mobile and embedded applications.

The project includes:
  Model Training (PyTorch)
  Dataset Preprocessing
  Evaluation and Accuracy Analysis
  Inference / Prediction Script
  Visualization of Training Results

Features
  Uses MobileNetV2 for efficient performance
  Trained on custom dataset of plant leaves
  Achieves high accuracy with minimal computational cost
  Modular PyTorch-based implementation
  Easily scalable for multiple crops and diseases

⚙️ Installation
1) Clone the Repository
    git clone https://github.com/mranshsisodiya/Plant-Disease-Detection-Using-CNN.git
    cd Plant-Disease-Detection-Using-CNN
2) Create a Virtual Environment
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # source venv/bin/activate   # On Mac/Linux

Model Architecture

The project uses MobileNetV2, a CNN model optimized for lightweight image classification.
It applies depthwise separable convolutions to reduce computation while maintaining high accuracy.
Key parameters:
Optimizer: Adam
Loss Function: CrossEntropyLoss
Epochs: 15
Batch Size: 32
Image Size: 224x224

Training
  To train the model, run:
  python src/training.py

Prediction
  To test or predict new images:
  python src/prediction.py


Future Improvements
 Add real-time webcam-based disease detection
 Deploy the model using Flask or Streamlit
 Expand dataset with more crop species
 Convert to TensorFlow Lite for mobile devices

Why MobileNetV2?
  Lightweight & Fast for deployment
  Excellent for edge/mobile AI applications
  Retains high accuracy with fewer parameters

Contributors
 Ansh Sisodiya
 mranshsisodiya@gmail.com
 GitHub: mranshsisodiya

License
  This project is released under the MIT License — feel free to use, modify, and distribute it.


Show Your Support
  If you find this project useful, please ⭐ the repository to support development and improvements!


