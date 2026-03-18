<<<<<<< HEAD
# Explainable Crack Detection in Concrete
=======
# Explainable Crack Detection in Civil Infrastructures 
>>>>>>> b48b3055 (Complete project with all source code and updates)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://explainablecrackdetectioncnn.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning system that detects cracks in concrete structures using **MobileNet + CBAM attention** with **Grad-CAM explainability**. The model provides visual heatmaps and AI‑generated explanations.

## ✨ Features
- 🧠 **MobileNet backbone** + **CBAM attention** for accurate crack detection
- 🔥 **Grad-CAM heatmaps** showing model focus areas
- 🤖 **Gemini AI** explanations of results
- 🌐 **Web interface** with image upload and history
- 📄 **PDF/TXT report download** with embedded images

## 🚀 Live Demo
Try it now: [https://explainablecrackdetectioncnn.onrender.com/]

## 📦 Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app/app.py`

## 🧠 Model Architecture
- Base: MobileNet (pretrained on ImageNet)
- Attention: Convolutional Block Attention Module (CBAM)
- Classifier: Global pooling + Dense layer
- Explainability: Grad-CAM

## 📊 Dataset
The model was trained on the [Concrete Crack Images dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (40,000 images).

## 📄 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
