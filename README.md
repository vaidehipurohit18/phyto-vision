# PLANT DISEASE DETECTION AND CLASSIFICATION USING ATTENTION-BASED CNN

> **B.Tech Major Project** • Integrated Deep Learning System with MobileNetV2, CBAM Attention Architecture, Flask Backend, SQLite Database, Explainable AI (Grad-CAM), Chart.js Dashboard, and Disease Encyclopedia.

---

## 📌 Project Overview

This project presents an end-to-end, production-ready computer vision application for detecting and classifying plant leaf pathologies. By integrating a custom **Convolutional Block Attention Module (CBAM)** into a pre-trained **MobileNetV2** backbone, the system adaptively emphasizes subtle lesion boundaries and foliage discolorations while suppressing irrelevant background soil and greenhouse clutter.

Unlike basic prototype applications, this web app features **zero hardcoded predictions or fake AI metrics**. All inference results are dynamically computed from the trained TensorFlow/Keras model binaries.

---

## 🛠️ Technology Stack

| Layer | Technology Used |
| :--- | :--- |
| **Frontend** | HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+), Chart.js (CDN) |
| **Backend** | Python 3, Flask Web Server, Gunicorn |
| **AI / Machine Learning** | TensorFlow 2.x, Keras, NumPy, Pillow, OpenCV |
| **Database** | SQLite 3 |
| **Explainable AI** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **PWA & Cache** | Web App Manifest (`manifest.json`), Service Worker (`service-worker.js`) |
| **Testing** | Pytest |

---

## 📐 Model Architecture & CBAM Attention Module

The system utilizes a hybrid **MobileNetV2 + CBAM Attention** architecture:

```text
Input Leaf Image (224x224x3)
           │
           ▼
 MobileNetV2 Backbone Feature Extractor
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │  Convolutional Block Attention Module (CBAM) │
 │                                              │
 │  1. Channel Attention (AvgPool + MaxPool)    │
 │     --> Shared MLP (Dense -> ReLU -> Dense)  │
 │     --> Sigmoid Activation                   │
 │                                              │
 │  2. Spatial Attention (Channel-wise Pool)    │
 │     --> 7x7 Conv2D -> Sigmoid Activation     │
 └──────────────────────────────────────────────┘
           │
           ▼
 Global Average Pooling (GAP)
           │
           ▼
 Dropout (0.3)
           │
           ▼
 Dense Classification Head (Softmax Outputs)
```

### CBAM Mathematical Equations

1. **Channel Attention**:
   $$\mathbf{M}_c(\mathbf{F}) = \sigma \left( W_1 (W_0 (\mathbf{F}_{\text{avg}}^c)) + W_1 (W_0 (\mathbf{F}_{\text{max}}^c)) \right)$$
2. **Spatial Attention**:
   $$\mathbf{M}_s(\mathbf{F}') = \sigma \left( f^{7 \times 7} \left( [ \mathbf{F}_{\text{avg}}^s ; \mathbf{F}_{\text{max}}^s ] \right) \right)$$

---

## 📁 Project Structure

```text
plant-disease-detection/
├── app.py                      # Main Flask Web Controller & API Endpoints
├── config.py                   # Centralized Configuration & Environment Variables
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation & Viva Guide
├── .gitignore                  # Git Exclusions
├── .env.example                # Sample Environment File
│
├── model/                      # Saved Model Binaries (Generated upon training)
│   ├── plant_disease_model.keras
│   ├── class_names.json
│   └── model_metadata.json
│
├── training/                   # Machine Learning Training Pipeline
│   ├── train_model.py          # Reproducible Training Script with Early Stopping
│   ├── evaluate_model.py       # Metrics Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
│   ├── split_dataset.py        # Automated Train/Validation/Test Splitter
│   ├── model_architecture.py   # MobileNetV2 + CBAM Assembly
│   ├── attention.py            # Custom CBAM Keras Layer Implementation
│   └── results/                # Evaluation Plots & Text Reports
│
├── dataset/                    # PlantVillage Dataset Folder Structure
│   ├── train/
│   ├── validation/
│   └── test/
│
├── data/
│   └── disease_info.json       # Plant Disease Knowledge Base Encyclopedia Data
│
├── database/
│   ├── database.py             # SQLite Connection & CRUD Operations
│   ├── models.py               # Table DDL Definitions
│   └── plant_disease.db        # SQLite Database (Created Automatically)
│
├── utils/
│   ├── preprocessing.py        # Image Resizing (224x224) & Normalization
│   ├── prediction.py           # Real Model Inference Singleton Engine
│   ├── validation.py           # Pillow File Integrity & Security Validator
│   └── gradcam.py              # Gradient Heatmap Generator
│
├── templates/                  # Modular HTML5 Templates
│   ├── base.html
│   ├── index.html
│   ├── detect.html
│   ├── result.html
│   ├── dashboard.html
│   ├── history.html
│   ├── encyclopedia.html
│   ├── disease.html
│   ├── about.html
│   └── 404.html
│
├── static/                     # CSS Design System & JS Modules
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── main.js
│       ├── detect.js
│       ├── dashboard.js
│       ├── history.js
│       └── encyclopedia.js
│
├── uploads/                    # User Image Uploads & Generated Grad-CAM Heatmaps
├── manifest.json               # Progressive Web App Manifest
├── service-worker.js           # PWA Shell Caching Service Worker
└── tests/
    └── test_app.py             # Pytest Unit Test Suite
```

---

## ⚡ Quick Start & Installation Guide (Windows + VS Code)

### 1. Clone & Setup Virtual Environment

Open PowerShell inside the project directory:

```powershell
# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment
venv\Scripts\activate

# Upgrade pip & Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Application

```powershell
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🏋️ Model Training & Evaluation Pipeline

To train the AI model on your own dataset (e.g. PlantVillage):

### Step 1: Split Dataset (If raw folder provided)
```powershell
python training/split_dataset.py
```

### Step 2: Train Attention CNN Model
```powershell
python training/train_model.py
```
This script will:
1. Apply training data augmentations (flips, rotations, zoom, brightness).
2. Build MobileNetV2 + CBAM architecture.
3. Train with `EarlyStopping` and `ReduceLROnPlateau`.
4. Save the trained model to `model/plant_disease_model.keras` and class mappings to `model/class_names.json`.

### Step 3: Evaluate Model Metrics
```powershell
python training/evaluate_model.py
```
Generates real evaluation outputs saved in `training/results/`:
- `confusion_matrix.png`
- `accuracy.png` & `loss.png`
- `classification_report.txt`
- `metrics.json`

---

## 🧪 Automated Testing

To run the Pytest test suite:

```powershell
pytest tests/
```

Tests verify system health endpoints (`/health`), Pillow file validation, database CRUD, and class name parser logic.

---

## 🎓 B.Tech Viva Defense Q&A

1. **Why CBAM Attention over standard CNNs?**
   *Standard CNNs process spatial features equally across channels. CBAM reweights feature channels (Channel Attention) and spatial regions (Spatial Attention), prioritizing subtle lesion spots over background greenhouse structures.*

2. **What happens if the model is not trained yet?**
   *The Flask web application enters a graceful "Model Missing State". The web UI remains functional for browsing the Encyclopedia and Dashboard, while `/detect` displays a clear prompt explaining that model training is required. No fake or random predictions are generated.*

3. **How does Grad-CAM work?**
   *Grad-CAM computes gradients of the target class logit with respect to the output feature maps of the final convolutional layer. These gradients act as weights to generate a coarse heatmap showing where the model focused.*

---

## 📜 License & Disclaimers

Developed for **B.Tech Major Project** demonstration purposes. Agricultural management guidelines provided in the encyclopedia are for educational reference.
