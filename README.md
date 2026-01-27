# Facial-Emotion-Recognition-using-Multi-Modal-Deep-Learning

# 🎭 Facial Emotion Recognition using Multi-Modal Deep Learning

> **2023 Undergraduate Graduation Project**  
> A multi-modal emotion recognition system combining facial images, landmarks, and depth features (Z-Values)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project implements a deep learning-based facial emotion recognition system that classifies **7 emotions** from facial expression images using a multi-modal approach.

### Key Features

- **Multi-Modal Learning**: Combines image features, facial landmark coordinates, and Z-value depth information
- **MediaPipe Integration**: Automated face detection and 468 landmark extraction
- **Multiple Model Architectures**: Custom CNN with Residual Blocks, VGG16/19, ResNet50, MobileNet
- **Class Imbalance Handling**: Focal Loss and RandomOverSampler
- **Real-time Demo**: Browser-based webcam emotion recognition

---

## 🎯 Emotion Classes

| Label | Emotion |
|:-----:|:--------|
| 0 | Happy |
| 1 | Sad |
| 2 | Anger |
| 3 | Surprise |
| 4 | Disgust |
| 5 | Neutral |
| 6 | Fear |

---

## 📊 Dataset

- **Training samples**: ~71,002
- **Format**: TFRecord for efficient I/O
- **Image size**: 64×64 grayscale

### TFRecord Structure
```python
features = {
    "image": tf.io.FixedLenFeature([], tf.string),      # JPEG encoded image
    "landmarks": tf.io.VarLenFeature(tf.float32),       # 468 × 3 landmarks
    "label": tf.io.FixedLenFeature([], tf.int64),       # Emotion label (0-6)
    "z_values": tf.io.FixedLenFeature([8], tf.float32), # Depth features
}
```

---

## 🔧 Preprocessing Pipeline

### 1. Face Detection & Cropping
- MediaPipe Face Detection with confidence threshold 0.2
- Face region + 10% margin → resize to 64×64

### 2. Landmark Extraction
Extracts **101 key landmarks** from 9 facial regions using MediaPipe FaceMesh:
- Eyebrows (left/right)
- Eyes (left/right)
- Lips (inner/outer)
- Face contour
- Nose & nose bridge

### 3. Z-Value Feature Engineering
```python
# Combines 8 depth values into 56 features using combinations
z_combinations = list(itertools.combinations(range(8), 3))  # C(8,3) = 56
```

### 4. Data Augmentation
```python
image = tf.image.random_flip_left_right(image)
image = tf.image.random_brightness(image, max_delta=0.1)
image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
```

---

## 🏗️ Model Architecture

### Main Model: Multi-Modal Fusion (Image + Landmark + Z-Value)

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
├─────────────────┬───────────────────┬───────────────────────┤
│  Image Input    │  Landmark Input   │    Z-Value Input      │
│   (64,64,1)     │      (303,)       │        (56,)          │
└────────┬────────┴─────────┬─────────┴───────────┬───────────┘
         │                  │                     │
         ▼                  ▼                     ▼
   ┌───────────┐      ┌───────────┐        ┌───────────┐
   │   IMAGE   │      │ LANDMARK  │        │  Z-VALUE  │
   │  SUBNET   │      │  SUBNET   │        │  SUBNET   │
   ├───────────┤      ├───────────┤        ├───────────┤
   │ Conv(64)  │      │ Dense(101)│        │ Dense(101)│
   │ Conv(128) │      │ Dense(64) │        │ BN → ELU  │
   │ ResBlock  │      │ Dense(32) │        │ Dropout   │
   │   ×3      │      │ +BN,ELU   │        └─────┬─────┘
   │ Conv(512) │      │ +Dropout  │              │
   │ Flatten   │      └─────┬─────┘              │
   └─────┬─────┘            │                    │
         │                  │                    │
         └────────┬─────────┘                    │
                  ▼                              │
           ┌────────────┐                        │
           │ Concatenate│ ◄──────────────────────┘
           └──────┬─────┘
                  ▼
           ┌────────────┐
           │ Dense(128) │
           │ BN → ELU   │
           │ Dropout    │
           └──────┬─────┘
                  ▼
           ┌────────────┐
           │ Dense(7)   │
           │ Softmax    │
           └────────────┘
```

### Image Subnet Details
- **Conv2D layers**: 64 → 128 → 256 (×3 Residual Blocks) → 512
- **Regularization**: BatchNorm, Dropout (0.2-0.5), L2
- **Activation**: ReLU

### Residual Block
```python
for _ in range(3):
    shortcut = x
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(256, (1,1))(shortcut)  # Projection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
```

---

## 🧪 Experiments

### Model Variants Tested

| Category | Model | Input |
|----------|-------|-------|
| Single Input | Image Only | (64,64,1) |
| Single Input | Landmark Only | (303,) |
| Single Input | Z-Value Only | (56,) |
| Multi-Modal | Image + Landmark | 2 branches |
| Multi-Modal | Image + Landmark + Z-Value | 3 branches |
| Transfer Learning | VGG16, VGG19 | ImageNet pretrained |
| Transfer Learning | ResNet50, MobileNet | ImageNet pretrained |

---

## ⚙️ Training Configuration

```python
# Reproducibility
seed = 42

# Training
epochs = 100-200
batch_size = 32
learning_rate = 5e-4

# Callbacks
EarlyStopping(patience=10, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

# Loss Function: Focal Loss (for class imbalance)
focal_loss(gamma=2.0, alpha=0.25)

# Regularization
L2(0.001) for hidden layers
Dropout: 0.2-0.6
```

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall and per-emotion
- **Precision / Recall / F1 Score** (Macro average)
- **ROC-AUC** (One-vs-Rest)
- **Confusion Matrix**
- **Precision-Recall Curve**

---

## 🚀 Getting Started

### Installation
```bash
pip install tensorflow>=2.10.0 mediapipe opencv-python
pip install imbalanced-learn scikit-learn
pip install matplotlib seaborn pandas
```

### Running on Google Colab (Recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
!pip install mediapipe imblearn
```

### Training
```python
# Run preprocessing
python preprocessing.py

# Run model training
python train.py
```

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing.py      # Face detection, landmark extraction
│   └── train.py              # Model training and evaluation
├── data/
│   └── *.tfrecord            # Preprocessed data
├── models/
│   └── *.keras               # Saved models
└── results/
    ├── *_confusion_matrix.png
    ├── *_roc_curve.png
    └── *_data.txt            # Performance summary
```

---

## 📋 Output Files

Each model generates:
| File | Description |
|------|-------------|
| `{model}_accuracy_loss.png` | Training curves |
| `{model}_confusion_matrix.png` | 7×7 confusion matrix |
| `{model}_roc_curve.png` | ROC curves with AUC |
| `{model}_data.txt` | Performance metrics summary |
| `{model}_model.keras` | Saved model |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | TensorFlow, Keras |
| Computer Vision | MediaPipe, OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Development | Google Colab, GPU |

---

## 📝 Notes

- Developed and tested on Google Colab with GPU runtime
- MediaPipe may fail on low-quality or occluded faces
- Transfer learning models convert grayscale to RGB internally

---

## 👥 Contributors

- 2023 Undergraduate Graduation Project

---

## 📄 License

MIT License

---

## 📚 References

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)

---

<p align="center">
  <i>🎓 2023 Graduation Project</i>
</p>
