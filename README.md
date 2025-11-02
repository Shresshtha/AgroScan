# 🌾 AgroScan — Crop Disease Classification using Deep CNNs

> A deep learning project aimed at early detection of crop leaf diseases using a custom Convolutional Neural Network (CNN) trained on the PlantVillage dataset.  
> The goal: assist precision agriculture and reduce crop yield losses due to late or incorrect disease diagnosis.

---

## 🎯 Objective

Develop an image-based classification model that can accurately identify crop diseases from leaf images across multiple crops and disease classes — enabling faster, field-ready diagnostic tools for farmers and agronomists.

---

## 📊 Dataset — PlantVillage

**Source:** [Kaggle – PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
**Records:** ~54,000 labeled leaf images  
**Classes:** 38 diseases across 14 major crops (tomato, apple, maize, potato, etc.)  

| Attribute | Description |
|------------|-------------|
| Image | RGB leaf image |
| Label | Disease type (e.g., *Tomato Early Blight*) |
| Format | `.jpg` |
| Resolution | 256×256 px |
| Split | 70% Train / 20% Validation / 10% Test |

---

## 🧠 Model Architecture — Balanced Deep CNN

A custom CNN trained **from scratch** (no transfer learning) with L2 regularization and dropout to handle high intra-class variability across 38 disease types.



Input (224x224x3)
│
├── [Conv2D(32) + Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)]
├── [Conv2D(64) + Conv2D(64) + BatchNorm + MaxPool + Dropout(0.3)]
├── [Conv2D(128) + Conv2D(128) + BatchNorm + MaxPool + Dropout(0.35)]
├── [Conv2D(256) + Conv2D(256) + BatchNorm + MaxPool + Dropout(0.4)]
├── GlobalAveragePooling2D
├── Dense(512, ReLU) + Dropout(0.5)
└── Dense(38, Softmax)

**Optimizer:** Adam (lr = 1e-4)  
**Loss Function:** Categorical Crossentropy  
**Regularization:** L2 weight decay (0.001)  
**Epochs:** 40  
**Batch Size:** 32  

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|--------|
| 🧠 Deep Learning | TensorFlow, Keras |
| 🐍 Language | Python |
| 📊 Data Analysis | NumPy, Pandas |
| 🎨 Visualization | Matplotlib, Seaborn |
| 🖼️ Image Handling | OpenCV |

---

## 🧮 Training Setup

- **Augmentation:** Rotation, brightness shift, zoom, horizontal flips  
- **Regularization:** Dropout + L2 weight decay  
- **Normalization:** Pixel values scaled to [0, 1]  
- **Callbacks:** EarlyStopping + ModelCheckpoint  

---

## 📈 Results

| Metric | Score |
|:-------:|:------:|
| 🏋️ Training Accuracy | 98.7% |
| 🔍 Validation Accuracy | 96.8% |
| 🎯 F1-Score | 0.95 |
| 🧮 Parameters | ~7.5M |

---


---

## 💡 Key Insights

- Data imbalance handled with augmentation and dropout prevented overfitting.  
- Model successfully generalized across crop species.  
- Grad-CAM verified correct focus on diseased leaf regions (not background).  
- Architecture remains lightweight enough for deployment on edge/mobile devices.

---

