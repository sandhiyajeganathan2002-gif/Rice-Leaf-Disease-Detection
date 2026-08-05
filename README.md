<div align="center">

# 🌾 Rice Leaf Disease Detection

**Deep learning image classifier that identifies rice leaf diseases from photographs, using transfer learning with EfficientNetB0.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Results](#results)
- [Installation / How to Run](#installation--how-to-run)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

Rice is a staple food for billions of people, and crop diseases can significantly reduce yield if not caught early. This project builds an image classifier that looks at a photo of a rice leaf and predicts which of three common diseases is present — automating a task that normally requires a trained agronomist's visual inspection.

The final model is an **EfficientNetB0** backbone fine-tuned via transfer learning, benchmarked against a **MobileNetV2** baseline and validated with **5-fold stratified cross-validation** to get an honest read on generalization from a small (119-image) dataset.

## Problem Statement

Traditionally, rice leaf disease diagnosis depends on farmers or agronomists visually inspecting plants — a process that is slow, requires expertise that isn't always locally available, and can delay treatment long enough for yield loss to occur. This project frames diagnosis as a supervised image classification problem: given a leaf photo, predict the disease class, enabling faster, more accessible, and more scalable early intervention.

## Dataset

119 labeled rice leaf images across three disease classes (roughly balanced, ~40 images each):

| Class | Cause | Visual Symptom | Images |
|---|---|---|---|
| **Bacterial Leaf Blight** | Bacterial infection | Water-soaked lesions along leaf margins | ~40 |
| **Brown Spot** | Fungal disease | Small, brown, oval-shaped lesions | ~40 |
| **Leaf Smut** | Fungal disease | Small black spots scattered across the leaf surface | ~39 |

> **Note:** The dataset is not included in this repository to keep it lightweight.

To reproduce this project, download the dataset from:
[PRCP-1001-RiceLeaf.zip](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1001-RiceLeaf.zip)

The notebook itself handles unzipping and flattening nested folders if you start from the raw archive — see the `PRCP-1001-RiceLeaf.zip` handling in the first few cells. Once processed, the data is organized as:
data/
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Model Architecture:** EfficientNetB0 (ImageNet-pretrained, fine-tuned) — with MobileNetV2 as a comparison baseline
- **Supporting Libraries:** scikit-learn, NumPy, Pandas, Matplotlib, Pillow
- **Environment:** Jupyter Notebook

## Project Structure
Rice-Leaf-Disease-Detection/
├── rice_leaf_disease_detection.ipynb 
├── data/ 
├── README.md
└── requirements.txt
## Approach

1. **Exploratory Data Analysis** — checked class distribution, image sizes/modes, and viewed sample images per class given the small dataset size.
2. **Preprocessing & Augmentation** — resized images to 224×224, applied random flips, rotation, zoom, and contrast jitter to compensate for the limited number of training images, and used each backbone's matching `preprocess_input` function to avoid silent accuracy loss.
3. **Model Development**
   - **Baseline attempt:** a small CNN trained from scratch, which collapsed to predicting a single class — 119 images simply isn't enough to learn useful features from scratch.
   - **MobileNetV2 (transfer learning baseline):** two-stage training — frozen ImageNet backbone first, then fine-tuning the top ~30 layers at a low learning rate.
   - **EfficientNetB0 (final model):** same two-stage strategy, which outperformed MobileNetV2 on this dataset and was selected as the final model.
4. **Handling Class Imbalance** — computed balanced class weights (`sklearn.utils.class_weight`) and applied them during training, since the classes are close to but not perfectly balanced.
5. **Regularization & Training Controls** — dropout layers, `EarlyStopping` on validation accuracy, and `ReduceLROnPlateau` on validation loss.
6. **Evaluation** — classification reports and confusion matrices on a held-out validation split, plus **5-fold stratified cross-validation** to get a more reliable estimate of generalization than a single train/validation split allows.

### Techniques Used

1. **Data augmentation** (flip, rotate, zoom, contrast) to expand a tiny dataset
2. **Transfer learning** (MobileNetV2, EfficientNetB0) to avoid training from scratch on too little data
3. **Two-stage fine-tuning** — frozen backbone first, then unfreeze the top layers at a low learning rate
4. **Model-specific `preprocess_input`** to avoid silent accuracy loss from mismatched preprocessing
5. **Dropout** to reduce overfitting
6. **EarlyStopping + ReduceLROnPlateau** to stop training wisely and adapt the learning rate
7. **Class weighting** to handle mild class imbalance
8. **5-fold stratified cross-validation** for a more honest generalization estimate

### Challenges & Solutions

| Challenge | Solution |
|---|---|
| Small dataset (119 images) | Used transfer learning instead of training from scratch |
| Model collapsed to one class | Tuned `EarlyStopping` patience and learning rate schedule |
| Unstable BatchNorm statistics on tiny batches | Reduced batch size |
| Incorrect single-image predictions | Fixed a double-preprocessing bug in the inference path |

## Results

### Final model: EfficientNetB0

| Metric | Value |
|---|---|
| Validation Accuracy (single 80/20 split) | **91.3%** |
| 5-Fold Stratified CV Mean Accuracy | **~80.6%** |
| Validation Loss | 0.20 |

The gap between the single-split and cross-validated accuracy reflects the overfitting risk that comes with only 119 images — the cross-validation figure is the more trustworthy estimate of real-world performance, and fold-level accuracy varied noticeably (from ~67% to ~92%), underscoring that more data would meaningfully tighten this estimate.

### Model comparison (single 80/20 split)

| Model | Validation Accuracy | Notes |
|---|---|---|
| MobileNetV2 | 83% | Lightweight, fast inference, stable training |
| **EfficientNetB0** | **91.3%** | Higher accuracy and stronger feature extraction on this dataset — **selected as final model** |

**Recommendation:** EfficientNetB0, since the accuracy gain outweighs MobileNetV2's speed advantage for this use case (offline/batch diagnosis rather than real-time inference on constrained hardware).

## Installation / How to Run

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/sandhiyajeganathan2002-gif/Rice-Leaf-Disease-Detection.git
cd Rice-Leaf-Disease-Detection
pip install -r requirements.txt
```

### Dataset

Download the dataset (see [Dataset](#dataset) above) and place it in the project root before running the notebook.

### Run

```bash
jupyter notebook rice_leaf_disease_detection.ipynb
```

Run the cells in order. The notebook will:
1. Unzip and organize the dataset.
2. Run EDA and show sample images per class.
3. Train the MobileNetV2 baseline and the EfficientNetB0 final model.
4. Run 5-fold cross-validation.
5. Print classification reports and confusion matrices.
6. Predict the disease class for a single test image (`rice_image.jfif`) as a usage example.

To predict on your own image, swap in your file path where the notebook loads `rice_image.jfif` before calling `.predict()`.

## Future Improvements

- Expand the dataset with more images per class to shrink the gap between single-split and cross-validated accuracy
- Deploy the model as an interactive web app (Streamlit or FastAPI) so users can upload a leaf photo and get an instant prediction
- Benchmark additional architectures (ResNet, Vision Transformers)
- Add Grad-CAM visualizations to show which regions of the leaf drive each prediction, improving interpretability and trust for end users like farmers and agronomists

## 👤 Author

**Sandhiya Jeganathan**

Open to **Data Science Intern** / **Junior Data Scientist** / **AI-ML Engineer** roles

📍 Bengaluru, India

- 🔗 LinkedIn: https://www.linkedin.com/in/sandhiyajegan
- 📧 Email: sandhiyajeganathan2002@gmail.com
- 💻 GitHub: https://github.com/sandhiyajeganathan2002-gif

---

<div align="center">
If you found this project useful, consider giving it a ⭐
</div>
