# EuroSAT CNN–Transformer

Implementation and evaluation of a hybrid CNN–Transformer model for satellite image classification using the **EuroSAT RGB dataset**.

The project is based on the paper:

> **Hybrid deep learning for satellite image classification: Integrating CNN and transformer architectures**

The provided GitHub implementation was also studied and corrected where necessary, particularly the dataset splitting issue.

---

## Overview

The goal of this project is to classify satellite images into **10 different land-cover classes** using a CNN–Transformer architecture.

The implemented workflow consists of:

- CNN-based local feature extraction
- Conversion of CNN feature maps into image tokens
- Learnable positional embeddings
- Transformer encoder layers for global contextual information
- Global average pooling
- Fully connected classification

---

## Dataset

The project uses the **EuroSAT RGB dataset**.

- Total images: **27,000**
- Number of classes: **10**
- Input size: **64 × 64 × 3**

### Classes

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Dataset Split

A proper stratified-independent dataset split was created:

| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 18,900 |
| Validation | 15% | 4,050 |
| Testing | 15% | 4,050 |

Data augmentation was applied only to the training set.

---

## Important Correction to the Original GitHub Implementation

The original GitHub notebook used the same dataset directory for training, validation, and testing:

```python
train_dataset = datasets.ImageFolder(DATASET_PATH, transform)
val_dataset = datasets.ImageFolder(DATASET_PATH, transform)
test_dataset = datasets.ImageFolder(DATASET_PATH, transform)
