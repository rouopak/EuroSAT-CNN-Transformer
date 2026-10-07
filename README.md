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
```
Input Image
64 × 64 × 3
      │
      ▼
CNN Feature Extraction
      │
      ▼
32 × 16 × 16 Feature Map
      │
      ▼
Tokenization
256 × 32 Tokens
      │
      ▼
Learnable Positional Embedding
      │
      ▼
4 Transformer Encoder Layers
8 Attention Heads
      │
      ▼
Global Average Pooling
      │
      ▼
Fully Connected Layer
      │
      ▼
10 Classes
| Parameter | Value |
|---|---:|
| Encoder layers | 4 |
| Attention heads | 8 |
| Embedding dimension | 32 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |

Difference Between the Paper and GitHub Implementation

The accompanying GitHub implementation does not completely match the architecture and training configuration described in the paper.
For example, the GitHub implementation uses:
- One Transformer block
- Four attention heads
- Different training hyperparameters
while the paper specifies:
- Four Transformer layers
- Eight attention heads
- Learning rate of 0.0001
- 100 epochs
The paper also discusses feature fusion and an attention mechanism, but does not provide enough architectural details to reproduce those components exactly.
Therefore, this project does not claim that an independently designed fusion/attention implementation is an exact reproduction of the paper.
The implemented model follows the architecture that is explicitly specified.

Results
Validation Performance
Best validation accuracy: 94.42%
Best epoch: 99

Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **94.37%** |
| Test Loss | **0.1813** |
| Macro Precision | **94.23%** |
| Macro Recall | **94.18%** |
| Macro F1 | **94.16%** |
| Weighted F1 | **94.34%** |
| Test Samples | **4,050** |

Visualizations

The notebook contains the following visualizations:
Training Curves
- Training accuracy vs. validation accuracy
- Training loss vs. validation loss
Classification Analysis
- Confusion matrix
- Per-class performance
Explainability
- Grad-CAM visualization
- Transformer token-importance visualization
Note: The Transformer visualization is a gradient-based token-importance visualization. It is not presented as a direct visualization of the Transformer's internal attention weights.

Comparison With the Paper

The paper reports a final accuracy of:
98.36%
The implementation in this repository achieved:
94.37% test accuracy
The results are therefore not claimed to be an exact reproduction of the reported 98.36%.
There are several differences and ambiguities between the paper and the provided implementation, including:
- Dataset-count discrepancy
- Different references to input resolution
- Different Transformer configurations
- Incomplete architectural details for feature fusion and attention
- Incorrect dataset splitting in the original GitHub implementation
The 94.37% result represents the explicitly specified CNN + Transformer architecture evaluated using a corrected 70/15/15 dataset split.

References
Paper
Hybrid deep learning for satellite image classification: Integrating CNN and transformer architectures
Dataset
EuroSAT RGB Dataset
Original Implementation
The GitHub implementation provided alongside the paper was used as the starting point for reproducing and auditing the workflow.
