# Arabic Sign Language Recognition

A convolutional neural network for recognizing Arabic Sign Language (ArSL) hand gestures across all 28 letter classes, built from scratch using PyTorch.

## Overview

This project develops a deep learning pipeline to classify hand gesture images into their corresponding Arabic letters. The model achieves **96.74% test accuracy** on a 28-class dataset without any pre-trained weights or transfer learning.

## Dataset

The dataset contains approximately 5,500 images across 28 Arabic letter classes (~196 images per class) captured under controlled conditions. Split: 70% training, 15% validation, 15% test.

## Model Architecture

The model consists of 5 convolutional blocks, each containing two convolutional layers with Batch Normalization and a residual shortcut connection, followed by Max Pooling. Features are aggregated with Global Average Pooling before a two-layer fully connected classifier with Dropout.

| Component | Details |
|-----------|---------|
| Input | 128×128 RGB |
| Conv Blocks | 5 blocks (64, 128, 256, 512, 512 channels) |
| Residual Shortcuts | 1×1 conv to match channel dimensions |
| Global Average Pooling | Replaces Flatten + large FC layer |
| Classifier | FC(1024) + Dropout(0.5) + FC(512) + Dropout(0.3) + FC(28) |
| Parameters | ~2.1M |

## Training

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 with 5-epoch warmup + cosine annealing |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 60 |
| Label Smoothing | 0.1 |
| Mixed Precision | AMP (fp16) |

Augmentation includes random crop, horizontal flip, rotation (±15°), color jitter, random grayscale, affine translation, and random erasing.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.74% |
| Validation Accuracy | 96.38% |
| Mean Per-Class Accuracy | 96.8% |
| Classes at 100% | 13 / 28 |

The train-validation gap stays under 1% across all 60 epochs with no overfitting.

## Files

- `train.py` — full training pipeline: data loading, model definition, training loop, evaluation, and plot generation
- `demo.py` — Gradio web application for inference via image upload or webcam

## Usage

### Training

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

Organize your dataset as:

```
data/
├── ALIF/
├── BAA/
├── TA/
└── ...
```

Set `data_dir` in the `CONFIG` dictionary, then run:

```bash
python train.py
```

Outputs are saved to `./outputs/` including the best model checkpoint, training curves, confusion matrix, and per-class accuracy plot.

### Demo

```bash
pip install gradio
python demo.py
```

Open the generated Gradio URL in any browser. Supports image upload and webcam capture.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- gradio (demo only)
