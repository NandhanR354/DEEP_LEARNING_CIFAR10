# Task 2: Deep Learning Project - Image Classification

## Overview
This project implements a deep learning model for image classification using PyTorch. The model is trained on the CIFAR-10 dataset and includes comprehensive visualizations of training results.

## Features
- CNN architecture optimized for CIFAR-10
- Data augmentation and preprocessing
- Training with validation monitoring
- Model checkpointing and early stopping
- Comprehensive visualizations (loss curves, accuracy plots, confusion matrix)
- Model evaluation and performance metrics

## Project Structure
```
Task2_Deep_Learning/
├── data/
│   └── cifar10/  # Auto-downloaded
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── models.py
│   ├── dataset.py
│   ├── utils.py
│   └── config.py
├── models/
│   ├── checkpoints/
│   └── best_model.pth
├── outputs/
│   ├── training_plots.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── metrics.json
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train the model
python src/train.py

# Train with custom configuration
python src/train.py --epochs 50 --batch_size 64 --lr 0.001

# Evaluate existing model
python src/train.py --evaluate --model_path models/best_model.pth
```

## Model Architecture
- Convolutional Neural Network with residual connections
- Batch normalization and dropout for regularization
- Adaptive average pooling for flexible input sizes
- 10 output classes for CIFAR-10 classification

## Results
- Training and validation accuracy/loss curves
- Confusion matrix for detailed performance analysis
- Per-class classification metrics
- Sample predictions with confidence scores

## Author
CODTECH Internship - Data Science Track
