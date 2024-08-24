
---

# U-Net: Convolutional Networks for Biomedical Image Segmentation

This repository contains an implementation of the U-Net architecture for biomedical image segmentation, inspired by the seminal paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

I’ve written a detailed article on Medium that explains the UNet model in-depth, including its implementation in Python. You can read it here:

[Understanding and Implementing the UNet Model]([https://medium.com/@pramodyasahan.edu/understanding-and-implementing-the-unet-model-for-biomedical-image-segmentation-abedfd3be3d7]) on Medium.


## Overview

The U-Net architecture is a convolutional neural network designed for precise segmentation of biomedical images. It consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. This implementation is based on the original paper and is adapted for the Carvana Image Masking Challenge dataset.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Features

- **U-Net Architecture**: Implementation of the U-Net model with customizable depth and width.
- **Carvana Dataset Integration**: Preprocessing and loading of the Carvana Image Masking Challenge dataset.
- **Training & Validation Loop**: Customizable training loop with real-time loss monitoring.
- **Model Saving**: Save the trained model to a specified path for later use.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/pramodyasahan/unet-biomedical-segmentation.git
    cd unet-biomedical-segmentation
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```


## Usage

### 1. Dataset Preparation

- Download the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) dataset.
- Place the dataset in the `data/` directory, with the following structure:
    ```
    data/
    ├── train/
    ├── train_mask/
    ├── manual_test/
    └── manual_test_mask/
    ```

### 2. Training the Model

Run the training script:
```bash
python main.py
```

- **Adjust Parameters**: Modify the `LEARNING_RATE`, `BATCH_SIZE`, and `EPOCHS` directly in the `train.py` script.

### 3. Model Inference

To use the trained model for inference:
```python
import torch
from unet import UNet
from carvana_dataset import CarvanaDataset

model = UNet(in_channels=3, n_classes=1)
model.load_state_dict(torch.load('unet.pth'))
model.eval()

# Load and preprocess your image, then pass it through the model
```

## Dataset

The model was trained and evaluated on the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) dataset, which consists of high-resolution images of cars and their corresponding masks.

### Preprocessing

- Images are resized to `256x256` pixels.
- The dataset is split into training (80%) and validation (20%) sets.

## Training

- **Optimizer**: `AdamW` with a learning rate of `3e-4`.
- **Loss Function**: `Binary Cross-Entropy with Logits` (`BCEWithLogitsLoss`).
- **Batch Size**: `8` (modifiable)
- **Epochs**: `2` (modifiable)

## Results

Results can be monitored during training with loss values printed for both training and validation datasets. Final model weights are saved as `unet.pth`.

## References

- [Original U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)

## Acknowledgments

This implementation is inspired by the U-Net model described by Ronneberger et al. (2015). Special thanks to the authors and the open-source community for making such implementations accessible.

---

### Additional Notes:

- Customize the above sections based on your specific implementation details.
- If you have more sections (e.g., about performance metrics, or additional datasets), feel free to add them to the README.
