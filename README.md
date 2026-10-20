
# Tiny Super-Resolution Demo

A beginner deep learning project exploring image reconstruction using PyTorch and Gradio.

## Overview

This project uses a convolutional neural network trained on CIFAR-10 images to explore image super-resolution concepts.

The model learns to reconstruct 32 × 32 images from bicubic-upsampled, lower-resolution inputs.
![Super-resolution demo](images/demo.png)

## Features

- PyTorch convolutional neural network
- Image preprocessing with Torchvision
- Interactive image upload interface built with Gradio
- Training and validation workflow
- Local web app for model inference

## How It Works

1. An image is converted to RGB.
2. The image is resized to simulate a low-resolution input.
3. The trained model predicts a reconstructed image.
4. The app displays the model's output.

## Tech Stack

- Python 3.11
- PyTorch
- Torchvision
- Gradio
- Pillow

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/tiny-super-resolution.git
cd tiny-super-resolution
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL printed in the terminal.

## Dataset and Training

The model was trained as a learning experiment using CIFAR-10, with 32 × 32 target images and bicubic resizing to simulate reduced resolution.

The training workflow included a training/validation split, a convolutional neural network, an AdamW optimizer, and mean squared error loss.

## Limitations

- The model was trained on small CIFAR-10 images.
- The output is limited to 32 × 32 pixels.
- It has not been validated for general-purpose photo enhancement.
- Results may vary depending on the input image.

This is an educational prototype, not a production-ready image restoration system.

## What I Learned

- Preparing image datasets for supervised learning
- Building and training a CNN in PyTorch
- Evaluating a model with validation data
- Saving and loading trained model weights
- Building an interactive app with Gradio
- Managing Python dependencies and environments
- Publishing an ML project on GitHub

## Future Improvements

- Train on larger, higher-resolution images
- Compare reconstruction quality using additional metrics
- Improve visual comparisons between input and output
- Deploy an online interactive demo
- Experiment with more advanced super-resolution architectures

---

**Project status:** Initial working prototype.

Built as part of my hands-on journey into deep learning and applied machine learning.
