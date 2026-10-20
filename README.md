---
title: Tiny Super-Resolution Demo
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Tiny Super-Resolution Demo

A beginner portfolio project built while learning practical deep learning with PyTorch.

## What it does

The app reduces an uploaded image to 16 × 16 pixels, enlarges it to 32 × 32 using bicubic interpolation, and passes it through a small convolutional neural network. The model returns a 32 × 32 output.

## Training setup

- Dataset: CIFAR-10
- Target: original 32 × 32 images
- Input construction: 32 × 32 → 16 × 16 → bicubic resize back to 32 × 32
- Model: small residual CNN implemented in PyTorch
- Loss: mean squared error (MSE)
- Optimizer: AdamW

## Limitations

This is an educational baseline, not a production image-restoration model. It was trained on small CIFAR-10 images and always outputs 32 × 32 pixels. It is not expected to improve arbitrary high-resolution photos reliably.

## Run locally

1. Put `app.py`, `requirements.txt`, and your `sr_model.pth` checkpoint in the same folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch: `python app.py`

## Files

- `app.py`: Gradio UI and inference code
- `requirements.txt`: Python dependencies
- `sr_model.pth`: trained weights, copied from the training notebook output

## Next improvements

- Compare bicubic input and model output side by side.
- Report validation loss and an image-quality metric such as PSNR.
- Train on larger image patches before claiming general-purpose photo enhancement.
