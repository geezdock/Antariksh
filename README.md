# Generative AI-Based Cloud Removal and Reconstruction for Satellite Imagery

AI-powered system that removes clouds from satellite imagery using a U-Net deep learning model.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic training data
python generate_synthetic_data.py

# 2. Train the model
python -m models.train

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Project Structure

```
cloud-removal-isro/
├── data/                   # Dataset (cloudy / clean pairs)
├── models/
│   ├── unet.py             # U-Net architecture
│   ├── train.py            # Training loop
│   └── predict.py          # Inference pipeline
├── evaluation/
│   ├── psnr.py             # PSNR metric
│   └── ssim.py             # SSIM metric
├── app/
│   └── streamlit_app.py    # Web dashboard
├── results/                # Training curves
├── requirements.txt
└── generate_synthetic_data.py
```

## Tech Stack

Python, PyTorch, OpenCV, Streamlit, scikit-image
