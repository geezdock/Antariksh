# Product Requirements Document

## Generative AI-Based Cloud Removal and Reconstruction for Satellite Imagery

---

## 1. Objective

Build a proof-of-concept system that takes a cloud-covered satellite image as input and generates a reconstructed cloud-free image using a deep learning U-Net. The system demonstrates the feasibility of using AI for restoring obscured Earth observation imagery.

**Target:** Academic/innovation competition  
**Timeline:** 1 week  
**Team:** 3 students  

---

## 2. Problem Statement

Satellite imagery is frequently affected by cloud cover, hiding critical surface information needed for agriculture, disaster management, environmental monitoring, and urban planning. This system uses AI-powered image reconstruction to restore cloud-obscured regions and generate usable cloud-free imagery.

---

## 3. Scope

### In Scope

- Satellite image preprocessing
- Dataset loading and augmentation
- Cloudy image input (upload or synthetic)
- Cloud-free image reconstruction via U-Net
- PSNR and SSIM evaluation
- Streamlit web interface
- Result visualization (side-by-side comparison)
- Model inference pipeline
- Checkpoint saving and loading
- Training loss tracking and curves

### Out of Scope

- Diffusion models
- Multi-temporal fusion
- Real-time satellite feeds
- LISS-IV specific optimization
- Production deployment
- User authentication
- Distributed training
- Video or multi-frame input

---

## 4. User Flow

```
1. User uploads a cloudy satellite image
2. System preprocesses the image (resize to 256×256, normalize)
3. Trained U-Net model performs inference
4. System displays:
   ┌─────────────────────┬─────────────────────┐
   │   Input (Cloudy)    │  Reconstructed      │
   │                     │  (Cloud-Free)       │
   ├─────────────────────┴─────────────────────┤
   │   PSNR: 31.2 dB    │   SSIM: 0.89         │
   ├───────────────────────────────────────────┤
   │          [Download Reconstructed]         │
   └───────────────────────────────────────────┘
5. User can download the reconstructed output
```

---

## 5. Functional Requirements

### 5.1 Data Generation Module

**File:** `generate_synthetic_data.py`

Responsibilities:

- Generate realistic synthetic cloudy/clean image pairs for training
- Use procedural terrain generation (fractal noise with water, sand, plains, forest, rock, snow biomes + rivers)
- Apply sensor noise (Gaussian) and JPEG compression artifacts to simulate satellite-like textures
- Generate diverse cloud types:
  - **Cumulus** — bright white, opaque (intensity 235–255)
  - **Cirrus** — thin, semi-transparent (intensity 200–230)
  - **Stratus** — uniform gray-white (intensity 210–240)
  - **Mixed** — random intensity range
- Each cloud generated via multi-octave fractal noise with wind-blown displacement mapping
- Cloud opacity varies per sample (0.6–1.0)
- Output: 256×256 PNG pairs organized as:
  ```
  data/
  ├── train/
  │   ├── cloudy/   (1000 images)
  │   └── clean/    (1000 images)
  └── val/
      ├── cloudy/   (200 images)
      └── clean/    (200 images)
  ```

### 5.2 Model Module

**File:** `models/unet.py`

Architecture: U-Net

| Component | Specification |
|---|---|
| Input channels | 3 (RGB) |
| Output channels | 3 (RGB) |
| Base features | 32 |
| Depth | 4 down/up blocks |
| Skip connections | Concatenation |
| Upsampling | Bilinear (no learned transpose convs) |
| Output activation | Sigmoid (values in [0, 1]) |
| Total parameters | ~2 million |
| Normalization | BatchNorm after each conv |

Requirements:

- Encoder-decoder with skip connections
- RGB (3-channel) in/out
- Sigmoid output bounded to [0, 1]
- Save/load checkpoints via `torch.save` / `torch.load`

### 5.3 Training Module

**File:** `models/train.py`

Dataset:

- `CloudDataset` loads paired cloudy/clean images from `data/train/` and `data/val/`
- Resizes to 256×256 (skips if already correct size)
- Normalizes to [0, 1] via division by 255.0
- Per-pixel augmentation (training only):
  - Random brightness (0.85–1.15×)
  - Random contrast (0.85–1.15×)
  - Additive Gaussian noise (50% probability, σ = 0.005–0.02)

Training Loop:

| Hyperparameter | Value |
|---|---|
| Loss function | L1 Loss |
| Optimizer | Adam |
| Learning rate | 1×10⁻⁴ |
| Batch size | 16 |
| Epochs | 30 (default) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Device | CPU (fallback) or CUDA |

Outputs:

- `models/checkpoints/unet_final.pth` — final model weights
- `models/checkpoints/unet_epoch_{N}.pth` — periodic checkpoints (every 10 epochs)
- `results/training_curves/loss_curves.png` — train/val loss plot

### 5.4 Inference Module

**File:** `models/predict.py`

Function: `predict(image_path, checkpoint_path)`

1. Load image via OpenCV
2. Resize to 256×256
3. Normalize to [0, 1]
4. Run model forward pass (no grad)
5. Clamp output to [0, 1], scale to [0, 255], convert to uint8
6. Resize back to original input dimensions
7. Return BGR image (OpenCV format)

### 5.5 Evaluation Module

**Files:** `evaluation/psnr.py`, `evaluation/ssim.py`

- `compute_psnr(original, reconstructed, data_range=255)` — wraps `skimage.metrics.peak_signal_noise_ratio`
- `compute_ssim(original, reconstructed, data_range=255, channel_axis=-1)` — wraps `skimage.metrics.structural_similarity`
- Accepts uint8 or float images in [0, 255] range

### 5.6 Streamlit Dashboard

**File:** `app/streamlit_app.py`

Sections:

| Section | Content |
|---|---|
| **Header** | Title + description |
| **Upload** | File uploader (PNG, JPG, JPEG, TIF) |
| **Prediction** | Two-column layout: input (cloudy) vs. reconstructed (cloud-free) |
| **Metrics** | PSNR (dB) and SSIM score cards |
| **Download** | Download button for reconstructed image |

Behavior:

- Gracefully handles missing checkpoint (shows info message)
- Temporarily saves uploaded file, runs inference, then deletes temp file
- Resizes images for SSIM computation to ensure valid window size

---

## 6. Non-Functional Requirements

- **Python** 3.10–3.13 (note: Python 3.14 is not fully compatible with all dependencies)
- **GPU** optional (runs fully on CPU)
- Works on Google Colab (T4 GPU) and local Windows/Linux
- Modular codebase (models/, evaluation/, app/ separation)
- Reproducible training (fixed seed recommended)
- Total training time on CPU: ~15–30 minutes (with features=32, batch_size=16)

---

## 7. Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| PyTorch | ≥2.0 | Deep learning framework |
| OpenCV | ≥4.8 | Image I/O and processing |
| NumPy | ≥1.24 | Numerical operations |
| Matplotlib | ≥3.7 | Loss curve plotting |
| scikit-image | ≥0.21 | PSNR / SSIM metrics |
| Streamlit | ≥1.28 | Web dashboard |
| pandas | ≥2.0 | (optional) result logging |

---

## 8. Project Structure

```
cloud-removal/
├── data/
│   ├── train/
│   │   ├── cloudy/        # Input: cloudy satellite images
│   │   └── clean/         # Target: cloud-free ground truth
│   └── val/
│       ├── cloudy/
│       └── clean/
├── models/
│   ├── __init__.py
│   ├── unet.py            # U-Net architecture
│   ├── train.py           # Training loop + dataset
│   └── predict.py         # Inference pipeline
├── evaluation/
│   ├── __init__.py
│   ├── psnr.py            # PSNR metric
│   └── ssim.py            # SSIM metric
├── app/
│   └── streamlit_app.py   # Web dashboard
├── notebooks/             # (optional) Jupyter notebooks
├── results/
│   └── training_curves/   # Loss plots
├── reports/               # Generated reports and assets
├── generate_synthetic_data.py   # Synthetic data generator
├── requirements.txt
├── .gitignore
├── README.md
└── PRD.md
```

---

## 9. Setup & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data (1200 pairs)
python generate_synthetic_data.py

# 3. Train the model (~15-30 min on CPU)
python -m models.train

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 10. Deliverables

1. Trained U-Net model checkpoint (`unet_final.pth`)
2. Evaluation metrics (PSNR ≥ 25 dB, SSIM ≥ 0.7 on synthetic test set)
3. Streamlit dashboard showing end-to-end workflow
4. Sample reconstructed outputs (saved to `results/`)
5. GitHub repository with full commit history
6. This PRD and a README for documentation

---

## 11. Success Criteria

- [ ] Model trains successfully without errors
- [ ] Reconstruction output is visually plausible (clouds removed, terrain preserved)
- [ ] PSNR and SSIM metrics are displayed in the dashboard
- [ ] Dashboard accepts user uploads and runs inference
- [ ] Team can present the complete working pipeline live

---

## 12. Known Issues & Mitigations

| Issue | Mitigation |
|---|---|
| Synthetic data ≠ real satellite imagery | Train on real paired data (Sentinel-2, Landsat) for production use |
| CPU training is slow | Reduce `features` (default: 32) or `batch_size`; use GPU if available |
| Python 3.14 incompatibility | Use Python 3.10–3.13 for best package compatibility |
| Noisy/gray reconstructions from untrained model | Train for at least 30 epochs with L1 loss |
| OpenCV import fails on Streamlit Cloud | Use `opencv-python-headless` in requirements |

---

## 13. Future Enhancements (Post-Competition)

- Pix2Pix GAN for sharper outputs
- Attention U-Net / U-Net++ for better detail preservation
- Diffusion models for state-of-the-art reconstruction
- LISS-IV / Sentinel-2 specific adaptation
- Multi-temporal cloud removal (using multiple time-series frames)
- Real-time inference API
