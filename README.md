# Adaptive Notch Filtering for Periodic Noise Removal

**COMP430 Digital Image Processing — Term Project**  
Author: Yaren Kahraman

## Overview

This project implements an **automatic adaptive notch filter** for removing periodic (sinusoidal) noise from grayscale images, without requiring prior knowledge of noise frequencies.

### Method Summary

1. Compute 2-D DFT magnitude spectrum
2. Detect noise peaks via MAD-robust statistical threshold:  `T = μ̃ + k·σ̂`
3. Retain only conjugate-symmetric peak pairs (real-image property)
4. Adaptive notch bandwidth:  `BW_i = BW_base · (1 + α · min(z_i, z_clip) / z_clip)`
5. Apply notch mask → reconstruct via IFFT

## Repository Structure

```
adaptive-notch-filter/
├── src/
│   ├── noise_generator.py   # Synthetic periodic noise injection
│   ├── baselines.py         # Baseline-1 (manual) and Baseline-2 (fixed BW)
│   ├── adaptive_notch.py    # Proposed method
│   ├── metrics.py           # PSNR, SSIM, SNR, MSE
│   └── run_experiment.py    # Full experiment pipeline
├── results/
│   └── figures/             # All generated figures
└── README.md
```

## Requirements

```bash
pip install numpy scipy matplotlib scikit-image Pillow
```

## Usage

```python
from src.adaptive_notch import adaptive_notch_filter
from src.noise_generator import add_periodic_noise
from skimage import data, img_as_float

img   = img_as_float(data.camera())
noisy = add_periodic_noise(img, frequencies=[(30,15),(20,40)],
                           amplitudes=[0.15,0.12])
filtered = adaptive_notch_filter(noisy, k=5.0, bw_base=5.0, alpha=0.5)
```

## Run Experiments

```bash
cd adaptive-notch-filter
python src/run_experiment.py
```

## Results (camera image, noise amplitude 0.15)

| Method | PSNR (dB) | SSIM | Note |
|--------|-----------|------|------|
| Noisy input | 17.92 | 0.299 | — |
| Baseline-1 (Manual) | 29.33 | 0.829 | Oracle frequencies |
| Baseline-2 (Fixed BW) | 29.33 | 0.829 | Oracle frequencies |
| **Proposed (Adaptive)** | **25.03** | **0.755** | **No prior knowledge** |

The proposed method achieves +7.1 dB PSNR over the noisy input without any prior knowledge of noise frequencies.

## AI Tool Usage

GitHub Copilot and Claude (Anthropic) were used to assist with code structure and debugging. All experimental design, analysis, and interpretations are the author's own work.
