# Adaptive Notch Filtering for Periodic Noise Removal

**COMP430 Digital Image Processing — Term Project**  
Author: Yaren Kahraman  
Abdullah Gül University, Department of Industrial Engineering, Kayseri, Turkey

## Overview

This project implements a fully automatic adaptive notch filter that removes periodic (sinusoidal) noise from grayscale images without requiring prior knowledge of noise frequencies.

### Method Summary

1. Compute 2-D DFT magnitude spectrum
2. Detect noise peaks via MAD-robust statistical threshold: `T = μ̃ + k·σ̂`
3. Retain only conjugate-symmetric peak pairs (real-image property)
4. Adaptive notch bandwidth: `BW_i = BW_base · (1 + α · min(z_i, z_clip) / z_clip)`
5. Apply notch mask → reconstruct via IFFT

## Repository Structure

```
adaptive-notch-filter/
├── noise_generator.py   # Synthetic periodic noise injection
├── baselines.py         # Baseline-1 (manual), Baseline-2 (fixed BW), Baseline-3 (auto+fixed)
├── adaptive_notch.py    # Proposed method (auto detection + adaptive BW)
├── metrics.py           # PSNR, SSIM, SNR, MSE
├── run_experiment.py    # Full experiment pipeline
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```python
from adaptive_notch import adaptive_notch_filter
from noise_generator import add_periodic_noise
from skimage import data, img_as_float

img   = img_as_float(data.camera())
noisy = add_periodic_noise(img, frequencies=[(30,15),(20,40)],
                           amplitudes=[0.15,0.12])
filtered = adaptive_notch_filter(noisy, k=5.0, bw_base=5.0, alpha=0.5)
```

## Run Experiments

```bash
python run_experiment.py
```

## Results (camera image, noise amplitude 0.15)

| Method | PSNR (dB) | SSIM | Note |
|--------|-----------|------|------|
| Noisy input | 17.92 | 0.299 | — |
| Baseline-1 (Manual, fixed BW) | 29.33 | 0.829 | Oracle: uses true frequencies |
| Baseline-2 (Manual, fixed BW) | 29.33 | 0.829 | Oracle: uses true frequencies |
| Baseline-3 (Auto detect, fixed BW) | — | — | Automatic detection, no BW adaptation |
| **Proposed (Auto detect, adaptive BW)** | **25.03** | **0.755** | **No prior knowledge** |

Baseline-1 and Baseline-2 are oracle baselines because they use the true synthetic noise frequencies as input. The proposed method achieves significant noise reduction without any frequency prior.

## AI Tool Usage

Claude (Anthropic) was used to assist with code structure and debugging. All experimental design, analysis, and interpretations are the author's own work.
