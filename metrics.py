"""
metrics.py
----------
Quantitative evaluation: PSNR, SSIM, MSE, SNR.
All functions expect float images in [0, 1].
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_func


def mse(reference: np.ndarray, restored: np.ndarray) -> float:
    """Mean Squared Error (lower is better)."""
    return float(np.mean((reference - restored) ** 2))


def psnr(reference: np.ndarray, restored: np.ndarray,
         data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB (higher is better).
    PSNR = 10 * log10(data_range^2 / MSE)
    """
    m = mse(reference, restored)
    if m == 0:
        return float('inf')
    return float(10 * np.log10(data_range ** 2 / m))


def snr(reference: np.ndarray, restored: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB (higher is better).
    SNR = 10 * log10(var(signal) / MSE)
    """
    signal_power = float(np.var(reference))
    noise_power = mse(reference, restored)
    if noise_power == 0:
        return float('inf')
    return float(10 * np.log10(signal_power / noise_power))


def ssim(reference: np.ndarray, restored: np.ndarray,
         data_range: float = 1.0) -> float:
    """Structural Similarity Index (higher is better, max=1)."""
    return float(ssim_func(reference, restored, data_range=data_range))


def evaluate_all(reference: np.ndarray,
                 restored: np.ndarray) -> dict[str, float]:
    """Return all metrics as a dictionary."""
    return {
        'MSE':  mse(reference, restored),
        'PSNR': psnr(reference, restored),
        'SNR':  snr(reference, restored),
        'SSIM': ssim(reference, restored),
    }
