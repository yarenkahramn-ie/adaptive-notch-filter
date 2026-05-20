"""
noise_generator.py
------------------
Adds synthetic periodic noise to grayscale images.
Noise model: sum of sinusoidal components at specified frequencies.

  I_noisy(x,y) = I(x,y) + sum_k [ A_k * sin(2*pi*(u_k*x + v_k*y) + phi_k) ]
"""

import numpy as np
from skimage import io, color, img_as_float, img_as_ubyte


def add_periodic_noise(image: np.ndarray,
                       frequencies: list[tuple],
                       amplitudes: list[float] | None = None,
                       phases: list[float] | None = None,
                       seed: int = 42) -> np.ndarray:
    """
    Add sinusoidal periodic noise to a grayscale float image.

    Parameters
    ----------
    image       : 2D float array in [0, 1]
    frequencies : list of (u, v) tuples — cycles per image width/height
    amplitudes  : noise amplitude for each frequency (default 0.15 each)
    phases      : phase offset in radians (default random)
    seed        : random seed for reproducibility

    Returns
    -------
    noisy image clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)
    H, W = image.shape
    if amplitudes is None:
        amplitudes = [0.15] * len(frequencies)
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, len(frequencies)).tolist()

    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    noise = np.zeros((H, W), dtype=np.float64)
    for (u, v), A, phi in zip(frequencies, amplitudes, phases):
        noise += A * np.sin(2 * np.pi * (u * xx / W + v * yy / H) + phi)

    return np.clip(image + noise, 0.0, 1.0)


def load_gray(path: str) -> np.ndarray:
    """Load image from path and convert to float grayscale [0,1]."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img_as_float(img)


def save_gray(image: np.ndarray, path: str) -> None:
    """Save float grayscale image to disk."""
    io.imsave(path, img_as_ubyte(np.clip(image, 0, 1)))
