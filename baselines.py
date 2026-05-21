"""
baselines.py
------------
Baseline 1 – Manual Notch Filter:
    User provides exact (u, v) peak locations; circular notch applied.

Baseline 2 – Fixed Notch Filter:
    Notch positions are hardcoded to the known noise frequencies.
    Bandwidth is constant regardless of peak energy.

Both operate in the Fourier domain.
"""

import numpy as np


# ── shared utility ──────────────────────────────────────────────────────────

def _notch_mask(H: int, W: int,
                centers: list[tuple[int, int]],
                bandwidth: float) -> np.ndarray:
    """
    Build a multiplicative mask (1 = keep, 0 = suppress).
    Suppresses circular regions of radius `bandwidth` around each center
    and its symmetric counterpart.
    """
    mask = np.ones((H, W), dtype=np.float64)
    u_ax = np.fft.fftfreq(W) * W   # pixel-frequency coordinates
    v_ax = np.fft.fftfreq(H) * H
    uu, vv = np.meshgrid(u_ax, v_ax)

    for (u0, v0) in centers:
        # suppress peak and its conjugate-symmetric twin
        for su, sv in [(u0, v0), (-u0, -v0)]:
            dist = np.sqrt((uu - su) ** 2 + (vv - sv) ** 2)
            mask[dist < bandwidth] = 0.0
    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a frequency-domain mask and return filtered spatial image."""
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    # mask is in fftfreq order, no shift needed
    F_filtered = F * mask
    filtered = np.real(np.fft.ifft2(F_filtered))
    return np.clip(filtered, 0.0, 1.0)


# ── Baseline 1: Manual Notch Filter ─────────────────────────────────────────

def manual_notch_filter(image: np.ndarray,
                        centers: list[tuple[int, int]],
                        bandwidth: float = 10.0) -> np.ndarray:
    """
    Remove periodic noise at user-specified frequency centers.

    Parameters
    ----------
    image     : 2D float array [0,1]
    centers   : list of (u, v) frequency-pixel coordinates to suppress
    bandwidth : notch radius in frequency pixels

    Returns
    -------
    Filtered image [0,1]
    """
    H, W = image.shape
    mask = _notch_mask(H, W, centers, bandwidth)
    return apply_mask_to_image(image, mask)


# ── Baseline 2: Fixed Notch Filter ──────────────────────────────────────────

def fixed_notch_filter(image: np.ndarray,
                       known_frequencies: list[tuple[int, int]],
                       bandwidth: float = 10.0) -> np.ndarray:
    """
    Remove periodic noise at pre-defined, fixed frequency locations.
    Bandwidth is constant regardless of actual peak energy.

    Parameters
    ----------
    image            : 2D float array [0,1]
    known_frequencies: list of (u, v) pairs — assumed known a priori
    bandwidth        : fixed notch radius in frequency pixels

    Returns
    -------
    Filtered image [0,1]
    """
    H, W = image.shape
    mask = _notch_mask(H, W, known_frequencies, bandwidth)
    return apply_mask_to_image(image, mask)


# ── Baseline 3: Automatic Detection + Fixed Notch ────────────────────────────

def auto_fixed_notch_filter(image: np.ndarray,
                            k: float = 5.0,
                            bandwidth: float = 5.0,
                            dc_radius: int = 20,
                            max_peaks: int = 20) -> np.ndarray:
    """
    Automatic peak detection (same as proposed method) but with a
    FIXED bandwidth instead of adaptive. This isolates the contribution
    of the adaptive bandwidth component.

    Parameters
    ----------
    image     : 2D float array [0, 1]
    k         : peak detection threshold multiplier
    bandwidth : fixed notch radius in frequency pixels
    dc_radius : DC exclusion zone radius
    max_peaks : max symmetric pairs to suppress

    Returns
    -------
    Filtered image [0, 1]
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from adaptive_notch import _magnitude_spectrum, detect_peaks, build_adaptive_mask

    H, W = image.shape
    spec  = _magnitude_spectrum(image)
    peaks = detect_peaks(spec, k=k, dc_radius=dc_radius, max_peaks=max_peaks)

    # Fixed bandwidth for all peaks (no amplitude-proportional scaling)
    bw_map = {p: float(bandwidth) for p in peaks}
    mask   = build_adaptive_mask(H, W, peaks, bw_map)

    F = np.fft.fft2(image)
    filtered = np.clip(np.real(np.fft.ifft2(F * mask)), 0.0, 1.0)
    return filtered
