"""
adaptive_notch.py  —  Proposed Adaptive Notch Filter
------------------------------------------------------
Pipeline
--------
1. Compute 2-D DFT magnitude spectrum (fftshift centred).
2. Detect noise peaks via MAD-robust threshold; require conjugate-symmetric
   counterpart; keep top-N pairs by amplitude.
3. Adaptive bandwidth per peak:
       BW_i = BW_base · (1 + α · ẑ_i)
   where ẑ_i = min(z_i, z_clip) / z_clip  ∈ [0, 1]
         z_i = (S_i − μ̃) / σ̂
   This maps any amplitude into [BW_base, BW_base·(1+α)] —
   i.e. the BW grows by at most a factor of (1+α) for the strongest peaks.

4. Suppress each peak and its conjugate twin in the FFT (zero mask).
5. Reconstruct via IFFT.

Paper notation:
   BW_i = BW_base · (1 + α · min(z_i, z_clip) / z_clip)
"""

import numpy as np


# ─── helpers ─────────────────────────────────────────────────────────────────

def _magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """fftshift-centred magnitude spectrum."""
    return np.fft.fftshift(np.abs(np.fft.fft2(image)))


def _background_stats(spectrum: np.ndarray,
                      dc_radius: int = 20) -> tuple[float, float]:
    """Robust (μ̃, σ̂) from spectrum outside DC zone."""
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    yy = (np.arange(H) - cy)[:, None]
    xx = (np.arange(W) - cx)[None, :]
    bg = spectrum[np.sqrt(yy**2 + xx**2) > dc_radius].ravel()
    mu    = float(np.median(bg))
    sigma = float(1.4826 * np.median(np.abs(bg - mu))) + 1e-8
    return mu, sigma


# ─── Step 2: detect symmetric peak pairs ─────────────────────────────────────

def detect_peaks(spectrum_shifted: np.ndarray,
                 k: float = 5.0,
                 dc_radius: int = 20,
                 max_peaks: int = 20) -> list[tuple[int, int]]:
    """
    Return up to `max_peaks` upper-half representatives of symmetric pairs.

    Threshold:  T = μ̃ + k · σ̂

    Both (r,c) and its conjugate (r_sym, c_sym) must exceed T.
    Pairs are ranked by combined amplitude.
    """
    H, W = spectrum_shifted.shape
    cy, cx = H // 2, W // 2
    mu, sigma = _background_stats(spectrum_shifted, dc_radius)
    T = mu + k * sigma

    rows, cols = np.where(spectrum_shifted > T)
    keep = np.sqrt((rows - cy)**2 + (cols - cx)**2) > dc_radius
    rows, cols = rows[keep].tolist(), cols[keep].tolist()

    cand = set(zip(rows, cols))
    seen, pairs = set(), []

    for r, c in zip(rows, cols):
        if (r, c) in seen:
            continue
        r_sym, c_sym = 2*cy - r, 2*cx - c
        if (r_sym, c_sym) in cand and (r_sym, c_sym) != (r, c):
            seen.update({(r, c), (r_sym, c_sym)})
            amp = float(spectrum_shifted[r, c] + spectrum_shifted[r_sym, c_sym])
            upper = (r, c) if r <= cy else (r_sym, c_sym)
            pairs.append((amp, upper[0], upper[1]))

    pairs.sort(reverse=True)
    return [(r, c) for (_, r, c) in pairs[:max_peaks]]


# ─── Step 3: adaptive bandwidth ───────────────────────────────────────────────

def adaptive_bandwidth(spectrum_shifted: np.ndarray,
                       peaks: list[tuple[int, int]],
                       bw_base: float = 5.0,
                       alpha: float = 0.5,
                       dc_radius: int = 20,
                       z_clip: float = 100.0) -> dict[tuple[int, int], float]:
    """
    BW_i = BW_base · (1 + α · min(z_i, z_clip) / z_clip)

    z_i   = (S_i − μ̃) / σ̂           [robust z-score]
    z_clip = upper clip for z normalisation

    This bounds BW in [BW_base, BW_base·(1 + α)] regardless of peak height,
    satisfying the paper's mathematical formulation.
    """
    mu, sigma = _background_stats(spectrum_shifted, dc_radius)
    bw_map = {}
    for (r, c) in peaks:
        z = max(0.0, (float(spectrum_shifted[r, c]) - mu) / sigma)
        z_norm = min(z, z_clip) / z_clip          # ∈ [0, 1]
        bw = bw_base * (1.0 + alpha * z_norm)
        bw_map[(r, c)] = max(float(bw), 2.0)
    return bw_map


# ─── Step 4: build mask ───────────────────────────────────────────────────────

def build_adaptive_mask(H: int, W: int,
                        peaks: list[tuple[int, int]],
                        bw_map: dict[tuple[int, int], float]) -> np.ndarray:
    """Multiplicative notch mask in FFT (non-shifted) order."""
    mask_shifted = np.ones((H, W), dtype=np.float64)
    cy, cx = H // 2, W // 2
    rr, cc = np.mgrid[0:H, 0:W]

    for (r, c), bw in bw_map.items():
        r_sym, c_sym = 2*cy - r, 2*cx - c
        for pr, pc in [(r, c), (r_sym, c_sym)]:
            mask_shifted[(rr - pr)**2 + (cc - pc)**2 < bw**2] = 0.0

    return np.fft.ifftshift(mask_shifted)


# ─── Main API ─────────────────────────────────────────────────────────────────

def adaptive_notch_filter(image: np.ndarray,
                          k: float = 5.0,
                          bw_base: float = 5.0,
                          alpha: float = 0.5,
                          dc_radius: int = 20,
                          max_peaks: int = 20,
                          z_clip: float = 100.0,
                          return_diagnostics: bool = False):
    """
    Adaptive notch filter for periodic noise removal.

    Parameters
    ----------
    image    : 2-D float [0,1]
    k        : threshold multiplier for peak detection
    bw_base  : base notch radius in frequency pixels
    alpha    : bandwidth adaptation strength  (0 = fixed BW)
    dc_radius: DC exclusion radius in pixels
    max_peaks: max symmetric pairs to suppress
    z_clip   : z-score upper clip for BW normalisation
    """
    H, W = image.shape
    spec   = _magnitude_spectrum(image)
    peaks  = detect_peaks(spec, k=k, dc_radius=dc_radius, max_peaks=max_peaks)
    bw_map = adaptive_bandwidth(spec, peaks, bw_base=bw_base, alpha=alpha,
                                dc_radius=dc_radius, z_clip=z_clip)
    mask   = build_adaptive_mask(H, W, peaks, bw_map)
    filtered = np.clip(np.real(np.fft.ifft2(np.fft.fft2(image) * mask)), 0.0, 1.0)

    if return_diagnostics:
        return filtered, {'spectrum': spec, 'peaks_raw': peaks,
                          'peaks_verified': peaks, 'bw_map': bw_map,
                          'mask': mask}
    return filtered
