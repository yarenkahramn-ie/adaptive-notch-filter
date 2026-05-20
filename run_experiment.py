"""
run_experiment.py
-----------------
Full experiment pipeline for the Adaptive Notch Filter project.
Reproduces all tables and figures used in the IEEE paper.

Usage:
    python src/run_experiment.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data as sk_data, color, img_as_float

from noise_generator import add_periodic_noise
from baselines import manual_notch_filter, fixed_notch_filter
from adaptive_notch import adaptive_notch_filter
from metrics import evaluate_all

# ── Configuration ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_FREQS  = [(30, 15), (20, 40)]
NOISE_AMPS   = [0.15, 0.12]
NOISE_PHASES = [0.5, 1.2]

K          = 5.0
BW_BASE    = 5.0
ALPHA      = 0.5
DC_RADIUS  = 20
MAX_PEAKS  = 20
BASELINE_BW = 10.0


# ── Helpers ────────────────────────────────────────────────────────────────
def load(d):
    a = img_as_float(d)
    return color.rgb2gray(a) if a.ndim == 3 else a


def get_images():
    return (
        [load(sk_data.camera()), load(sk_data.astronaut()),
         load(sk_data.chelsea()), load(sk_data.coins()), load(sk_data.horse())],
        ['camera', 'astronaut', 'chelsea', 'coins', 'horse']
    )


# ── Main experiment ────────────────────────────────────────────────────────
def run_main():
    images, names = get_images()
    all_res = {}

    print(f"{'Image':<12} {'Method':<14}" +
          "".join(f"{m:>8}" for m in ['MSE', 'PSNR', 'SNR', 'SSIM']))
    print('-' * 64)

    for img, name in zip(images, names):
        noisy    = add_periodic_noise(img, NOISE_FREQS, NOISE_AMPS, NOISE_PHASES)
        bl1      = manual_notch_filter(noisy, centers=NOISE_FREQS, bandwidth=BASELINE_BW)
        bl2      = fixed_notch_filter (noisy, known_frequencies=NOISE_FREQS, bandwidth=BASELINE_BW)
        proposed, diag = adaptive_notch_filter(
            noisy, k=K, bw_base=BW_BASE, alpha=ALPHA,
            dc_radius=DC_RADIUS, max_peaks=MAX_PEAKS,
            return_diagnostics=True)

        all_res[name] = dict(ref=img, noisy=noisy, bl1=bl1, bl2=bl2,
                             proposed=proposed, diag=diag)

        for label, r in [('Noisy', noisy), ('Manual', bl1),
                         ('Fixed', bl2), ('Proposed', proposed)]:
            m = evaluate_all(img, r)
            print(f"{name:<12} {label:<14}" +
                  "".join(f"{m[k]:>8.4f}" for k in ['MSE', 'PSNR', 'SNR', 'SSIM']))
        print()

    # Save comparison figures
    for name, d in all_res.items():
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        pairs = [('Original', d['ref']), ('Noisy', d['noisy']),
                 ('Baseline-1\n(Manual)', d['bl1']),
                 ('Baseline-2\n(Fixed BW)', d['bl2']),
                 ('Proposed\n(Adaptive)', d['proposed'])]
        for ax, (ti, im) in zip(axes, pairs):
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
            ax.set_title(ti, fontsize=9, fontweight='bold')
            ax.axis('off')
        plt.suptitle(f'Adaptive Notch Filter — {name}', fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{name}_comparison.png', dpi=130, bbox_inches='tight')
        plt.close()

    # Spectrum analysis (camera)
    d = all_res['camera']
    diag = d['diag']
    spec_log = np.log1p(diag['spectrum'])
    mask_vis = np.fft.fftshift(diag['mask'])
    H, W = d['ref'].shape; cy, cx = H//2, W//2
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(spec_log, cmap='inferno')
    axes[0].set_title('Log Magnitude Spectrum', fontweight='bold')
    axes[0].axis('off')
    for (r, c), bw in diag['bw_map'].items():
        for pr, pc in [(r, c), (2*cy-r, 2*cx-c)]:
            axes[0].add_patch(plt.Circle((pc, pr), bw, color='cyan', fill=False, lw=2))
    axes[1].imshow(spec_log * mask_vis, cmap='inferno')
    axes[1].set_title('After Notch Mask', fontweight='bold'); axes[1].axis('off')
    axes[2].imshow(d['proposed'], cmap='gray')
    axes[2].set_title('Filtered Image', fontweight='bold'); axes[2].axis('off')
    plt.suptitle('Spectral Analysis — camera', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/camera_spectrum_analysis.png', dpi=130, bbox_inches='tight')
    plt.close()

    return all_res


# ── Ablation studies ───────────────────────────────────────────────────────
def run_ablation(img, noisy):
    # k ablation
    k_vals = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    abl_k = {}
    for k in k_vals:
        f, d = adaptive_notch_filter(noisy, k=k, bw_base=BW_BASE, alpha=ALPHA,
                                     max_peaks=MAX_PEAKS, return_diagnostics=True)
        m = evaluate_all(img, f); m['n_peaks'] = len(d['peaks_verified'])
        abl_k[k] = m

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    ax1.plot(k_vals, [abl_k[k]['PSNR'] for k in k_vals], 'o-b', lw=2)
    ax1.set_xlabel('k'); ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR vs k'); ax1.grid(alpha=0.3)
    ax2.plot(k_vals, [abl_k[k]['SSIM'] for k in k_vals], 's-g', lw=2)
    ax2.set_xlabel('k'); ax2.set_ylabel('SSIM'); ax2.set_title('SSIM vs k'); ax2.grid(alpha=0.3)
    ax3.plot(k_vals, [abl_k[k]['n_peaks'] for k in k_vals], '^-r', lw=2)
    ax3.set_xlabel('k'); ax3.set_ylabel('# Peaks'); ax3.set_title('Peaks vs k'); ax3.grid(alpha=0.3)
    plt.suptitle('Ablation: Threshold k', fontsize=12); plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ablation_k.png', dpi=130, bbox_inches='tight'); plt.close()

    # alpha ablation
    alpha_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    abl_a = {}
    for a in alpha_vals:
        f = adaptive_notch_filter(noisy, k=K, bw_base=BW_BASE, alpha=a, max_peaks=MAX_PEAKS)
        abl_a[a] = evaluate_all(img, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(alpha_vals, [abl_a[a]['PSNR'] for a in alpha_vals], 'o-b', lw=2)
    ax1.set_xlabel('alpha'); ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR vs Alpha'); ax1.grid(alpha=0.3)
    ax2.plot(alpha_vals, [abl_a[a]['SSIM'] for a in alpha_vals], 's-g', lw=2)
    ax2.set_xlabel('alpha'); ax2.set_ylabel('SSIM'); ax2.set_title('SSIM vs Alpha'); ax2.grid(alpha=0.3)
    plt.suptitle('Ablation: Bandwidth Adaptation Strength alpha', fontsize=12); plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ablation_alpha.png', dpi=130, bbox_inches='tight'); plt.close()

    # noise level analysis
    noise_levels = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    res_nl = {'manual': [], 'proposed': []}
    for amp in noise_levels:
        n = add_periodic_noise(img, NOISE_FREQS, [amp, amp*0.8], NOISE_PHASES)
        bl = manual_notch_filter(n, centers=NOISE_FREQS, bandwidth=BASELINE_BW)
        pr = adaptive_notch_filter(n, k=K, bw_base=BW_BASE, alpha=ALPHA)
        res_nl['manual'].append(evaluate_all(img, bl)['PSNR'])
        res_nl['proposed'].append(evaluate_all(img, pr)['PSNR'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise_levels, res_nl['manual'],   'o-b', lw=2, label='Manual (oracle)')
    ax.plot(noise_levels, res_nl['proposed'], 's--r', lw=2, label='Proposed (adaptive)')
    ax.set_xlabel('Noise Amplitude'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR vs Noise Level'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/noise_level_analysis.png', dpi=130, bbox_inches='tight'); plt.close()
    print('Ablation figures saved.')


if __name__ == '__main__':
    print("=" * 64)
    print("ADAPTIVE NOTCH FILTER — EXPERIMENT RUNNER")
    print("=" * 64)
    all_res = run_main()
    print("\n[Ablation study on camera image]")
    d = all_res['camera']
    run_ablation(d['ref'], d['noisy'])
    print("\nAll done. Figures saved to results/figures/")
