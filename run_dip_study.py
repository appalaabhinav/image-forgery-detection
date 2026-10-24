"""
DIP Study Execution & Visualization Pipeline
============================================
Executes the comprehensive Digital Image Processing study for the CG-1050 Image Forgery
Dataset, producing high-resolution visualization grids, histograms, and quantitative
forensic metrics.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import dip_pipeline as dip

# Ensure output directories exist
VIZ_DIR = Path("outputs/dip_visualizations")
ANALYSIS_DIR = Path("outputs/dip_analysis")
VIZ_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Set consistent matplotlib plotting style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def find_representative_samples():
    """Locates representative Authentic and Tampered images in the dataset."""
    rgb_orig = "TRAINING_CG-1050/TRAINING/ORIGINAL/Im10_2.jpg"
    rgb_tamp = "TRAINING_CG-1050/TRAINING/TAMPERED/Im10_f2.jpg"
    
    gray_orig = "TRAINING_CG-1050/TRAINING/ORIGINAL/Im100_2_cm.jpg"
    gray_tamp = "TRAINING_CG-1050/TRAINING/TAMPERED/Im100_cm2.jpg"
    
    # Fallback search if specific paths differ
    if not (os.path.exists(rgb_orig) and os.path.exists(rgb_tamp)):
        import glob
        origs = sorted(glob.glob("TRAINING_CG-1050/**/ORIGINAL/*.jpg", recursive=True))
        tamps = sorted(glob.glob("TRAINING_CG-1050/**/TAMPERED/*.jpg", recursive=True))
        rgb_orig = origs[0]
        rgb_tamp = tamps[0]
        gray_orig = origs[1]
        gray_tamp = tamps[1]

    return rgb_orig, rgb_tamp, gray_orig, gray_tamp


def plot_image_and_histogram(ax_img, ax_hist, img, title, is_rgb=True, cmap=None):
    """Utility to plot an image alongside its intensity histogram."""
    if is_rgb and img.ndim == 3 and img.shape[2] == 3:
        ax_img.imshow(img)
    else:
        gray_view = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ax_img.imshow(gray_view, cmap=cmap or "gray")
    ax_img.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax_img.axis("off")

    # Histogram
    if is_rgb and img.ndim == 3 and img.shape[2] == 3:
        colors = ("red", "green", "blue")
        for i, col in enumerate(colors):
            hist, _ = np.histogram(img[:, :, i].ravel(), bins=256, range=[0, 256])
            ax_hist.plot(hist, color=col, alpha=0.75, linewidth=1.0)
    else:
        gray_view = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray_view.ravel(), bins=256, range=[0, 256])
        ax_hist.plot(hist, color="#333333", linewidth=1.2)
        ax_hist.fill_between(range(256), hist, color="#888888", alpha=0.3)

    ax_hist.set_xlim([0, 255])
    ax_hist.set_title("Histogram", fontsize=8)
    ax_hist.tick_params(axis="both", labelsize=7)
    ax_hist.grid(True, linestyle="--", alpha=0.3)


# ==============================================================================
# Pipeline 1: Intensity Transformations
# ==============================================================================
def run_intensity_transformations(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 1. Intensity Transformations...")
    img = img_rgb.copy()
    
    neg = dip.image_negative(img)
    log_t = dip.logarithmic_transform(img)
    gamma_05 = dip.gamma_correction(img, gamma=0.5)
    gamma_18 = dip.gamma_correction(img, gamma=1.8)
    stretched = dip.contrast_stretching(img, p_low=2.0, p_high=98.0)

    items = [
        ("Original Image", img),
        ("Image Negative (s = 255 - r)", neg),
        ("Logarithmic (s = c*log(1+r))", log_t),
        (r"Gamma Correction ($\gamma = 0.5$)", gamma_05),
        (r"Gamma Correction ($\gamma = 1.8$)", gamma_18),
        ("Contrast Stretching (2-98%)", stretched)
    ]

    fig, axes = plt.subplots(2, 6, figsize=(20, 6.5))
    for col, (title, proc_img) in enumerate(items):
        plot_image_and_histogram(axes[0, col], axes[1, col], proc_img, title, is_rgb=True)

    plt.suptitle(f"1. Intensity Transformations & Histograms ({tag} Image)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = VIZ_DIR / f"01_intensity_transformations_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 2: Histogram Processing
# ==============================================================================
def run_histogram_processing(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 2. Histogram Processing...")
    img = img_rgb.copy()

    ghe = dip.global_histogram_equalization(img)
    clahe_res = dip.clahe(img, clip_limit=2.5, tile_grid_size=(8, 8))

    items = [
        ("Original Image", img),
        ("Global Equalization (GHE on Y)", ghe),
        ("CLAHE (Clip=2.5, Grid=8x8 on L)", clahe_res)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5))
    for col, (title, proc_img) in enumerate(items):
        plot_image_and_histogram(axes[0, col], axes[1, col], proc_img, title, is_rgb=True)

    plt.suptitle(f"2. Histogram Processing & Equalization ({tag} Image)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = VIZ_DIR / f"02_histogram_processing_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 3: Noise Filtering
# ==============================================================================
def run_noise_filtering(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 3. Noise Filtering...")
    img = img_rgb.copy()

    mean_f = dip.mean_filter(img, kernel_size=5)
    gauss_f = dip.gaussian_filter(img, kernel_size=5, sigma=1.2)
    med_f = dip.median_filter(img, kernel_size=5)
    bilat_f = dip.bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)
    nlm_f = dip.non_local_means_filter(img, h=10, template_window=7, search_window=21)

    items = [
        ("Original Image", img),
        ("Mean Filter (5x5 Box)", mean_f),
        (r"Gaussian Filter ($\sigma=1.2$, 5x5)", gauss_f),
        ("Median Filter (5x5 Rank)", med_f),
        (r"Bilateral Filter ($d=9, \sigma=75$)", bilat_f),
        ("Non-Local Means (h=10)", nlm_f)
    ]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    for col, (title, proc_img) in enumerate(items):
        axes[col].imshow(proc_img)
        axes[col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[col].axis("off")

    plt.suptitle(f"3. Spatial Noise Filtering Techniques ({tag} Image)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = VIZ_DIR / f"03_noise_filtering_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 4: Sharpening Filters
# ==============================================================================
def run_sharpening_filters(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 4. Sharpening Filters...")
    img = img_rgb.copy()

    lap_sharp = dip.laplacian_sharpening(img, kernel_type="8-neighbor", alpha=1.0)
    unsharp = dip.unsharp_masking(img, sigma=1.5, strength=1.5)
    high_boost_15 = dip.high_boost_filter(img, A=1.5, sigma=1.5)
    high_boost_20 = dip.high_boost_filter(img, A=2.0, sigma=1.5)

    items = [
        ("Original Image", img),
        (r"Laplacian Sharpening ($\nabla^2 f$)", lap_sharp),
        ("Unsharp Masking (k=1.5)", unsharp),
        ("High-Boost (A=1.5)", high_boost_15),
        ("High-Boost (A=2.0)", high_boost_20)
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for col, (title, proc_img) in enumerate(items):
        axes[col].imshow(proc_img)
        axes[col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[col].axis("off")

    plt.suptitle(f"4. Image Sharpening & High-Boost Enhancement ({tag} Image)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = VIZ_DIR / f"04_sharpening_filters_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 5: Frequency-Domain Analysis
# ==============================================================================
def run_frequency_domain_analysis(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 5. Frequency-Domain Analysis...")
    img = img_rgb.copy()

    spectrum = dip.fourier_magnitude_spectrum(img)
    lpf = dip.gaussian_low_pass_filter(img, d0=40)
    hpf = dip.gaussian_high_pass_filter(img, d0=35)
    homomorphic = dip.homomorphic_filter(img, gamma_l=0.5, gamma_h=1.5, d0=30)

    items = [
        ("Original Image (Grayscale)", dip.to_numpy_gray(img), "gray"),
        ("2D Fourier Spectrum (log|F|)", spectrum, "magma"),
        ("Gaussian LPF (D0=40)", lpf, "gray"),
        ("Gaussian HPF (D0=35)", hpf, "gray"),
        ("Homomorphic Filter (Reflectance)", homomorphic, "gray")
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for col, (title, proc_img, cmap) in enumerate(items):
        axes[col].imshow(proc_img, cmap=cmap)
        axes[col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[col].axis("off")

    plt.suptitle(f"5. Frequency-Domain Transforms & Spatial Frequency Filtering ({tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = VIZ_DIR / f"05_frequency_domain_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 6: Edge Detection
# ==============================================================================
def run_edge_detection(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 6. Edge Detection...")
    img = img_rgb.copy()

    sobel_m = dip.sobel_gradient_magnitude(img, ksize=3)
    prewitt_m = dip.prewitt_gradient_magnitude(img)
    lap = dip.laplacian_edge(img, ksize=3)
    log_edge = dip.laplacian_of_gaussian(img, ksize=5, sigma=1.2)
    canny = dip.canny_edge_detector(img, low_thresh=50, high_thresh=150)

    items = [
        ("Original Image", img, None),
        ("Sobel Gradient Magnitude", sobel_m, "viridis"),
        ("Prewitt Gradient Magnitude", prewitt_m, "viridis"),
        ("Laplacian (|d^2 f|)", lap, "inferno"),
        (r"LoG ($\nabla^2 G_{\sigma=1.2}$)", log_edge, "inferno"),
        ("Canny Edge Detector", canny, "gray")
    ]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    for col, (title, proc_img, cmap) in enumerate(items):
        if cmap is None:
            axes[col].imshow(proc_img)
        else:
            axes[col].imshow(proc_img, cmap=cmap)
        axes[col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[col].axis("off")

    plt.suptitle(f"6. Classical Edge Detection Operators ({tag} Image)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = VIZ_DIR / f"06_edge_detection_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 7: Morphological Processing
# ==============================================================================
def run_morphological_processing(img_rgb: np.ndarray, tag: str = "RGB"):
    print(f"[{tag}] Executing 7. Morphological Processing...")
    img = img_rgb.copy()

    thresh_val, otsu_bin = dip.otsu_thresholding(img)
    opening = dip.morphological_opening(img, kernel_size=5, shape="rect")
    closing = dip.morphological_closing(img, kernel_size=5, shape="rect")
    tophat = dip.morphological_tophat(img, kernel_size=9, shape="rect")
    blackhat = dip.morphological_blackhat(img, kernel_size=9, shape="rect")

    items = [
        ("Original (Grayscale)", dip.to_numpy_gray(img), "gray"),
        (f"Otsu Threshold (T={thresh_val})", otsu_bin, "gray"),
        ("Morphological Opening (5x5)", opening, "gray"),
        ("Morphological Closing (5x5)", closing, "gray"),
        ("Top-Hat (Bright Micro-features)", tophat, "magma"),
        ("Black-Hat (Dark Micro-features)", blackhat, "magma")
    ]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    for col, (title, proc_img, cmap) in enumerate(items):
        axes[col].imshow(proc_img, cmap=cmap)
        axes[col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[col].axis("off")

    plt.suptitle(f"7. Morphological Transformations & Otsu Binarization ({tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = VIZ_DIR / f"07_morphological_processing_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Pipeline 8: Forensic ELA Comparison (Authentic vs Tampered Pair)
# ==============================================================================
def run_forensic_ela_study(orig_path: str, tamp_path: str, tag: str = "RGB"):
    print(f"[{tag}] Executing 8. Forensic ELA & Splicing Residual Analysis...")
    im_orig = dip.to_numpy_rgb(orig_path)
    im_tamp = dip.to_numpy_rgb(tamp_path)

    ela_orig = dip.compute_ela(orig_path, quality=90, scale=12.0)
    ela_tamp = dip.compute_ela(tamp_path, quality=90, scale=12.0)

    # Difference residual between authentic reference and tampered counterpart
    diff = np.abs(im_orig.astype(np.float32) - im_tamp.astype(np.float32))
    diff_vis = np.clip(diff * 5.0, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(im_orig)
    axes[0, 0].set_title("Authentic (ORIGINAL) Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ela_orig)
    axes[0, 1].set_title("Authentic ELA Map (Uniform Error)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(diff_vis)
    axes[0, 2].set_title("Ground-Truth Tampering Delta (|O - T| x 5)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(im_tamp)
    axes[1, 0].set_title("Tampered (FORGED) Image", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ela_tamp)
    axes[1, 1].set_title("Tampered ELA Map (Discrepancy at Seams)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Tampered minus Authentic ELA discrepancy
    ela_diff = np.abs(ela_tamp.astype(np.float32) - ela_orig.astype(np.float32))
    ela_diff_vis = np.clip(ela_diff * 3.0, 0, 255).astype(np.uint8)
    axes[1, 2].imshow(ela_diff_vis)
    axes[1, 2].set_title("Forensic Compression Residual Mismatch", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle(f"Forensic Error Level Analysis (ELA) & Compression Diagnostics ({tag} Pair)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = VIZ_DIR / f"08_forensic_ela_comparison_{tag.lower()}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {out_path}")


# ==============================================================================
# Part 4: Quantitative DIP Analysis
# ==============================================================================
def run_quantitative_analysis(orig_path: str, tamp_path: str):
    print("\n" + "="*70)
    print("PART 4: QUANTITATIVE DIGITAL IMAGE PROCESSING ANALYSIS")
    print("="*70)

    im_orig = dip.to_numpy_rgb(orig_path)
    im_tamp = dip.to_numpy_rgb(tamp_path)

    methods = [
        ("Raw Input", lambda x: x),
        ("Image Negative", dip.image_negative),
        ("Logarithmic Transform", dip.logarithmic_transform),
        ("Gamma Correction (g=1.5)", lambda x: dip.gamma_correction(x, gamma=1.5)),
        ("Contrast Stretching", dip.contrast_stretching),
        ("Global Hist Equalization", dip.global_histogram_equalization),
        ("CLAHE (Clip=2.5)", lambda x: dip.clahe(x, clip_limit=2.5)),
        ("Mean Filter (5x5)", dip.mean_filter),
        ("Gaussian Filter (5x5)", dip.gaussian_filter),
        ("Median Filter (5x5)", dip.median_filter),
        ("Bilateral Filter", dip.bilateral_filter),
        ("Non-Local Means", dip.non_local_means_filter),
        ("Laplacian Sharpening", dip.laplacian_sharpening),
        ("Unsharp Masking", dip.unsharp_masking),
        ("High-Boost (A=1.5)", dip.high_boost_filter),
        ("Gaussian LPF (D0=40)", dip.gaussian_low_pass_filter),
        ("Gaussian HPF (D0=35)", dip.gaussian_high_pass_filter),
        ("Homomorphic Filter", dip.homomorphic_filter),
        ("Sobel Gradient Mag", dip.sobel_gradient_magnitude),
        ("Prewitt Gradient Mag", dip.prewitt_gradient_magnitude),
        ("Laplacian Edge", dip.laplacian_edge),
        ("LoG (sigma=1.2)", dip.laplacian_of_gaussian),
        ("Canny Edge Detector", dip.canny_edge_detector),
        ("Otsu Binarization", lambda x: dip.otsu_thresholding(x)[1]),
        ("Morph Opening (5x5)", dip.morphological_opening),
        ("Morph Closing (5x5)", dip.morphological_closing),
        ("Morph Top-Hat", dip.morphological_tophat),
        ("Morph Black-Hat", dip.morphological_blackhat),
        ("Forensic ELA (q=90)", dip.compute_ela)
    ]

    records = []
    for name, func in methods:
        print(f"  Measuring {name}...")
        # Benchmark runtime on authentic image
        runtime_ms = dip.benchmark_runtime(func, im_orig, n_repeats=3)

        # Process both authentic and tampered
        p_orig = func(im_orig)
        p_tamp = func(im_tamp)

        # Quantitative Metrics for Authentic
        stats_orig = dip.compute_intensity_stats(p_orig)
        ent_orig = dip.compute_shannon_entropy(p_orig)
        cont_orig = dip.compute_contrast_metrics(p_orig)
        edge_orig = dip.compute_edge_metrics(p_orig)

        # Quantitative Metrics for Tampered
        stats_tamp = dip.compute_intensity_stats(p_tamp)
        ent_tamp = dip.compute_shannon_entropy(p_tamp)
        cont_tamp = dip.compute_contrast_metrics(p_tamp)
        edge_tamp = dip.compute_edge_metrics(p_tamp)

        # PSNR and SSIM between reference original and processed
        psnr_ssim_orig = dip.compute_psnr_ssim(im_orig, p_orig)
        psnr_ssim_tamp = dip.compute_psnr_ssim(im_tamp, p_tamp)

        records.append({
            "Method": name,
            "Runtime_ms": round(runtime_ms, 2),
            "Orig_Mean": round(stats_orig["mean_intensity"], 2),
            "Tamp_Mean": round(stats_tamp["mean_intensity"], 2),
            "Orig_Std": round(stats_orig["std_intensity"], 2),
            "Tamp_Std": round(stats_tamp["std_intensity"], 2),
            "Orig_Entropy_bits": round(ent_orig, 3),
            "Tamp_Entropy_bits": round(ent_tamp, 3),
            "Orig_RMS_Contrast": round(cont_orig["rms_contrast"], 3),
            "Tamp_RMS_Contrast": round(cont_tamp["rms_contrast"], 3),
            "Orig_Michelson": round(cont_orig["michelson_contrast"], 3),
            "Tamp_Michelson": round(cont_tamp["michelson_contrast"], 3),
            "Orig_Edge_Density": round(edge_orig["edge_density"], 4),
            "Tamp_Edge_Density": round(edge_tamp["edge_density"], 4),
            "Orig_Mean_Gradient": round(edge_orig["mean_gradient_magnitude"], 2),
            "Tamp_Mean_Gradient": round(edge_tamp["mean_gradient_magnitude"], 2),
            "PSNR_vs_Raw_dB": round(psnr_ssim_orig["psnr_db"], 2),
            "SSIM_vs_Raw": round(psnr_ssim_orig["ssim"], 4)
        })

    df = pd.DataFrame(records)
    csv_path = ANALYSIS_DIR / "dip_quantitative_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Summary Table Saved] -> {csv_path}")

    # Generate Comparative Bar Plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Entropy Comparison
    df_plot = df.set_index("Method")
    df_plot[["Orig_Entropy_bits", "Tamp_Entropy_bits"]].plot(kind="bar", ax=axes[0, 0], color=["#2b5c8f", "#d95f02"], width=0.7)
    axes[0, 0].set_title("Shannon Entropy (bits/pixel) - Authentic vs Tampered", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Entropy (bits)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=90, labelsize=8)

    # 2. RMS Contrast Comparison
    df_plot[["Orig_RMS_Contrast", "Tamp_RMS_Contrast"]].plot(kind="bar", ax=axes[0, 1], color=["#2b5c8f", "#d95f02"], width=0.7)
    axes[0, 1].set_title("RMS Contrast Comparison", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Normalized RMS Contrast")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=8)

    # 3. Edge Density Comparison
    df_plot[["Orig_Edge_Density", "Tamp_Edge_Density"]].plot(kind="bar", ax=axes[1, 0], color=["#2b5c8f", "#d95f02"], width=0.7)
    axes[1, 0].set_title("Canny Edge Density Comparison", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Fraction of Edge Pixels")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=8)

    # 4. Filter Runtime (Log Scale)
    df_plot["Runtime_ms"].plot(kind="bar", ax=axes[1, 1], color="#4daf4a", width=0.6)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("DIP Method Execution Runtime (Log Scale ms)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Runtime (ms, Log Scale)")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=90, labelsize=8)

    plt.suptitle("Quantitative Image Quality, Structural & Computational DIP Analysis",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_plot = ANALYSIS_DIR / "dip_metrics_comparison.png"
    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved analysis plots: {out_plot}")

    return df


def main():
    rgb_orig, rgb_tamp, gray_orig, gray_tamp = find_representative_samples()
    print(f"Sample RGB Pair: \n  Original: {rgb_orig}\n  Tampered: {rgb_tamp}")
    print(f"Sample Gray Pair: \n  Original: {gray_orig}\n  Tampered: {gray_tamp}")

    im_rgb = dip.to_numpy_rgb(rgb_orig)
    im_gray = dip.to_numpy_gray(gray_orig)

    # Run Part 1 Visualizations on RGB representative pair
    run_intensity_transformations(im_rgb, tag="RGB")
    run_histogram_processing(im_rgb, tag="RGB")
    run_noise_filtering(im_rgb, tag="RGB")
    run_sharpening_filters(im_rgb, tag="RGB")
    run_frequency_domain_analysis(im_rgb, tag="RGB")
    run_edge_detection(im_rgb, tag="RGB")
    run_morphological_processing(im_rgb, tag="RGB")

    # Run Part 1 Visualizations on Grayscale representative pair
    run_intensity_transformations(im_gray, tag="Grayscale")
    run_histogram_processing(im_gray, tag="Grayscale")
    run_noise_filtering(im_gray, tag="Grayscale")
    run_sharpening_filters(im_gray, tag="Grayscale")
    run_frequency_domain_analysis(im_gray, tag="Grayscale")
    run_edge_detection(im_gray, tag="Grayscale")
    run_morphological_processing(im_gray, tag="Grayscale")

    # Forensic ELA analysis on both RGB and Grayscale pairs
    run_forensic_ela_study(rgb_orig, rgb_tamp, tag="RGB")
    run_forensic_ela_study(gray_orig, gray_tamp, tag="Grayscale")

    # Run Part 4 Quantitative Analysis
    df = run_quantitative_analysis(rgb_orig, rgb_tamp)
    print("\nCompleted DIP Study & Visualization Pipeline!")


if __name__ == "__main__":
    main()
