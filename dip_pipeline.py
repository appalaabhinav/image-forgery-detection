"""
Digital Image Processing (DIP) Pipeline for CG-1050 Image Forensics
===================================================================
A standalone, modular library implementing classical DIP algorithms using OpenCV,
NumPy, Matplotlib, and Pillow.

Covers all 7 fundamental DIP families:
1. Intensity transformations (Negative, Log, Gamma, Contrast Stretching)
2. Histogram processing (Histogram calculation, Global Equalization, CLAHE)
3. Noise filtering (Mean, Gaussian, Median, Bilateral, Non-Local Means)
4. Sharpening (Laplacian Sharpening, Unsharp Masking, High-Boost Filtering)
5. Frequency-domain analysis (Fourier Magnitude Spectrum, Gaussian LPF, Gaussian HPF, Homomorphic Filtering)
6. Edge detection (Sobel, Prewitt, Laplacian, Laplacian of Gaussian, Canny)
7. Morphological processing (Otsu Thresholding, Opening, Closing, Top-Hat, Black-Hat)

Plus:
- Forensic Error Level Analysis (ELA)
- Quantitative DIP metrics (Mean, Std, Entropy, RMS/Michelson Contrast, PSNR, SSIM, Edge Density, Runtime)
"""

import time
import math
import io
from typing import Tuple, Dict, Any, Union, Optional

import numpy as np
import cv2
from PIL import Image, ImageChops


# ==============================================================================
# Helper Utilities: Image Format & Color Space Handling
# ==============================================================================

def to_numpy_rgb(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Safely loads and returns an image as an RGB uint8 NumPy array.
    Guarantees no in-place mutation of the input.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def to_numpy_gray(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Safely loads and returns an image as a Grayscale uint8 NumPy array.
    """
    rgb = to_numpy_rgb(image_input)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def is_color(arr: np.ndarray) -> bool:
    """Checks whether an array is a 3-channel color image."""
    return arr.ndim == 3 and arr.shape[2] == 3


# ==============================================================================
# 1. Intensity Transformations
# ==============================================================================

def image_negative(img: np.ndarray) -> np.ndarray:
    """
    Image Negative Transformation:
      s = (L - 1) - r = 255 - r

    Purpose:
      Inverts intensity values to reveal details and subtle artifacts embedded
      inside deep shadow regions (e.g. spliced objects placed against dark backgrounds).

    Parameters:
      img: Input RGB or Grayscale image array (uint8).

    Limitations & Forensic Impact:
      Purely mirrors the intensity histogram without altering dynamic range. Inverts
      natural shading cues, which disrupts standard CNN luminance expectations.
    """
    res = 255 - img.copy()
    return res


def logarithmic_transform(img: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """
    Logarithmic Transformation:
      s = c * log(1 + r), where c = 255 / log(1 + max(r))

    Purpose:
      Dynamically expands dark pixel values while compressing brighter pixel values,
      making low-intensity forensic boundary artifacts visible.

    Parameters:
      img: Input RGB or Grayscale image array (uint8).
      c: Scaling constant. If None, automatically calibrated to map [0, 255].

    Limitations & Forensic Impact:
      Heavily compresses high-intensity highlights and alters non-linear camera response
      curves. Destroys delicate JPEG high-frequency quantization structures in bright regions.
    """
    img_f = img.astype(np.float32)
    max_val = np.max(img_f)
    if max_val == 0:
        return img.copy()
    if c is None:
        c = 255.0 / np.log(1.0 + max_val)
    log_transformed = c * np.log(1.0 + img_f)
    return np.clip(log_transformed, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.5, c: float = 1.0) -> np.ndarray:
    """
    Power-Law (Gamma) Transformation:
      s = c * (r / 255.0) ^ gamma * 255.0

    Purpose:
      Adjusts image tonal balance. Gamma < 1 expands dark tones (brightens shadows),
      while Gamma > 1 expands bright tones (darkens mid-tones and highlights contrast).

    Parameters:
      img: Input RGB or Grayscale image array (uint8).
      gamma: Exponent (e.g. 0.5 for shadow brightening, 1.8 for mid-tone darkening).
      c: Gain factor (default 1.0).

    Limitations & Forensic Impact:
      Global gamma manipulation alters authentic Photo Response Non-Uniformity (PRNU)
      sensor noise signatures and obscures camera response function (CRF) inconsistencies
      between spliced host and donor elements.
    """
    inv_gamma = gamma
    table = np.array([
        np.clip(c * ((i / 255.0) ** inv_gamma) * 255.0, 0, 255)
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img.copy(), table)


def contrast_stretching(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Piecewise Linear Contrast Stretching (Percentile-Based):
      s = ((r - r_min) / (r_max - r_min)) * 255

    Purpose:
      Linearly stretches pixel intensities across the full 8-bit dynamic range [0, 255],
      enhancing visual contrast in low-contrast scenes without noise outlier saturation.

    Parameters:
      img: Input RGB or Grayscale image array (uint8).
      p_low: Lower percentile for dynamic clipping (default 2%).
      p_high: Upper percentile for dynamic clipping (default 98%).

    Limitations & Forensic Impact:
      Can clip subtle boundary pixels at extremes and alters genuine camera sensor
      histogram distributions.
    """
    res = img.copy().astype(np.float32)
    if is_color(img):
        # Process in LAB colorspace L-channel to preserve chromaticity
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        r_min, r_max = np.percentile(l_chan, (p_low, p_high))
        if r_max > r_min:
            l_chan = np.clip((l_chan - r_min) / (r_max - r_min) * 255.0, 0, 255.0)
        lab[:, :, 0] = l_chan
        res = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        r_min, r_max = np.percentile(res, (p_low, p_high))
        if r_max > r_min:
            res = np.clip((res - r_min) / (r_max - r_min) * 255.0, 0, 255.0)
        res = res.astype(np.uint8)
    return res


# ==============================================================================
# 2. Histogram Processing
# ==============================================================================

def compute_histogram(img: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes intensity histograms.
    Returns dictionary with channel-wise counts and bin edges.
    """
    hist_data = {}
    if is_color(img):
        for i, color in enumerate(["Red", "Green", "Blue"]):
            h, bins = np.histogram(img[:, :, i].ravel(), bins=256, range=[0, 256])
            hist_data[color] = (h, bins)
    else:
        h, bins = np.histogram(img.ravel(), bins=256, range=[0, 256])
        hist_data["Luminance"] = (h, bins)
    return hist_data


def global_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization (GHE):
      s_k = T(r_k) = (L - 1) * sum_{j=0}^k (n_j / N)

    Purpose:
      Flattens the image histogram to maximize global contrast and entropy.
      For color images, operates strictly on the Y channel (YCrCb) to prevent hue/chroma distortion.

    Parameters:
      img: Input RGB or Grayscale image array (uint8).

    Limitations & Forensic Impact:
      Extremely disruptive for forgery detection! Over-amplifies background sensor grain,
      induces severe false-color artifacts if applied naively to RGB channels, and destroys
      microscopic JPEG block grid boundaries.
    """
    if is_color(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        return cv2.equalizeHist(img.copy())


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE):
      Divides the image into contextual tiles, computes localized histograms,
      clips bins exceeding clip_limit, redistributes clipped pixels uniformly,
      and bilinearly interpolates across tile boundaries.

    Purpose:
      Provides localized, adaptive contrast enhancement while mitigating the excessive
      noise amplification characteristic of global histogram equalization.

    Parameters:
      img: Input RGB or Grayscale image array (uint8).
      clip_limit: Threshold for contrast limiting (typical: 2.0 to 3.0).
      tile_grid_size: Grid size for contextual tile regions (e.g. (8, 8)).

    Forensic Impact:
      Mild CLAHE can reveal faint boundary seams and illumination differences between
      forged and host segments. However, aggressive clip limits can introduce artificial
      micro-edges that trigger false-alarm CNN detections.
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if is_color(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        return clahe_obj.apply(img.copy())


# ==============================================================================
# 3. Noise Filtering
# ==============================================================================

def mean_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Mean (Box) Filter:
      g(x, y) = (1 / K^2) * sum_{i,j in W} f(x+i, y+j)

    Purpose:
      Uniform linear spatial smoothing to reduce random noise.

    Limitations & Forensic Impact:
      Severe loss of high-frequency forensic evidence. Blurs sharp tampering seams,
      erodes JPEG block compression grids, and eliminates camera sensor PRNU noise.
    """
    return cv2.blur(img.copy(), (kernel_size, kernel_size))


def gaussian_filter(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """
    Gaussian Filter:
      G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))

    Purpose:
      Isotropic weighted low-pass filter providing smoother attenuation than box filters.

    Limitations & Forensic Impact:
      Attenuates high-frequency boundary discrepancies and softens spliced object outlines,
      harming forensic classifier sensitivity.
    """
    return cv2.GaussianBlur(img.copy(), (kernel_size, kernel_size), sigma)


def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median Filter:
      g(x, y) = median{ f(x+i, y+j) | (i,j) in W }

    Purpose:
      Non-linear rank-order filter effective at removing salt-and-pepper / impulse noise
      while preserving sharp step edges.

    Limitations & Forensic Impact:
      Median filtering destroys periodic JPEG DCT coefficients and alters natural pixel
      difference correlations, masking forensic traces.
    """
    return cv2.medianBlur(img.copy(), kernel_size)


def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Bilateral Filter:
      I_filtered(p) = (1 / W_p) * sum_{q in Omega} I(q) * g_sigma_s(||p-q||) * g_sigma_r(||I(p)-I(q)||)

    Purpose:
      Edge-preserving smoothing that combines spatial geometric proximity with photometric
      radiometric similarity. Smooths flat regions while preserving distinct edge boundaries.

    Parameters:
      d: Pixel neighborhood diameter.
      sigma_color: Filter sigma in the color/intensity range space.
      sigma_space: Filter sigma in the coordinate spatial space.

    Forensic Impact:
      Preserves prominent tampering seams while suppressing uniform background sensor noise.
      However, subtle, feathered donor seams with intensity differences below sigma_color
      can be smoothed out.
    """
    return cv2.bilateralFilter(img.copy(), d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def non_local_means_filter(img: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """
    Non-Local Means (NLM) Filter:
      Averages pixels weighted by the photometric similarity of entire image patches
      around them across a search window.

    Purpose:
      High-fidelity denoising that preserves repetitive textures and fine structural patterns.

    Parameters:
      h: Filtering parameter regulating decay of weights with patch distance.
      template_window: Size in pixels of the template patch (odd integer, default 7).
      search_window: Size in pixels of the search window (odd integer, default 21).

    Limitations & Forensic Impact:
      Computationally intensive. Tends to over-regularize micro-textures, altering
      natural sensor noise statistics.
    """
    if is_color(img):
        return cv2.fastNlMeansDenoisingColored(img.copy(), None, h=h, hColor=h,
                                               templateWindowSize=template_window,
                                               searchWindowSize=search_window)
    else:
        return cv2.fastNlMeansDenoising(img.copy(), None, h=h,
                                        templateWindowSize=template_window,
                                        searchWindowSize=search_window)


# ==============================================================================
# 4. Sharpening Filters
# ==============================================================================

def laplacian_sharpening(img: np.ndarray, kernel_type: str = "8-neighbor", alpha: float = 1.0) -> np.ndarray:
    """
    Laplacian Sharpening:
      f_sharp = f - alpha * grad^2(f)

    Purpose:
      Enhances fine edges, texture transitions, and microscopic boundaries by subtracting
      the second spatial derivative.

    Parameters:
      kernel_type: '4-neighbor' or '8-neighbor' (including diagonals).
      alpha: Sharpening gain multiplier.

    Limitations & Forensic Impact:
      Extremely sensitive to noise. In noisy images, it severely amplifies background
      sensor grain into speckles, potentially deceiving neural networks.
    """
    if kernel_type == "4-neighbor":
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    else:
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float32)

    img_f = img.astype(np.float32)
    lap = cv2.filter2D(img_f, -1, kernel)
    sharp = img_f + alpha * lap
    return np.clip(sharp, 0, 255).astype(np.uint8)


def unsharp_masking(img: np.ndarray, sigma: float = 1.5, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp Masking:
      f_mask = f - f_smooth
      f_sharp = f + strength * f_mask

    Purpose:
      Subtracts a Gaussian-blurred version from the original image to generate a high-frequency
      detail mask, which is added back to boost edge contrast.

    Parameters:
      sigma: Standard deviation of Gaussian blur.
      strength: Scaling factor for the high-frequency mask.

    Limitations & Forensic Impact:
      Produces halo artifacts along high-contrast boundaries, which can mimic artificial
      splicing halos.
    """
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
    mask = img_f - blurred
    sharp = img_f + strength * mask
    return np.clip(sharp, 0, 255).astype(np.uint8)


def high_boost_filter(img: np.ndarray, A: float = 1.8, sigma: float = 1.5) -> np.ndarray:
    """
    High-Boost Filter:
      f_hb = A * f - f_smooth = (A - 1) * f + f_mask, where A >= 1.0

    Purpose:
      Generalization of unsharp masking where the original image component is boosted by A,
      retaining more base luminance while amplifying high-frequency edge cues.

    Parameters:
      A: High-boost amplification factor (A=1 yields standard unsharp mask).
      sigma: Standard deviation of smoothing filter.

    Limitations & Forensic Impact:
      Can cause dynamic range saturation in bright regions and exacerbate compression ringing.
    """
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
    mask = img_f - blurred
    hb = (A - 1.0) * img_f + mask + img_f
    return np.clip(hb, 0, 255).astype(np.uint8)


# ==============================================================================
# 5. Frequency-Domain Analysis
# ==============================================================================

def fourier_magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """
    2D Fourier Magnitude Spectrum:
      F(u, v) = FFT2D{ f(x, y) }
      Spectrum = log(1 + |F_shifted(u, v)|)

    Purpose:
      Analyzes global frequency distributions, periodicity, and directional energy.
      Centering places the DC component at the center of the spectrum.

    Forensic Impact:
      Exposes periodic artifacts from resampling, image resizing, and repetitive copy-move
      cloning, visible as bright spectral peaks or cross-grid harmonics.
    """
    gray = to_numpy_gray(img)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1.0 + np.abs(fshift))
    # Normalize to [0, 255] for visualization
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def gaussian_low_pass_filter(img: np.ndarray, d0: float = 40.0) -> np.ndarray:
    """
    Gaussian Low-Pass Filter (GLPF) in Frequency Domain:
      H(u, v) = exp( -D^2(u, v) / (2 * D_0^2) )

    Purpose:
      Suppresses high spatial frequencies in the frequency domain, providing smooth blurring
      without the ringing artifacts associated with ideal low-pass filters.

    Parameters:
      d0: Cutoff frequency radius.

    Limitations & Forensic Impact:
      Strips away high-frequency forensic evidence (tampering boundaries, compression residues).
    """
    gray = to_numpy_gray(img)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Compute FFT and shift DC
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Create Gaussian Low Pass Filter mask
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)
    H = np.exp(-(D**2) / (2.0 * (d0**2)))

    # Apply filter and inverse FFT
    filtered = fshift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def gaussian_high_pass_filter(img: np.ndarray, d0: float = 40.0) -> np.ndarray:
    """
    Gaussian High-Pass Filter (GHPF) in Frequency Domain:
      H_hp(u, v) = 1 - H_lp(u, v) = 1 - exp( -D^2(u, v) / (2 * D_0^2) )

    Purpose:
      Attenuates low frequencies (smooth background and lighting) while transmitting
      high frequencies (edges, fine structures, and boundary anomalies).

    Parameters:
      d0: Cutoff frequency radius.

    Forensic Impact:
      Isolates high-frequency boundary seams and edge inconsistencies between forged donor
      elements and host backgrounds.
    """
    gray = to_numpy_gray(img)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)
    H = 1.0 - np.exp(-(D**2) / (2.0 * (d0**2)))

    filtered = fshift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    # Scale for visual display
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def homomorphic_filter(img: np.ndarray, gamma_l: float = 0.5, gamma_h: float = 1.5,
                       d0: float = 30.0, c: float = 1.0) -> np.ndarray:
    """
    Homomorphic Filtering:
      Separates illumination i(x,y) (low-frequency) from reflectance r(x,y) (high-frequency)
      via the logarithmic transform:
        ln(f) = ln(i) + ln(r)
      Applies frequency filter H(u,v) = (gamma_h - gamma_l) * [1 - exp(-c * D^2 / D_0^2)] + gamma_l,
      followed by inverse FFT and exponentiation.

    Purpose:
      Compresses dynamic range caused by uneven lighting while enhancing object reflectance details.

    Parameters:
      gamma_l: Gain for low frequencies (< 1.0, suppresses illumination variation).
      gamma_h: Gain for high frequencies (> 1.0, amplifies reflectance details).
      d0: Cutoff frequency.
      c: Sharpness of slope.

    Forensic Impact:
      Equalizes non-uniform illumination across the scene, highlighting mismatched surface
      reflectance characteristics of spliced donor objects.
    """
    gray = to_numpy_gray(img).astype(np.float32)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Natural logarithm of (1 + image)
    img_log = np.log1p(gray)

    # 2D FFT
    f = np.fft.fft2(img_log)
    fshift = np.fft.fftshift(f)

    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)

    # Filter transfer function
    H = (gamma_h - gamma_l) * (1.0 - np.exp(-c * (D**2) / (d0**2))) + gamma_l

    filtered = fshift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_exp = np.expm1(np.real(img_back))

    # Normalize result
    res = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(res, 0, 255).astype(np.uint8)


# ==============================================================================
# 6. Edge Detection
# ==============================================================================

def sobel_gradient_magnitude(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel Gradient Magnitude:
      G_x = S_x * I,  G_y = S_y * I
      G = sqrt( G_x^2 + G_y^2 )

    Purpose:
      First-order spatial derivative edge detector with directional weighting.
      Detects sharp transitions in intensity.

    Forensic Impact:
      Highlights abrupt gradient boundaries introduced when inserting a tampered segment
      into a host background.
    """
    gray = to_numpy_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def prewitt_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Prewitt Gradient Magnitude:
      Uses unweighted 3x3 directional convolution masks.

    Purpose:
      First-order edge detector estimating orthogonal gradient components.

    Limitations:
      Lacks perpendicular Gaussian smoothing weights; slightly more sensitive to noise than Sobel.
    """
    gray = to_numpy_gray(img).astype(np.float32)
    px_kernel = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    py_kernel = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, px_kernel)
    gy = cv2.filter2D(gray, -1, py_kernel)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def laplacian_edge(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian Operator:
      grad^2(f) = d^2 f / dx^2 + d^2 f / dy^2

    Purpose:
      Second-order isotropic derivative detecting zero-crossings.

    Limitations:
      Extremely sensitive to high-frequency noise and produces double-edge contours.
    """
    gray = to_numpy_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
    return lap_norm.astype(np.uint8)


def laplacian_of_gaussian(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Laplacian of Gaussian (LoG / Marr-Hildreth Operator):
      LoG(x, y) = grad^2 [ G_sigma(x, y) * I(x, y) ]

    Purpose:
      Mitigates Laplacian noise sensitivity by smoothing with a Gaussian filter
      prior to computing second-order derivatives.

    Forensic Impact:
      Robustly identifies boundary contours at controlled spatial scale sigma.
    """
    gray = to_numpy_gray(img)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    log_img = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    log_abs = np.abs(log_img)
    log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX)
    return log_norm.astype(np.uint8)


def canny_edge_detector(img: np.ndarray, low_thresh: float = 50.0, high_thresh: float = 150.0) -> np.ndarray:
    """
    Canny Edge Detector:
      Multi-stage pipeline:
        1. Gaussian smoothing
        2. Sobel gradient magnitude and orientation
        3. Non-Maximum Suppression (NMS) along gradient normal
        4. Hysteresis double thresholding

    Purpose:
      Optimal edge detector providing clean, continuous, single-pixel wide edge contours.

    Limitations & Forensic Impact:
      Generates binary masks that discard all intra-region forensic compression textures.
    """
    gray = to_numpy_gray(img)
    return cv2.Canny(gray, int(low_thresh), int(high_thresh))


# ==============================================================================
# 7. Morphological Processing
# ==============================================================================

def otsu_thresholding(img: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Otsu's Global Thresholding:
      Selects threshold t* maximizing between-class variance:
        sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2

    Purpose:
      Non-parametric, unsupervised binarization based entirely on histogram bimodal variance.

    Returns:
      (optimal_threshold, binary_image)
    """
    gray = to_numpy_gray(img)
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val), binary


def get_structuring_element(kernel_size: int = 5, shape: str = "rect") -> np.ndarray:
    """Constructs morphological structuring element."""
    shape_dict = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE
    }
    cv_shape = shape_dict.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv_shape, (kernel_size, kernel_size))


def morphological_opening(img: np.ndarray, kernel_size: int = 5, shape: str = "rect") -> np.ndarray:
    """
    Morphological Opening:
      A o B = (A (-) B) (+) B  (Erosion followed by Dilation)

    Purpose:
      Eliminates small bright foreground specks, smooths contours, and severs thin connections.

    Forensic Impact:
      Erodes micro-traces and single-pixel splicing artifacts.
    """
    gray = to_numpy_gray(img)
    se = get_structuring_element(kernel_size, shape)
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)


def morphological_closing(img: np.ndarray, kernel_size: int = 5, shape: str = "rect") -> np.ndarray:
    """
    Morphological Closing:
      A . B = (A (+) B) (-) B  (Dilation followed by Erosion)

    Purpose:
      Fills small dark voids, bridges narrow breaks in contours, and merges nearby elements.

    Forensic Impact:
      Can bridge micro-breaks in tampering seams, artificially hiding splicing boundaries.
    """
    gray = to_numpy_gray(img)
    se = get_structuring_element(kernel_size, shape)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)


def morphological_tophat(img: np.ndarray, kernel_size: int = 9, shape: str = "rect") -> np.ndarray:
    """
    Morphological Top-Hat (White Top-Hat):
      T_hat(f) = f - (f o B)

    Purpose:
      Isolates bright details that are smaller than the structuring element on darker backgrounds.

    Forensic Impact:
      Highlights microscopic bright tampering anomalies and specular gradient peaks.
    """
    gray = to_numpy_gray(img)
    se = get_structuring_element(kernel_size, shape)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def morphological_blackhat(img: np.ndarray, kernel_size: int = 9, shape: str = "rect") -> np.ndarray:
    """
    Morphological Black-Hat:
      B_hat(f) = (f . B) - f

    Purpose:
      Isolates dark details that are smaller than the structuring element on brighter backgrounds.

    Forensic Impact:
      Highlights dark boundary crevices, shadow irregularities, and compression lines.
    """
    gray = to_numpy_gray(img)
    se = get_structuring_element(kernel_size, shape)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)


# ==============================================================================
# 8. Forensic Error Level Analysis (ELA)
# ==============================================================================

def compute_ela(image_input: Union[str, Image.Image, np.ndarray],
                quality: int = 90, scale: float = 12.0) -> np.ndarray:
    """
    Error Level Analysis (ELA):
      Saves the image at a known JPEG compression quality factor (e.g. 90%),
      computes absolute pixel-level difference against the original, and scales
      the residual for visual and numerical inspection:
        ELA = clip( |I - I_recomp| * scale, 0, 255 )

    Purpose:
      Exposes compression error discrepancies. In an authentic image, compression error
      is uniform across similar textures. In a tampered image, spliced or modified regions
      often exhibit markedly higher or lower error levels compared to untouched regions.

    Returns:
      uint8 NumPy array (RGB) representing the ELA map.
    """
    if isinstance(image_input, (str, Image.Image)):
        im = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input.convert("RGB")
    else:
        im = Image.fromarray(to_numpy_rgb(image_input))

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    diff = ImageChops.difference(im, recompressed)
    diff_np = np.array(diff, dtype=np.float32) * scale
    diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
    return diff_np


# ==============================================================================
# 9. Quantitative DIP Analysis Metrics
# ==============================================================================

def compute_intensity_stats(img: np.ndarray) -> Dict[str, float]:
    """Computes mean, standard deviation, min, max, and dynamic range."""
    gray = to_numpy_gray(img).astype(np.float64)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    return {
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "min_intensity": min_val,
        "max_intensity": max_val,
        "dynamic_range": max_val - min_val
    }


def compute_shannon_entropy(img: np.ndarray) -> float:
    """
    Shannon Entropy:
      H = - sum_{k=0}^{255} p(k) * log2( p(k) )

    Measures average information content in bits per pixel.
    Forensic caveat: High entropy does not equate to authenticity!
    """
    gray = to_numpy_gray(img)
    counts = np.bincount(gray.ravel(), minlength=256)
    p = counts / counts.sum()
    p_nonzero = p[p > 0]
    entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
    return entropy


def compute_contrast_metrics(img: np.ndarray) -> Dict[str, float]:
    """
    Computes RMS Contrast and Michelson Contrast:
      RMS Contrast = std(I) / 255.0
      Michelson Contrast = (I_max - I_min) / (I_max + I_min)
    """
    gray = to_numpy_gray(img).astype(np.float64)
    rms = float(np.std(gray) / 255.0)
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    denom = max_val + min_val
    michelson = float((max_val - min_val) / denom) if denom > 0 else 0.0
    return {
        "rms_contrast": rms,
        "michelson_contrast": michelson
    }


def compute_psnr_ssim(ref_img: np.ndarray, target_img: np.ndarray) -> Dict[str, float]:
    """
    Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM).
    Requires matching spatial dimensions.
    """
    ref_g = to_numpy_gray(ref_img).astype(np.float64)
    tar_g = to_numpy_gray(target_img).astype(np.float64)

    if ref_g.shape != tar_g.shape:
        tar_g = cv2.resize(tar_g, (ref_g.shape[1], ref_g.shape[0]))

    mse = float(np.mean((ref_g - tar_g) ** 2))
    if mse == 0:
        psnr = 100.0  # Identical images
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))

    # Simplified SSIM calculation (Wang et al.)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(np.mean(ref_g))
    mu_y = float(np.mean(tar_g))
    sigma_x_sq = float(np.var(ref_g))
    sigma_y_sq = float(np.var(tar_g))
    sigma_xy = float(np.mean((ref_g - mu_x) * (tar_g - mu_y)))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim = float(numerator / denominator)

    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim
    }


def compute_edge_metrics(img: np.ndarray) -> Dict[str, float]:
    """
    Computes Edge Density and Mean Gradient Magnitude:
      Edge Density = (Count of Canny edge pixels) / (Total pixels)
      Mean Gradient = Average Sobel magnitude
    """
    gray = to_numpy_gray(img)
    canny = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(canny) / canny.size)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient = float(np.mean(np.sqrt(gx**2 + gy**2)))

    return {
        "edge_density": edge_density,
        "mean_gradient_magnitude": mean_gradient
    }


def benchmark_runtime(func, *args, n_repeats: int = 5, **kwargs) -> float:
    """Measures average execution runtime in milliseconds over n_repeats."""
    # Warmup
    _ = func(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = func(*args, **kwargs)
    t1 = time.perf_counter()
    return float((t1 - t0) / n_repeats * 1000.0)
