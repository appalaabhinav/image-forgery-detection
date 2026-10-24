"""
Preprocessing and Digital Image Processing Utilities for Forgery Inference
==========================================================================
Replicates the exact preprocessing pipelines used during training:
- Resizing: Bilinear interpolation to 128x128
- Normalization: Floating point scaled to [-1.0, 1.0] via (x / 127.5) - 1.0
- ELA: In-memory JPEG recompression at Quality=90 with 12.0x linear amplification
- CLAHE: Applied strictly to perceptual Luminance (L) channel in LAB color space
"""

import io
import math
from typing import Tuple, Dict, Any, Union
import numpy as np
import cv2
from PIL import Image, ImageChops
import torch


def load_image_safely(image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """
    Safely opens and converts an arbitrary image input to an RGB PIL Image.
    Handles corruptions, palette modes, alpha channels, and grayscale inputs.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, io.BytesIO)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB representation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_ela_image(image: Image.Image, quality: int = 90, scale: float = 12.0) -> Image.Image:
    """
    Calculates Error Level Analysis (ELA) map by measuring JPEG recompression difference.

    Parameters:
      image: RGB PIL Image
      quality: JPEG compression quality factor (default: 90)
      scale: Linear contrast scaling factor (default: 12.0)

    Returns:
      PIL Image: Enhanced visual representation of compression artifacts.
    """
    buf = io.BytesIO()
    # Save image to in-memory buffer at specified JPEG quality
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    # Compute absolute difference between original and recompressed
    diff = ImageChops.difference(image, recompressed)

    # Amplify difference residuals
    diff_arr = np.array(diff, dtype=np.float32) * scale
    diff_arr = np.clip(diff_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(diff_arr)


def prepare_dual_stream_tensors(
    image: Image.Image, target_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """
    Preprocesses an input image into paired PyTorch tensors for the Dual-Stream network:
      1. RGB Tensor: Shape (1, 3, 128, 128), normalized to [-1.0, 1.0]
      2. ELA Tensor: Shape (1, 3, 128, 128), normalized to [-1.0, 1.0]

    Returns:
      (x_rgb, x_ela, ela_display_image)
    """
    # 1. Resize original image to target resolution (128x128)
    im_res = image.resize((target_size, target_size), Image.BILINEAR)

    # 2. Compute ELA at 128x128 for tensor branch
    ela_res = compute_ela_image(im_res, quality=90, scale=12.0)

    # 3. Also compute full-resolution ELA for side-by-side user display
    ela_display = compute_ela_image(image, quality=90, scale=12.0)

    # 4. Standardize to [-1.0, 1.0] float32 arrays
    rgb_norm = (np.array(im_res, dtype=np.float32) / 127.5) - 1.0
    ela_norm = (np.array(ela_res, dtype=np.float32) / 127.5) - 1.0

    # 5. Convert to PyTorch (1, C, H, W) tensors
    x_rgb = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float()
    x_ela = torch.from_numpy(ela_norm).permute(2, 0, 1).unsqueeze(0).float()

    return x_rgb, x_ela, ela_display


def prepare_boosted_dual_stream_tensors(
    image: Image.Image, target_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """
    Preprocesses an input image for the Boosted Transfer Dual-Stream network (ResNet18):
      - Resolution: 224x224
      - RGB: Mild CLAHE on L channel + ImageNet normalization
      - ELA: Full-res ELA (Q=90, scale=12.0) resized to 224x224 + ImageNet normalization
    """
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    ela_full = compute_ela_image(image, quality=90, scale=12.0)
    im_res = image.resize((target_size, target_size), Image.BILINEAR)
    ela_res = ela_full.resize((target_size, target_size), Image.BILINEAR)

    arr_rgb = np.array(im_res, dtype=np.uint8)
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    rgb_norm = (rgb_norm - mean_rgb) / std_rgb

    arr_ela = np.array(ela_res, dtype=np.float32) / 255.0
    ela_norm = (arr_ela - mean_rgb) / std_rgb

    x_rgb = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float()
    x_ela = torch.from_numpy(ela_norm).permute(2, 0, 1).unsqueeze(0).float()
    return x_rgb, x_ela, ela_full



def prepare_clahe_tensor(
    image: Image.Image, target_size: int = 128
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocesses image using Perceptual Luminance CLAHE in LAB space.
    Used for single-stream CLAHE model inference.
    """
    im_res = image.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(im_res, dtype=np.uint8)

    # Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    clahe_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    clahe_display = Image.fromarray(clahe_rgb)

    # Normalize to [-1.0, 1.0]
    out_norm = (clahe_rgb.astype(np.float32) / 127.5) - 1.0
    x_tensor = torch.from_numpy(out_norm).permute(2, 0, 1).unsqueeze(0).float()

    return x_tensor, clahe_display


def prepare_raw_tensor(
    image: Image.Image, target_size: int = 128
) -> torch.Tensor:
    """Preprocesses raw RGB image scaled to [-1.0, 1.0]."""
    im_res = image.resize((target_size, target_size), Image.BILINEAR)
    out_norm = (np.array(im_res, dtype=np.float32) / 127.5) - 1.0
    return torch.from_numpy(out_norm).permute(2, 0, 1).unsqueeze(0).float()


def compute_image_metadata(image: Image.Image, raw_bytes: bytes = None) -> Dict[str, Any]:
    """Extracts forensic metadata and statistical properties of an uploaded image."""
    w, h = image.size
    aspect = round(w / h, 2) if h > 0 else 0

    # Convert to grayscale numpy for luminance statistics
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean_lum = round(float(np.mean(gray)), 2)
    std_lum = round(float(np.std(gray)), 2)

    # Shannon Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = round(float(-np.sum(hist * np.log2(hist))), 3) if len(hist) > 0 else 0.0

    size_kb = round(len(raw_bytes) / 1024, 1) if raw_bytes else None

    return {
        "width": w,
        "height": h,
        "dimensions": f"{w} × {h}",
        "aspect_ratio": f"{aspect}:1",
        "format": image.format or "JPEG/PNG",
        "mode": image.mode,
        "size_kb": f"{size_kb} KB" if size_kb else "N/A",
        "mean_luminance": mean_lum,
        "std_luminance": std_lum,
        "entropy": f"{entropy} bits"
    }
