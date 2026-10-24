"""
Predict Image Forgery (Original vs. Tampered)
=============================================
High-accuracy inference module and CLI tool for predicting whether an image
is ORIGINAL or TAMPERED using the fine-tuned transfer learning model and
Error Level Analysis (ELA) forensic verification.
"""

import os
import sys
import io
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageChops

import torch
import torch.nn as nn
from torchvision import transforms, models


def compute_ela(image_path: str, quality: int = 90, scale: float = 12.0) -> Tuple[Image.Image, float, float]:
    """
    Computes Error Level Analysis (ELA) to visualize compression / tampering artifacts.
    Also returns mean error and max error discrepancy metrics.
    """
    orig = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    orig.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    
    diff = ImageChops.difference(orig, recompressed)
    diff_np = np.array(diff, dtype=np.float32) * scale
    diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
    
    mean_err = float(np.mean(diff_np))
    max_err = float(np.max(diff_np))
    return Image.fromarray(diff_np), mean_err, max_err


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, int, Dict[str, int]]:
    """Loads a trained model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_architecture" in checkpoint:
        arch = checkpoint["model_architecture"].lower()
        image_size = checkpoint.get("image_size", 224)
        class_to_idx = checkpoint.get("class_to_idx", {"ORIGINAL": 0, "TAMPERED": 1})
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        arch = "resnet18"
        image_size = 224
        class_to_idx = {"ORIGINAL": 0, "TAMPERED": 1}
        state_dict = checkpoint["model_state_dict"]
    else:
        arch = "resnet18"
        image_size = 224
        class_to_idx = {"ORIGINAL": 0, "TAMPERED": 1}
        state_dict = checkpoint

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, len(class_to_idx))
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, len(class_to_idx))
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, image_size, class_to_idx


def predict_image(
    image_path: str,
    checkpoint_path: str = "cg1050_improved_best.pth",
    device: torch.device = None,
    save_ela: bool = False
) -> Dict[str, Any]:
    """
    Predicts whether a single image is ORIGINAL or TAMPERED.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, image_size, class_to_idx = load_model(checkpoint_path, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_img = Image.open(image_path).convert("RGB")
    tensor_img = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    p_orig = float(probs[class_to_idx["ORIGINAL"]])
    p_tamp = float(probs[class_to_idx["TAMPERED"]])
    
    pred_label = "ORIGINAL" if p_orig >= p_tamp else "TAMPERED"
    confidence = max(p_orig, p_tamp) * 100.0

    ela_img, mean_err, max_err = compute_ela(image_path)
    ela_saved_path = None
    if save_ela:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        ela_saved_path = str(out_dir / f"ela_{Path(image_path).stem}.png")
        ela_img.save(ela_saved_path)

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": {
            "ORIGINAL": round(p_orig, 4),
            "TAMPERED": round(p_tamp, 4)
        },
        "forensic_metrics": {
            "mean_compression_error": round(mean_err, 2),
            "max_compression_error": round(max_err, 2)
        },
        "image_path": str(image_path),
        "ela_path": ela_saved_path
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Image Forgery (Original vs Tampered)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--checkpoint", type=str, default="cg1050_improved_best.pth", help="Model checkpoint path")
    parser.add_argument("--save_ela", action="store_true", help="Save Error Level Analysis visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 55)
    print("CG-1050 IMAGE FORGERY DETECTION")
    print("=" * 55)
    print(f"Target Image:     {args.image}")
    print(f"Model Checkpoint: {args.checkpoint}")
    print(f"Device:           {device}")

    try:
        res = predict_image(args.image, args.checkpoint, device, save_ela=args.save_ela)
        print("\n" + "-" * 55)
        print(f"VERDICT:    {res['prediction']}")
        print(f"CONFIDENCE: {res['confidence']:.2f}%")
        print(f"PROBABILITIES: Original={res['probabilities']['ORIGINAL']:.2%} | Tampered={res['probabilities']['TAMPERED']:.2%}")
        print(f"FORENSIC ELA:  Mean Error={res['forensic_metrics']['mean_compression_error']} | Max Error={res['forensic_metrics']['max_compression_error']}")
        if res["ela_path"]:
            print(f"SAVED ELA MAP: {res['ela_path']}")
        print("-" * 55)
    except Exception as e:
        print(f"Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
