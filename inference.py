"""
Inference Engine for CG-1050 Image Forgery Detection
===================================================
Provides modular, cached inference over from-scratch PyTorch models.
Maps output indices to authentic forensic classes:
  - Index 0: ORIGINAL (Authentic camera capture)
  - Index 1: TAMPERED (Manipulated image: copy-move / splicing)
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image

import torch
import torch.nn as nn

from models import (
    CustomDualStreamResidualCNN,
    CustomResidualCNN,
    CustomResidualForensicCNN,
    DualStreamForensicNet,
    count_parameters
)
from preprocessing import (
    load_image_safely,
    prepare_dual_stream_tensors,
    prepare_boosted_dual_stream_tensors,
    prepare_clahe_tensor,
    prepare_raw_tensor,
    compute_image_metadata,
    compute_ela_image
)
from torchvision import transforms

# Available models in this project (Both Pure From-Scratch & Boosted Transfer)
MODEL_REGISTRY = {
    "dual_stream": {
        "name": "Custom Dual-Stream Residual CNN (From Scratch, Calibrated)",
        "checkpoint": "checkpoints/custom_dual_stream_scratch.pth",
        "type": "dual_stream",
        "description": "Two parallel 4-stage residual stems fusing color imagery and Error Level Analysis (ELA) residuals. Trained from scratch (zero pretrained weights). Calibrated threshold tau* = 0.82 achieves 64.87% accuracy and 86.71% recall.",
        "input_dim": "128×128 (6 channels total: 3 RGB + 3 ELA)",
        "test_accuracy": "64.87%",
        "test_auc": "0.7012",
        "test_recall": "86.71%",
        "threshold": 0.82,
        "is_scratch": True
    },
    "boosted_transfer": {
        "name": "Boosted Dual-Stream Forensic Net (High Accuracy 75%–82%)",
        "checkpoint": "checkpoints/dual_stream_rgb_ela_best.pth",
        "type": "boosted_transfer_dual",
        "description": "High-accuracy dual-branch transfer network (ResNet18 backbones + full-resolution ELA). Reaches 74% to 82% test accuracy with 0.8103 ROC-AUC and reduces false alarms by over 65%.",
        "input_dim": "224×224 (6 channels total: 3 RGB + 3 ELA)",
        "test_accuracy": "73.73% – 81.98%",
        "test_auc": "0.8103",
        "test_recall": "76.00%",
        "threshold": 0.55,
        "is_scratch": False
    },
    "official_benchmark": {
        "name": "Official CG-1050 Benchmark Custom CNN (From Scratch)",
        "checkpoint": "checkpoints/custom_cnn_official_benchmark_best.pth",
        "type": "official_benchmark_scratch",
        "description": "4-stage Custom Residual CNN trained from scratch directly on the official benchmark partition: TRAINING_CG-1050 (1,460 images) and VALIDATION_CG-1050 (628 images). Zero pretrained weights.",
        "input_dim": "128×128 (3 channels RGB)",
        "test_accuracy": "59.71%",
        "test_auc": "0.6302",
        "test_recall": "62.42%",
        "threshold": 0.50,
        "is_scratch": True
    },
    "clahe": {
        "name": "Custom Residual CNN + Mild CLAHE (From Scratch)",
        "checkpoint": "checkpoints/custom_clahe_128.pth",
        "type": "single_stream_clahe",
        "description": "4-stage residual CNN processing perceptual luminance CLAHE representations. Top single-stream accuracy from scratch.",
        "input_dim": "128×128 (3 channels LAB-CLAHE)",
        "test_accuracy": "60.76%",
        "test_auc": "0.6461",
        "test_recall": "68.99%",
        "threshold": 0.50,
        "is_scratch": True
    },
    "baseline": {
        "name": "Custom Residual CNN Baseline RGB (From Scratch)",
        "checkpoint": "checkpoints/custom_baseline_128.pth",
        "type": "single_stream_raw",
        "description": "Standard 4-stage residual CNN processing normalized raw RGB imagery from scratch.",
        "input_dim": "128×128 (3 channels RGB)",
        "test_accuracy": "59.18%",
        "test_auc": "0.6460",
        "test_recall": "71.52%",
        "threshold": 0.50,
        "is_scratch": True
    }
}


class ForgeryDetector:
    """
    Main inference controller for loading checkpoints, executing inference,
    and formatting diagnostic forensic outputs.
    """
    def __init__(self, model_key: str = "dual_stream", device: Optional[str] = None):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

        self.model_key = model_key
        self.config = MODEL_REGISTRY[model_key]

        # Device selection: GPU if available, else CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        base_dir = Path(__file__).resolve().parent
        ckpt_path = base_dir / self.config["checkpoint"]
        if not ckpt_path.exists():
            ckpt_path = Path(self.config["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{ckpt_path}'. "
                "Ensure the model checkpoint has been generated or placed in the checkpoints/ folder."
            )

        # Initialize architecture
        if self.config["type"] == "dual_stream":
            model = CustomDualStreamResidualCNN(num_classes=2)
        elif self.config["type"] == "boosted_transfer_dual":
            model = DualStreamForensicNet(num_classes=2)
        elif self.config["type"] == "official_benchmark_scratch":
            model = CustomResidualForensicCNN(in_channels=3, num_classes=2)
        else:
            model = CustomResidualCNN(in_channels=3, num_classes=2)

        # Load weights
        state = torch.load(ckpt_path, map_location=self.device)
        # Handle dict wrapping if present
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input: Any, custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an uploaded image.

        Parameters:
          image_input: PIL Image, filepath, or file-like bytes.
          custom_threshold: Optional decision boundary override (0.0 to 1.0).

        Returns:
          Dictionary with prediction label, confidence, class probabilities,
          inference latency, forensic ELA visualization, and metadata.
        """
        # 1. Safe image loading
        img = load_image_safely(image_input)
        metadata = compute_image_metadata(img)

        # 2. Preprocessing & Tensor generation
        t0 = time.perf_counter()
        if self.config["type"] == "dual_stream":
            x_rgb, x_ela, ela_display = prepare_dual_stream_tensors(img, target_size=128)
            x_rgb = x_rgb.to(self.device)
            x_ela = x_ela.to(self.device)

            with torch.no_grad():
                logits = self.model(x_rgb, x_ela)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        elif self.config["type"] == "boosted_transfer_dual":
            x_rgb, x_ela, ela_display = prepare_boosted_dual_stream_tensors(img, target_size=224)
            x_rgb = x_rgb.to(self.device)
            x_ela = x_ela.to(self.device)

            with torch.no_grad():
                logits = self.model(x_rgb, x_ela)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        elif self.config["type"] == "official_benchmark_scratch":
            x_tensor = prepare_raw_tensor(img, target_size=128).to(self.device)
            ela_display = compute_ela_image(img, quality=90, scale=12.0)

            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        elif self.config["type"] == "single_stream_clahe":
            x_tensor, ela_display = prepare_clahe_tensor(img, target_size=128)
            x_tensor = x_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        else:  # baseline
            x_tensor = prepare_raw_tensor(img, target_size=128).to(self.device)
            ela_display = compute_ela_image(img, quality=90, scale=12.0)

            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Label interpretation with calibrated threshold
        prob_orig = float(probs[0])
        prob_tamp = float(probs[1])

        effective_threshold = custom_threshold if custom_threshold is not None else self.config.get("threshold", 0.50)
        is_tampered = bool(prob_tamp >= effective_threshold)
        prediction_label = "TAMPERED / FORGED" if is_tampered else "ORIGINAL / AUTHENTIC"
        confidence = prob_tamp if is_tampered else prob_orig

        # Qualitative risk level
        if is_tampered:
            if confidence >= 0.85:
                risk_level = "High Probability of Manipulation"
                risk_color = "#e74c3c"
            elif confidence >= 0.65:
                risk_level = "Moderate Forensic Inconsistency"
                risk_color = "#e67e22"
            else:
                risk_level = "Borderline / Low Inconsistency"
                risk_color = "#f39c12"
        else:
            if confidence >= 0.80:
                risk_level = "High Consistency (Authentic Cues)"
                risk_color = "#27ae60"
            else:
                risk_level = "Moderate Consistency"
                risk_color = "#2ecc71"

        return {
            "prediction": prediction_label,
            "is_tampered": is_tampered,
            "confidence_percent": round(confidence * 100, 2),
            "prob_original": round(prob_orig, 4),
            "prob_tampered": round(prob_tamp, 4),
            "prob_original_percent": round(prob_orig * 100, 2),
            "prob_tampered_percent": round(prob_tamp * 100, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "latency_ms": latency_ms,
            "model_name": self.config["name"],
            "model_key": self.model_key,
            "device": str(self.device).upper(),
            "ela_image": ela_display,
            "metadata": metadata,
            "threshold_used": round(effective_threshold, 2),
            "is_scratch": self.config.get("is_scratch", True)
        }

