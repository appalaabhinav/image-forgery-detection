"""
Comprehensive Evaluation of All Scratch Custom CNN Checkpoints on CG-1050
"""
import os
import sys
import time
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import cv2

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from run_ablation_study import (
    ResidualBlock, CustomResidualCNN, create_source_grouped_splits,
    preload_images_to_memory, build_representation_tensors
)
from train_custom_dual_stream_scratch import CustomDualStreamResidualCNN, compute_ela_array, prepare_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df, val_df, test_df = create_source_grouped_splits()

print(f"Validation images: {len(val_df)} | Test images: {len(test_df)}")

# Evaluate single-stream scratch models on validation
single_models = [
    ("Exp 1: Baseline RGB (128x128)", "checkpoints/custom_baseline_128.pth", 128, "raw", 3),
    ("Exp 2: Mild CLAHE (128x128)", "checkpoints/custom_clahe_128.pth", 128, "clahe", 3),
    ("Exp 3: Edge-Preserving Bilateral (128x128)", "checkpoints/custom_bilateral_128.pth", 128, "bilateral", 3),
    ("Exp 4: Sobel Gradient Fused 4-Ch (128x128)", "checkpoints/custom_sobel_128.pth", 128, "sobel_rgb", 4),
    ("Exp 5: Forensic ELA Residuals (128x128)", "checkpoints/custom_ela_128.pth", 128, "ela", 3),
    ("Exp 6: Higher Res RGB (160x160)", "checkpoints/custom_res160_raw.pth", 160, "raw", 3),
]

val_results = []

for name, ckpt, res, prep, in_ch in single_models:
    if not os.path.exists(ckpt):
        print(f"Skipping {name}: Checkpoint not found")
        continue
    img_arr, labels, paths = preload_images_to_memory(val_df, res)
    tensors = build_representation_tensors(img_arr, prep, paths)
    val_loader = DataLoader(TensorDataset(tensors, torch.from_numpy(labels)), batch_size=32, shuffle=False)

    model = CustomResidualCNN(in_channels=in_ch, num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    probs, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
            targets.extend(y.numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    auc = roc_auc_score(targets, probs)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    val_results.append({
        "Model": name,
        "Val_Acc": acc * 100,
        "Val_Bal_Acc": bal_acc * 100,
        "Val_Prec": prec * 100,
        "Val_Rec": rec * 100,
        "Val_F1": f1 * 100,
        "Val_AUC": auc,
        "FP": fp,
        "FN": fn,
        "Type": "Single-Stream Scratch"
    })

# Evaluate Dual-Stream Scratch model if checkpoint exists
if os.path.exists("checkpoints/custom_dual_stream_scratch.pth"):
    x_va_r, x_va_e, y_va = prepare_tensors(val_df, 128)
    val_loader = DataLoader(TensorDataset(x_va_r, x_va_e, y_va), batch_size=32, shuffle=False)
    dual_model = CustomDualStreamResidualCNN(num_classes=2).to(device)
    dual_model.load_state_dict(torch.load("checkpoints/custom_dual_stream_scratch.pth", map_location=device))
    dual_model.eval()

    probs, targets = [], []
    with torch.no_grad():
        for xr, xe, y in val_loader:
            xr, xe = xr.to(device), xe.to(device)
            out = dual_model(xr, xe)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
            targets.extend(y.numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    auc = roc_auc_score(targets, probs)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    val_results.append({
        "Model": "Exp 8: Custom Dual-Stream (RGB+ELA) Scratch",
        "Val_Acc": acc * 100,
        "Val_Bal_Acc": bal_acc * 100,
        "Val_Prec": prec * 100,
        "Val_Rec": rec * 100,
        "Val_F1": f1 * 100,
        "Val_AUC": auc,
        "FP": fp,
        "FN": fn,
        "Type": "Dual-Stream Scratch"
    })

df_res = pd.DataFrame(val_results)
print("\n" + "="*80)
print("SCRATCH MODELS VALIDATION COMPARISON (STRICT 0-SCENE LEAKAGE)")
print("="*80)
print(df_res[["Model", "Val_Acc", "Val_Bal_Acc", "Val_F1", "Val_Prec", "Val_Rec", "Val_AUC", "FP", "FN"]].to_string(index=False))

# Identify best model strictly by Validation F1 / Balanced Accuracy
best_row = df_res.sort_values(by=["Val_F1", "Val_Bal_Acc"], ascending=False).iloc[0]
print("\n" + "*"*80)
print(f"BEST MODEL SELECTED VIA VALIDATION: {best_row['Model']}")
print(f"Validation F1: {best_row['Val_F1']:.2f}% | Val Acc: {best_row['Val_Acc']:.2f}% | Val Bal Acc: {best_row['Val_Bal_Acc']:.2f}%")
print("*"*80)
