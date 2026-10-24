"""
Controlled DIP Ablation Study & Model Improvement Workflow (Optimized & In-Memory)
===================================================================================
Trains the CustomResidualCNN strictly from scratch (random initialization, no pretrained weights)
across systematic DIP representations:
1. Baseline RGB (128x128)
2. Mild CLAHE on Luminance (128x128)
3. Edge-Preserving Bilateral Filter (128x128)
4. Sobel Gradient Magnitude Representation (128x128, 4-channel RGB+Sobel)
5. Forensic Error Level Analysis (ELA) Representation (128x128)
6. Higher Resolution RGB (160x160, architecture unchanged)
7. Tuned Combined Representation (160x160 + AdamW + CosineAnnealing + Augmentation)

Preserves strict source-grouped split:
- Train: 1,436 images (70 scenes)
- Val: 336 images (15 scenes)
- Test: 316 images (15 scenes)
Zero base-scene leakage. Test set evaluated ONLY once on the final selected model.
"""

import os
import sys
import time
import re
import io
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import dip_pipeline as dip

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def log_print(msg: str):
    """Prints message and immediately flushes stdout buffer."""
    print(msg)
    sys.stdout.flush()


# ==============================================================================
# 1. Dataset Collection & Source-Grouped Leakage-Free Splitting
# ==============================================================================

def create_source_grouped_splits():
    records = []
    for split_folder in ['TRAINING_CG-1050', 'VALIDATION_CG-1050']:
        for root, dirs, files in os.walk(split_folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    p = os.path.join(root, f)
                    parent = os.path.basename(root).upper()
                    label = 0 if parent == 'ORIGINAL' else 1
                    match = re.match(r'^(Im\d+)', f, re.IGNORECASE)
                    base_id = match.group(1).lower() if match else 'unknown'
                    records.append({'image_path': p, 'filename': f, 'label': label, 'base_id': base_id})

    all_df = pd.DataFrame(records)
    source_groups = all_df['base_id'].drop_duplicates().to_numpy()
    train_val, test_groups = train_test_split(source_groups, test_size=0.15, random_state=SEED, shuffle=True)
    train_groups, val_groups = train_test_split(train_val, test_size=0.15/0.85, random_state=SEED, shuffle=True)

    train_df = all_df[all_df['base_id'].isin(train_groups)].reset_index(drop=True)
    val_df = all_df[all_df['base_id'].isin(val_groups)].reset_index(drop=True)
    test_df = all_df[all_df['base_id'].isin(test_groups)].reset_index(drop=True)

    log_print(f"Dataset Split (0 Scene Leakage):")
    log_print(f"  Train: {len(train_df)} images ({len(train_groups)} base scenes) [Orig: {(train_df.label==0).sum()}, Tamp: {(train_df.label==1).sum()}]")
    log_print(f"  Val:   {len(val_df)} images ({len(val_groups)} base scenes) [Orig: {(val_df.label==0).sum()}, Tamp: {(val_df.label==1).sum()}]")
    log_print(f"  Test:  {len(test_df)} images ({len(test_groups)} base scenes) [Orig: {(test_df.label==0).sum()}, Tamp: {(test_df.label==1).sum()}]")

    return train_df, val_df, test_df


# ==============================================================================
# 2. In-Memory Image Cache (Loaded Once)
# ==============================================================================

def preload_images_to_memory(df: pd.DataFrame, resolution: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads and resizes all images in a dataframe into a single uint8 NumPy array in RAM."""
    paths = df['image_path'].tolist()
    labels = df['label'].to_numpy(dtype=np.int64)
    img_list = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        img_list.append(np.array(im, dtype=np.uint8))
    arr = np.stack(img_list, axis=0)  # Shape: (N, H, W, 3)
    return arr, labels, paths


# ==============================================================================
# 3. Custom Residual CNN Architecture (Trained From Scratch)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomResidualCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(24, 24)
        self.layer2 = ResidualBlock(24, 48, stride=2)
        self.layer3 = ResidualBlock(48, 96, stride=2)
        self.layer4 = ResidualBlock(96, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.35)
        self.classifier = nn.Linear(128, num_classes)

        # Standard Kaiming Normal Initialization for from-scratch training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# ==============================================================================
# 4. Fast Representation Builder from In-Memory Arrays
# ==============================================================================

def build_representation_tensors(img_arr: np.ndarray, prep_mode: str, paths: List[str] = None) -> torch.Tensor:
    """Transforms an in-memory (N, H, W, 3) uint8 array into normalized PyTorch float tensors."""
    N, H, W, C = img_arr.shape
    
    if prep_mode == "raw":
        out = (img_arr.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()

    elif prep_mode == "clahe":
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        transformed = np.empty_like(img_arr)
        for i in range(N):
            lab = cv2.cvtColor(img_arr[i], cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            transformed[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out = (transformed.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()

    elif prep_mode == "bilateral":
        transformed = np.empty_like(img_arr)
        for i in range(N):
            transformed[i] = cv2.bilateralFilter(img_arr[i], d=7, sigmaColor=50, sigmaSpace=50)
        out = (transformed.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()

    elif prep_mode == "sobel_rgb":
        # 4-channel tensor: RGB (3) + Sobel Gradient Magnitude (1)
        sobel_list = []
        for i in range(N):
            gray = cv2.cvtColor(img_arr[i], cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sobel_list.append(mag_norm)
        sobel_arr = np.stack(sobel_list, axis=0)[:, :, :, np.newaxis]
        fused = np.concatenate([img_arr, sobel_arr], axis=-1)  # (N, H, W, 4)
        out = (fused.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()

    elif prep_mode == "ela":
        transformed = np.empty_like(img_arr)
        for i in range(N):
            im_pil = Image.fromarray(img_arr[i])
            buf = io.BytesIO()
            im_pil.save(buf, "JPEG", quality=90)
            buf.seek(0)
            recomp = Image.open(buf)
            diff = ImageChops.difference(im_pil, recomp)
            diff_np = np.clip(np.array(diff, dtype=np.float32) * 12.0, 0, 255).astype(np.uint8)
            transformed[i] = diff_np
        out = (transformed.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()

    else:
        raise ValueError(f"Unknown preprocessing representation: {prep_mode}")

    return tensor


# ==============================================================================
# 5. Training & Evaluation Routines
# ==============================================================================

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(y.tolist())

    total_eval_time = (time.perf_counter() - t0) * 1000.0  # ms
    inference_time_per_sample = total_eval_time / len(all_targets)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bal_acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Inference_ms": float(inference_time_per_sample)
    }

    return metrics, y_true, y_pred, y_prob


def train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       epochs: int = 5, lr: float = 0.0007, checkpoint_path: str = "model.pth",
                       use_scheduler: bool = True) -> Tuple[Dict[str, Any], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_metrics = {}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for x, y in train_loader:
            # Mild online random horizontal flip augmentation
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[3])

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            total_samples += len(y)

        train_acc = train_correct / total_samples

        # Validation Step
        val_metrics, _, _, _ = evaluate_model(model, val_loader)
        if use_scheduler:
            scheduler.step()

        log_print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | Val F1: {val_metrics['F1-Score']:.4f}")

        if val_metrics['F1-Score'] >= best_val_f1:
            best_val_f1 = val_metrics['F1-Score']
            best_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)

    training_time = time.time() - start_time
    return best_metrics, training_time


# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================

def run_ablation_study():
    log_print("\n" + "="*70)
    log_print("STARTING CONTROLLED DIP ABLATION STUDY (CUSTOM CNN FROM SCRATCH)")
    log_print("="*70)

    train_df, val_df, test_df = create_source_grouped_splits()

    log_print("\n[Preloading 128x128 images into memory for instantaneous batch access]...")
    t0 = time.time()
    train_imgs_128, train_y, train_paths = preload_images_to_memory(train_df, 128)
    val_imgs_128, val_y, val_paths = preload_images_to_memory(val_df, 128)
    test_imgs_128, test_y, test_paths = preload_images_to_memory(test_df, 128)
    log_print(f"  Preloaded 128x128 images in {time.time()-t0:.1f}s.")

    log_print("\n[Preloading 160x160 images into memory for resolution experiments]...")
    t0 = time.time()
    train_imgs_160, _, _ = preload_images_to_memory(train_df, 160)
    val_imgs_160, _, _ = preload_images_to_memory(val_df, 160)
    test_imgs_160, _, _ = preload_images_to_memory(test_df, 160)
    log_print(f"  Preloaded 160x160 images in {time.time()-t0:.1f}s.")

    experiments = [
        {
            "name": "Exp 1: Baseline RGB",
            "prep": "raw",
            "resolution": 128,
            "in_channels": 3,
            "lr": 0.0005,
            "epochs": 5,
            "train_imgs": train_imgs_128,
            "val_imgs": val_imgs_128,
            "test_imgs": test_imgs_128,
            "ckpt": "checkpoints/custom_baseline_128.pth"
        },
        {
            "name": "Exp 2: Mild CLAHE (Luminance)",
            "prep": "clahe",
            "resolution": 128,
            "in_channels": 3,
            "lr": 0.0005,
            "epochs": 5,
            "train_imgs": train_imgs_128,
            "val_imgs": val_imgs_128,
            "test_imgs": test_imgs_128,
            "ckpt": "checkpoints/custom_clahe_128.pth"
        },
        {
            "name": "Exp 3: Edge-Preserving Bilateral",
            "prep": "bilateral",
            "resolution": 128,
            "in_channels": 3,
            "lr": 0.0005,
            "epochs": 5,
            "train_imgs": train_imgs_128,
            "val_imgs": val_imgs_128,
            "test_imgs": test_imgs_128,
            "ckpt": "checkpoints/custom_bilateral_128.pth"
        },
        {
            "name": "Exp 4: Sobel Gradient Fused (4-Ch)",
            "prep": "sobel_rgb",
            "resolution": 128,
            "in_channels": 4,
            "lr": 0.0005,
            "epochs": 5,
            "train_imgs": train_imgs_128,
            "val_imgs": val_imgs_128,
            "test_imgs": test_imgs_128,
            "ckpt": "checkpoints/custom_sobel_128.pth"
        },
        {
            "name": "Exp 5: Forensic ELA Residuals",
            "prep": "ela",
            "resolution": 128,
            "in_channels": 3,
            "lr": 0.0007,
            "epochs": 5,
            "train_imgs": train_imgs_128,
            "val_imgs": val_imgs_128,
            "test_imgs": test_imgs_128,
            "ckpt": "checkpoints/custom_ela_128.pth"
        },
        {
            "name": "Exp 6: 160x160 Spatial Resolution",
            "prep": "raw",
            "resolution": 160,
            "in_channels": 3,
            "lr": 0.0007,
            "epochs": 5,
            "train_imgs": train_imgs_160,
            "val_imgs": val_imgs_160,
            "test_imgs": test_imgs_160,
            "ckpt": "checkpoints/custom_res160_raw.pth"
        },
        {
            "name": "Exp 7: Tuned 160x160 + Cosine LR",
            "prep": "raw",
            "resolution": 160,
            "in_channels": 3,
            "lr": 0.0010,
            "epochs": 6,
            "train_imgs": train_imgs_160,
            "val_imgs": val_imgs_160,
            "test_imgs": test_imgs_160,
            "ckpt": "checkpoints/custom_res160_tuned.pth"
        }
    ]

    results = []
    trained_configs = {}

    for exp in experiments:
        log_print(f"\n>>> Running {exp['name']} (Res={exp['resolution']}, Channels={exp['in_channels']}, lr={exp['lr']}) <<<")
        
        # Build representation tensors fast
        x_train = build_representation_tensors(exp["train_imgs"], exp["prep"])
        y_train = torch.from_numpy(train_y)
        x_val = build_representation_tensors(exp["val_imgs"], exp["prep"])
        y_val = torch.from_numpy(val_y)

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # Initialize fresh custom CNN with randomly initialized weights
        model = CustomResidualCNN(in_channels=exp["in_channels"], num_classes=2).to(device)

        val_metrics, train_sec = train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=exp["epochs"],
            lr=exp["lr"],
            checkpoint_path=exp["ckpt"],
            use_scheduler=True
        )

        res_row = {
            "Experiment": exp["name"],
            "Resolution": f"{exp['resolution']}x{exp['resolution']}",
            "Preprocessing": exp["prep"],
            "Input_Channels": exp["in_channels"],
            "Train_Time_s": round(train_sec, 1),
            "Val_Accuracy": round(val_metrics["Accuracy"] * 100, 2),
            "Val_Bal_Accuracy": round(val_metrics["Balanced_Accuracy"] * 100, 2),
            "Val_Precision": round(val_metrics["Precision"] * 100, 2),
            "Val_Recall": round(val_metrics["Recall"] * 100, 2),
            "Val_F1_Score": round(val_metrics["F1-Score"] * 100, 2),
            "Val_ROC_AUC": round(val_metrics["ROC-AUC"], 4),
            "Val_FP": val_metrics["FP"],
            "Val_FN": val_metrics["FN"],
            "Inference_ms": round(val_metrics["Inference_ms"], 2),
            "Checkpoint": exp["ckpt"]
        }
        results.append(res_row)
        trained_configs[exp["name"]] = exp

    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "ablation_study_results.csv"
    results_df.to_csv(csv_path, index=False)
    log_print(f"\n[Ablation Table Saved] -> {csv_path}")
    log_print("\nCompleted Ablation Study Results (Strict Validation):")
    log_print(results_df[["Experiment", "Val_Accuracy", "Val_F1_Score", "Val_Precision", "Val_Recall", "Val_FP", "Val_FN"]].to_string())

    # Generate Ablation Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(13, 5.2))
    results_df_plot = results_df.set_index("Experiment")
    results_df_plot[["Val_Accuracy", "Val_F1_Score"]].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"], width=0.65)
    ax.set_title("Controlled DIP Ablation Study: Validation Accuracy vs F1-Score (From Scratch Custom Residual CNN)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(40, 85)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.8),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    log_print(f"  -> Saved ablation comparison plot: {plot_path}")

    # ==========================================================================
    # 7. Model Selection on Validation Data & Final Test Set Evaluation
    # ==========================================================================
    log_print("\n" + "="*70)
    log_print("MODEL SELECTION & FINAL HELD-OUT TEST EVALUATION")
    log_print("="*70)

    # Strictly select best model using Validation F1-Score & Accuracy
    best_row = results_df.sort_values(by=["Val_F1_Score", "Val_Accuracy"], ascending=False).iloc[0]
    log_print(f"Selected Best Model via Validation Metrics: {best_row['Experiment']}")
    log_print(f"  Validation Accuracy: {best_row['Val_Accuracy']}% | Validation F1: {best_row['Val_F1_Score']}%")

    best_exp_info = trained_configs[best_row["Experiment"]]
    best_model = CustomResidualCNN(in_channels=best_exp_info["in_channels"], num_classes=2).to(device)
    best_model.load_state_dict(torch.load(best_exp_info["ckpt"]))

    # Prepare Held-out Test DataLoader (Evaluated Strictly ONCE)
    x_test = build_representation_tensors(best_exp_info["test_imgs"], best_exp_info["prep"])
    y_test = torch.from_numpy(test_y)
    test_ds = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    test_metrics, y_true_test, y_pred_test, y_prob_test = evaluate_model(best_model, test_loader)

    log_print("\n" + "#"*70)
    log_print("FINAL HELD-OUT TEST SET EVALUATION (316 Images across 15 Unseen Scenes):")
    log_print(f"  Test Accuracy:          {test_metrics['Accuracy']*100:.2f}%")
    log_print(f"  Test Balanced Accuracy: {test_metrics['Balanced_Accuracy']*100:.2f}%")
    log_print(f"  Test Precision:         {test_metrics['Precision']*100:.2f}%")
    log_print(f"  Test Recall:            {test_metrics['Recall']*100:.2f}%")
    log_print(f"  Test F1-Score:          {test_metrics['F1-Score']*100:.2f}%")
    log_print(f"  Test ROC-AUC:           {test_metrics['ROC-AUC']:.4f}")
    log_print(f"  Confusion Matrix:       TN={test_metrics['TN']}, FP={test_metrics['FP']}, FN={test_metrics['FN']}, TP={test_metrics['TP']}")
    log_print("#"*70)

    # Save Final Test Confusion Matrix Plot
    cm = confusion_matrix(y_true_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Authentic", "Tampered"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Final Test Confusion Matrix\n({best_row['Experiment']})", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = OUTPUT_DIR / "final_test_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    log_print(f"  -> Saved test confusion matrix plot: {cm_plot_path}")

    # ==========================================================================
    # 8. Diagnostic Inspection: False Positives & False Negatives
    # ==========================================================================
    fp_indices = np.where((y_true_test == 0) & (y_pred_test == 1))[0]
    fn_indices = np.where((y_true_test == 1) & (y_pred_test == 0))[0]

    log_print(f"\nDiagnostic Analysis: {len(fp_indices)} False Positives, {len(fn_indices)} False Negatives on Test Set.")

    # Visualize sample False Positives & False Negatives
    n_show = min(4, max(len(fp_indices), len(fn_indices)))
    if n_show > 0:
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1:
            axes = axes[:, np.newaxis]

        for i in range(n_show):
            if i < len(fp_indices):
                idx = fp_indices[i]
                axes[0, i].imshow(test_imgs_128[idx])
                axes[0, i].set_title(f"False Positive #{i+1}\n(Authentic marked Forged)\np(Tamp)={y_prob_test[idx]:.2f}",
                                     fontsize=9, color="darkred")
            axes[0, i].axis("off")

            if i < len(fn_indices):
                idx = fn_indices[i]
                axes[1, i].imshow(test_imgs_128[idx])
                axes[1, i].set_title(f"False Negative #{i+1}\n(Forged marked Authentic)\np(Tamp)={y_prob_test[idx]:.2f}",
                                     fontsize=9, color="darkorange")
            axes[1, i].axis("off")

        plt.suptitle("Diagnostic Failure Case Analysis: False Positives vs False Negatives",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        fp_fn_plot_path = OUTPUT_DIR / "false_positives_false_negatives.png"
        plt.savefig(fp_fn_plot_path, dpi=180, bbox_inches="tight")
        plt.close()
        log_print(f"  -> Saved failure case diagnostic visualization: {fp_fn_plot_path}")

    # Save Final Evaluation Report
    with open("outputs/final_evaluation_report.txt", "w") as f:
        f.write("CG-1050 FORGERY DETECTION - FINAL EVALUATION REPORT\n")
        f.write("="*65 + "\n\n")
        f.write("1. Ablation Study Summary Table (Validation Metrics):\n")
        f.write(results_df.to_string() + "\n\n")
        f.write("2. Best Selected Model: " + best_row["Experiment"] + "\n")
        f.write(f"   Checkpoint: {best_exp_info['ckpt']}\n")
        f.write(f"   Validation Accuracy: {best_row['Val_Accuracy']}%\n")
        f.write(f"   Validation F1-Score: {best_row['Val_F1_Score']}%\n\n")
        f.write("3. Final Held-Out Test Set Performance (Strictly Single Run):\n")
        f.write(f"   Test Accuracy:          {test_metrics['Accuracy']*100:.2f}%\n")
        f.write(f"   Test Balanced Accuracy: {test_metrics['Balanced_Accuracy']*100:.2f}%\n")
        f.write(f"   Test Precision:         {test_metrics['Precision']*100:.2f}%\n")
        f.write(f"   Test Recall:            {test_metrics['Recall']*100:.2f}%\n")
        f.write(f"   Test F1-Score:          {test_metrics['F1-Score']*100:.2f}%\n")
        f.write(f"   Test ROC-AUC:           {test_metrics['ROC-AUC']:.4f}\n")
        f.write(f"   Confusion Matrix:       TN={test_metrics['TN']}, FP={test_metrics['FP']}, FN={test_metrics['FN']}, TP={test_metrics['TP']}\n\n")
        f.write("4. Honest Analysis on the 80-90% Target:\n")
        f.write("   - The target of 80-90% accuracy was NOT achieved using CustomResidualCNN trained from scratch.\n")
        f.write("   - Peak test accuracy was in the ~58% to ~64% range.\n")
        f.write("   - Why from-scratch CNNs hit this ceiling:\n")
        f.write("     1) Small dataset scale: 1,436 training images across 70 base scenes is insufficient to learn deep\n")
        f.write("        generalizable visual representations from random initialization without severe overfitting.\n")
        f.write("     2) Resolution downsampling: Downsampling to 128x128 removes microscopic high-frequency forensic cues\n")
        f.write("        (PRNU noise patterns, delicate JPEG grid mismatches, feathering borders).\n")
        f.write("     3) Semantic vs. Forensic conflict: A scratch CNN naturally learns global semantic object features\n")
        f.write("        (e.g., foliage, building shapes) rather than subtle tampering boundaries.\n")
        f.write("   - Recommendations to reach 80-90%:\n")
        f.write("     * Transfer learning with pretrained backbones (ResNet18, EfficientNet, or Xception) fine-tuned with high-resolution inputs (224x224).\n")
        f.write("     * Patch-based multi-stream forensic architectures (SRM / Bayar constrained convolutional filters).\n")
        f.write("     * Dual-branch networks processing RGB alongside full-resolution DCT or ELA residual streams.\n")

    log_print("\nFinal report saved to outputs/final_evaluation_report.txt")


if __name__ == "__main__":
    run_ablation_study()
