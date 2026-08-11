#!/usr/bin/env python3
"""
Improved ResNet18 training pipeline for Leishmania classification
- Stratified 5-fold CV
- Progressive fine-tuning (layer3 + layer4)
- Mixup + Label Smoothing
- AMP + cosine schedule
- Weighted ensemble on test set
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ===============================================================
# 1. CONFIG
# ===============================================================
@dataclass
class Config:
    train_dir: str = "data/proceeded/train"
    val_dir: str = "data/proceeded/val"
    test_dir: str = "data/proceeded/test"
    weights_dir: str = "weights"
    seed: int = 42
    k_folds: int = 5
    num_epochs: int = 15
    fine_tune_epoch: int = 5          # unfreeze layer3+layer4 from this epoch
    patience: int = 5
    batch_size: int = 16
    img_size: int = 224
    lr_head: float = 1e-3
    lr_backbone: float = 3e-5
    weight_decay: float = 1e-4
    dropout: float = 0.4
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.2          # 0.0 = disabled
    num_workers: int = 4
    amp: bool = True

# ===============================================================
# 2. REPRODUCIBILITY
# ===============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ===============================================================
# 3. DATASET
# ===============================================================
class ImageListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================================================
# 4. MIXUP
# ===============================================================
def mixup_data(x, y, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===============================================================
# 5. MODEL
# ===============================================================
def build_model(num_classes: int, device: torch.device, dropout: float = 0.4) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes),
    )
    return model.to(device)

def unfreeze_backbone(model: nn.Module, layers: List[str] = ["layer3", "layer4"]) -> None:
    for name, module in model.named_children():
        if name in layers:
            for param in module.parameters():
                param.requires_grad = True

def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

# ===============================================================
# 6. METRICS
# ===============================================================
def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, Optional[float]]:
    average = "binary" if num_classes == 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "specificity": None,
        "auc": None,
    }

    if num_classes == 2:
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if probs is not None:
            try:
                metrics["auc"] = roc_auc_score(labels, probs[:, 1])
            except ValueError:
                metrics["auc"] = None

    return metrics

# ===============================================================
# 7. EVALUATION
# ===============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    num_classes: int = 2,
    use_amp: bool = True,
):
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        total += inputs.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    loss = running_loss / total if criterion is not None else None
    return loss, metrics, all_preds, all_labels, all_probs

# ===============================================================
# 8. PLOTTING
# ===============================================================
def plot_history(history: dict, fold: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    for key, ylabel, title in [
        (["train_loss", "val_loss"], "Loss", "Loss"),
        (["train_acc", "val_acc"], "Accuracy", "Accuracy"),
        (["val_f1"], "F1", "Validation F1"),
    ]:
        plt.figure(figsize=(8, 5))
        for k in key:
            plt.plot(epochs, history[k], label=k.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Fold {fold} – {title}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"fold{fold}_{title.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

# ===============================================================
# 9. TRAINING LOOP (one fold)
# ===============================================================
def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    pool_samples: list,
    pool_labels: np.ndarray,
    classes: list,
    cfg: Config,
    device: torch.device,
) -> Tuple[dict, Path]:

    num_classes = len(classes)
    weights_dir = Path(cfg.weights_dir)
    plots_dir = weights_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.25, 0.25, 0.15, 0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    pool_train = ImageListDataset(pool_samples, transform=train_tf)
    pool_eval = ImageListDataset(pool_samples, transform=eval_tf)

    train_subset = Subset(pool_train, train_idx)
    train_eval_subset = Subset(pool_eval, train_idx)
    val_subset = Subset(pool_eval, val_idx)

    # class weights
    fold_labels = pool_labels[train_idx]
    class_counts = np.bincount(fold_labels, minlength=num_classes)
    class_weights = len(fold_labels) / (num_classes * np.maximum(class_counts, 1))
    sample_weights = class_weights[fold_labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed + fold)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=cfg.num_workers > 0,
    )
    train_eval_loader = DataLoader(
        train_eval_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Class counts: {class_counts.tolist()}")

    # model & optim
    model = build_model(num_classes, device, dropout=cfg.dropout)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = optim.AdamW(model.fc.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    fold_model_path = weights_dir / f"fold{fold}_best.pth"

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
        "val_balanced_accuracy": [], "val_auc": [],
        "bias": [], "variance": [], "learning_rate": [],
    }

    for epoch in range(cfg.num_epochs):
        # progressive unfreeze
        if epoch + 1 == cfg.fine_tune_epoch:
            print("\n>>> Unfreezing layer3 + layer4 ...")
            unfreeze_backbone(model, layers=["layer3", "layer4"])
            optimizer = optim.AdamW(
                [
                    {"params": model.layer3.parameters(), "lr": cfg.lr_backbone},
                    {"params": model.layer4.parameters(), "lr": cfg.lr_backbone},
                    {"params": model.fc.parameters(), "lr": cfg.lr_head},
                ],
                weight_decay=cfg.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.num_epochs - epoch, eta_min=1e-6
            )

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold} Ep {epoch+1:02d}", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if cfg.mixup_alpha > 0:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, cfg.mixup_alpha)
            else:
                y_a = y_b = labels
                lam = 1.0

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.amp and device.type == "cuda"):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(n_samples, 1)
        scheduler.step()

        # clean train metrics (no mixup)
        _, train_metrics, _, _, _ = evaluate(
            model, train_eval_loader, device, criterion=None,
            num_classes=num_classes, use_amp=cfg.amp
        )

        # validation
        val_loss, val_metrics, _, _, _ = evaluate(
            model, val_loader, device, criterion=criterion,
            num_classes=num_classes, use_amp=cfg.amp
        )

        train_err = 1.0 - train_metrics["accuracy"]
        val_err = 1.0 - val_metrics["accuracy"]
        variance = max(0.0, val_err - train_err)
        bias = train_err

        current_lr = max(g["lr"] for g in optimizer.param_groups)

        # log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["val_auc"].append(val_metrics["auc"] or 0.0)
        history["bias"].append(bias)
        history["variance"].append(variance)
        history["learning_rate"].append(current_lr)

        auc_str = f"{val_metrics['auc']:.4f}" if val_metrics["auc"] is not None else "N/A"
        print(
            f"Ep {epoch+1:02d} | "
            f"TrLoss {train_loss:.4f} TrAcc {train_metrics['accuracy']:.4f} | "
            f"VaLoss {val_loss:.4f} VaAcc {val_metrics['accuracy']:.4f} "
            f"F1 {val_metrics['f1']:.4f} AUC {auc_str} | "
            f"Bias {bias:.4f} Var {variance:.4f} LR {current_lr:.2e}"
        )

        # checkpoint
        improved = (
            val_metrics["f1"] > best_val_f1 + 1e-4
            or (np.isclose(val_metrics["f1"], best_val_f1) and val_loss < best_val_loss)
        )
        if improved:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "classes": classes,
                    "num_classes": num_classes,
                    "fold": fold,
                    "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                    "config": asdict(cfg),
                },
                fold_model_path,
            )
            print(f"  ✓ Best model saved (F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    plot_history(history, fold, plots_dir)
    with open(plots_dir / f"fold{fold}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # final val metrics
    _, final_metrics, _, _, _ = evaluate(
        model, val_loader, device, criterion=criterion,
        num_classes=num_classes, use_amp=cfg.amp
    )
    print("\nFold result:")
    for k, v in final_metrics.items():
        if v is not None:
            print(f"  {k:20s}: {v:.4f}")

    return final_metrics, fold_model_path

# ===============================================================
# 10. MAIN
# ===============================================================
def main(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(cfg.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IMPROVED RESNET18 – LEISHMANIA CLASSIFIER")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Epochs          : {cfg.num_epochs}")
    print(f"K-Folds         : {cfg.k_folds}")
    print(f"Fine-tune from  : epoch {cfg.fine_tune_epoch}")
    print(f"Mixup α         : {cfg.mixup_alpha}")
    print(f"Label smoothing : {cfg.label_smoothing}")
    print(f"AMP             : {cfg.amp and device.type == 'cuda'}")
    print("=" * 70)

    # load data
    train_raw = datasets.ImageFolder(cfg.train_dir)
    val_raw = datasets.ImageFolder(cfg.val_dir)
    test_ds = datasets.ImageFolder(
        cfg.test_dir,
        transform=transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )

    assert train_raw.classes == val_raw.classes == test_ds.classes
    classes = train_raw.classes
    num_classes = len(classes)

    pool_samples = train_raw.samples + val_raw.samples
    pool_labels = np.array([lbl for _, lbl in pool_samples])

    print(f"\nClasses      : {classes}")
    print(f"Pool (tr+val): {len(pool_samples)}")
    print(f"Test         : {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # K-Fold
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    fold_results = []
    fold_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(pool_samples, pool_labels), 1):
        print("\n" + "=" * 70)
        print(f"FOLD {fold}/{cfg.k_folds}")
        print("=" * 70)

        metrics, path = train_one_fold(
            fold, train_idx, val_idx,
            pool_samples, pool_labels, classes, cfg, device
        )
        fold_results.append(metrics)
        fold_paths.append(path)

    # CV summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "specificity", "auc"]:
        vals = [r[metric] for r in fold_results if r[metric] is not None]
        if vals:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[metric] = {"mean": mean, "std": std}
            print(f"{metric:20s}: {mean:.4f} ± {std:.4f}")

    with open(weights_dir / "cross_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ===========================================================
    # ENSEMBLE ON TEST
    # ===========================================================
    print("\n" + "=" * 70)
    print("TEST SET – WEIGHTED ENSEMBLE")
    print("=" * 70)

    weights = np.array([max(r["f1"], 1e-8) for r in fold_results])
    weights /= weights.sum()
    print("Ensemble weights:", [f"{w:.3f}" for w in weights])

    summed_probs = None
    test_labels = None

    for i, path in enumerate(fold_paths):
        model = build_model(num_classes, device, dropout=cfg.dropout)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_probs, fold_labs = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                with autocast(enabled=cfg.amp and device.type == "cuda"):
                    out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                fold_probs.append(probs)
                fold_labs.append(y.numpy())

        fold_probs = np.concatenate(fold_probs)
        fold_labs = np.concatenate(fold_labs)

        if summed_probs is None:
            summed_probs = weights[i] * fold_probs
            test_labels = fold_labs
        else:
            summed_probs += weights[i] * fold_probs

    test_preds = np.argmax(summed_probs, axis=1)
    test_metrics = compute_metrics(test_labels, test_preds, summed_probs, num_classes)

    print("\nTEST METRICS")
    print("-" * 40)
    for k, v in test_metrics.items():
        if v is not None:
            print(f"{k:20s}: {v:.4f}")

    # confusion matrices
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion matrix:\n", cm)

    for normalize, suffix in [(False, ""), (True, "_normalized")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm if not normalize else cm.astype(float) / cm.sum(axis=1, keepdims=True),
            display_labels=classes,
        )
        disp.plot(ax=ax, values_format="d" if not normalize else ".2f")
        ax.set_title(f"Confusion Matrix – Ensemble{suffix.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(weights_dir / f"confusion_matrix_ensemble{suffix}.png", dpi=200)
        plt.close()

    # save everything
    final = {
        "classes": classes,
        "num_classes": num_classes,
        "config": asdict(cfg),
        "cv_summary": summary,
        "test_metrics": {k: (float(v) if v is not None else None) for k, v in test_metrics.items()},
        "confusion_matrix": cm.tolist(),
        "ensemble_weights": weights.tolist(),
    }
    with open(weights_dir / "final_results.json", "w") as f:
        json.dump(final, f, indent=2)

    # also save raw predictions for later analysis
    np.savez(
        weights_dir / "test_predictions.npz",
        labels=test_labels,
        preds=test_preds,
        probs=summed_probs,
        classes=np.array(classes),
    )

    print("\n" + "=" * 70)
    print("PIPELINE FINISHED")
    print(f"Final Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1       : {test_metrics['f1']:.4f}")
    print(f"Artifacts saved in  : {weights_dir}/")
    print("=" * 70)

# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leishmania ResNet18 trainer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        mixup_alpha=args.mixup,
        patience=args.patience,
        num_workers=args.workers,
        amp=not args.no_amp,
        seed=args.seed,
    )
    main(cfg)