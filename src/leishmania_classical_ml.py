#!/usr/bin/env python3
"""
Leishmania / Non-Leishmania classifier — classical ML pipeline
================================================================

Pipeline (matches the diagram):

    220 images
        -> Preprocessing
        -> Feature extraction:
              HOG   -> shape
              LBP   -> texture
              Color -> appearance
        -> Feature concatenation
        -> StandardScaler
        -> PCA
        -> SVM (RBF kernel)
        -> Leishmania / Non-Leishmania

This is a classical machine-learning pipeline (hand-crafted features + SVM),
NOT deep learning — there is no CNN here.

Expected data layout (like torchvision.datasets.ImageFolder):

    data/
        Leishmania/
            img001.png
            img002.png
            ...
        Non-Leishmania/
            img001.png
            ...

Usage:
    python leishmania_svm_pipeline.py --data-dir data --img-size 128
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ===============================================================
# 1. CONFIG
# ===============================================================
@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "weights_svm"
    img_size: int = 128
    test_size: float = 0.2
    seed: int = 42

    # HOG params
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 16
    hog_cells_per_block: int = 2

    # LBP params
    lbp_radius: int = 2
    lbp_n_points: int = 16  # typically 8 * radius
    lbp_method: str = "uniform"

    # Color histogram params
    color_bins: int = 32  # per channel

    # PCA
    pca_variance: float = 0.95  # keep components explaining 95% variance

    # SVM grid search
    svm_C: Tuple[float, ...] = (0.1, 1, 10, 100)
    svm_gamma: Tuple[str, ...] = ("scale", "auto")


# ===============================================================
# 2. IMAGE LOADING + PREPROCESSING
# ===============================================================
def list_images(data_dir: Path) -> Tuple[List[Path], List[str], List[str]]:
    """Walk a class-per-folder directory and return (paths, labels, class_names)."""
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No class subfolders found in {data_dir}")

    paths, labels = [], []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for cls in class_names:
        for p in sorted((data_dir / cls).iterdir()):
            if p.suffix.lower() in exts:
                paths.append(p)
                labels.append(cls)

    print(f"Found {len(paths)} images across {len(class_names)} classes: {class_names}")
    return paths, labels, class_names


def preprocess_image(path: Path, img_size: int) -> np.ndarray:
    """
    Load an image and apply basic preprocessing:
      - resize to a fixed square size
      - denoise slightly
    Returns a BGR uint8 image (OpenCV convention).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ===============================================================
# 3. FEATURE EXTRACTION
# ===============================================================
def extract_hog_features(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """HOG -> captures SHAPE / edge structure."""
    features = hog(
        gray,
        orientations=cfg.hog_orientations,
        pixels_per_cell=(cfg.hog_pixels_per_cell, cfg.hog_pixels_per_cell),
        cells_per_block=(cfg.hog_cells_per_block, cfg.hog_cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_lbp_features(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """LBP histogram -> captures TEXTURE."""
    lbp = local_binary_pattern(
        gray, P=cfg.lbp_n_points, R=cfg.lbp_radius, method=cfg.lbp_method
    )
    n_bins = cfg.lbp_n_points + 2  # uniform method bins
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )
    return hist.astype(np.float32)


def extract_color_features(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Color histogram (HSV) -> captures APPEARANCE."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [cfg.color_bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hists.append(h)
    return np.concatenate(hists).astype(np.float32)


def extract_features(path: Path, cfg: Config) -> np.ndarray:
    bgr = preprocess_image(path, cfg.img_size)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    hog_feat = extract_hog_features(gray, cfg)
    lbp_feat = extract_lbp_features(gray, cfg)
    color_feat = extract_color_features(bgr, cfg)

    return np.concatenate([hog_feat, lbp_feat, color_feat])


def build_feature_matrix(paths: List[Path], cfg: Config) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths, 1):
        feats.append(extract_features(p, cfg))
        if i % 25 == 0 or i == len(paths):
            print(f"  extracted features: {i}/{len(paths)}")
    return np.vstack(feats)


# ===============================================================
# 4. MAIN
# ===============================================================
def main(cfg: Config) -> None:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEISHMANIA / NON-LEISHMANIA — CLASSICAL ML PIPELINE (HOG+LBP+Color+SVM)")
    print("=" * 70)

    # ---- Load + label images ----
    paths, labels, class_names = list_images(data_dir)
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels])

    # ---- Feature extraction ----
    print("\nExtracting HOG (shape) + LBP (texture) + Color (appearance) features...")
    X = build_feature_matrix(paths, cfg)
    print(f"Feature matrix shape: {X.shape}")

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ---- Pipeline: StandardScaler -> PCA -> SVM(RBF) ----
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=cfg.pca_variance, random_state=cfg.seed)),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    )

    param_grid = {
        "svm__C": list(cfg.svm_C),
        "svm__gamma": list(cfg.svm_gamma),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search (StandardScaler -> PCA -> SVM RBF)...")
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    n_components = best_model.named_steps["pca"].n_components_
    print(f"PCA kept {n_components} components "
          f"(explaining >= {cfg.pca_variance*100:.0f}% variance)")

    # ---- Evaluation on held-out test set ----
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 (w)   : {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, values_format="d"
    )
    ax.set_title("Confusion Matrix — SVM (RBF)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ---- Save model + results ----
    joblib.dump(best_model, out_dir / "svm_pipeline.joblib")

    results = {
        "classes": class_names,
        "config": cfg.__dict__,
        "best_params": grid.best_params_,
        "best_cv_f1": grid.best_score_,
        "pca_n_components": int(n_components),
        "test_accuracy": acc,
        "test_f1_weighted": f1,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel and results saved to: {out_dir}/")
    print("=" * 70)


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leishmania SVM (HOG+LBP+Color) trainer")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="weights_svm")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        test_size=args.test_size,
        pca_variance=args.pca_variance,
        seed=args.seed,
    )
    main(cfg)