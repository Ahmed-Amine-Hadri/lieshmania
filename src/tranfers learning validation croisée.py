import os
import random
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from PIL import Image
import matplotlib.pyplot as plt


# ===============================================================
# 1. REPRODUCTIBILITÉ
# ===============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproductibilité.
    # Peut légèrement réduire les performances GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================================================
# 2. DATASET GÉNÉRIQUE
# ===============================================================

class ImageListDataset(Dataset):
    """
    Dataset basé sur une liste de (path, label).
    Permet d'utiliser une transformation différente pour le train
    et pour l'évaluation sur le même pool d'images.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ===============================================================
# 3. MODÈLE : RESNET18 + FINE-TUNING PROGRESSIF
# ===============================================================

def build_model(num_classes, device, dropout_rate=0.35):
    """
    ResNet18 pré-entraîné.

    Au départ :
        - backbone gelé
        - seule la tête classifier est entraînable

    Le fine-tuning de layer4 sera activé plus tard.
    """

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze complet du backbone
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(device)


def unfreeze_layer4(model):
    """
    Dégèle uniquement layer4 pour le fine-tuning.
    """

    for param in model.layer4.parameters():
        param.requires_grad = True


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


# ===============================================================
# 4. MÉTRIQUES
# ===============================================================

def compute_metrics(labels, preds, probs=None, num_classes=2):
    """
    Calcule plusieurs métriques.
    Pour une classification binaire :
        Accuracy
        Precision
        Recall
        F1
        Specificity
        Balanced Accuracy
        ROC-AUC
    """

    acc = accuracy_score(labels, preds)

    average_type = "binary" if num_classes == 2 else "weighted"

    precision = precision_score(
        labels,
        preds,
        average=average_type,
        zero_division=0
    )

    recall = recall_score(
        labels,
        preds,
        average=average_type,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        average=average_type,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(labels, preds)

    specificity = None

    if num_classes == 2:
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = None

    if num_classes == 2 and probs is not None:
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = None

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "auc": auc,
    }


# ===============================================================
# 5. ÉVALUATION
# ===============================================================

def evaluate(model, loader, device, criterion=None, num_classes=2):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(
        all_labels,
        all_preds,
        all_probs,
        num_classes=num_classes
    )

    loss = running_loss / total_samples if criterion is not None else None

    return loss, metrics, all_preds, all_labels, all_probs


# ===============================================================
# 6. COURBES D'APPRENTISSAGE
# ===============================================================

def plot_history(history, fold, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} - Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fold{fold}_loss.png"),
        dpi=150
    )
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Fold {fold} - Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fold{fold}_accuracy.png"),
        dpi=150
    )
    plt.close()

    # F1
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(f"Fold {fold} - Validation F1")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fold{fold}_f1.png"),
        dpi=150
    )
    plt.close()


# ===============================================================
# 7. MAIN
# ===============================================================

def main():

    # -----------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------

    TRAIN_DIR = "data/proceeded/train"
    VAL_DIR = "data/proceeded/val"
    TEST_DIR = "data/proceeded/test"

    WEIGHTS_DIR = "weights"
    PLOTS_DIR = os.path.join(WEIGHTS_DIR, "plots")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    SEED = 42

    K_FOLDS = 5

    # 10 epochs suffisent comme demandé.
    NUM_EPOCHS = 10

    # À partir de cette epoch, on dégèle layer4.
    FINE_TUNE_EPOCH = 4

    PATIENCE = 3

    BATCH_SIZE = 16

    IMG_SIZE = 224

    # Learning rates
    LR_HEAD = 1e-3
    LR_FINE_TUNE = 1e-4

    WEIGHT_DECAY = 1e-4

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("UPGRADED RESNET18 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(f"K-Folds         : {K_FOLDS}")
    print(f"Fine-tuning     : epoch {FINE_TUNE_EPOCH}")
    print(f"Batch size      : {BATCH_SIZE}")
    print("=" * 70)

    # -----------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.10
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # -----------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------

    train_raw = datasets.ImageFolder(
        root=TRAIN_DIR
    )

    val_raw = datasets.ImageFolder(
        root=VAL_DIR
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=eval_transforms
    )

    assert train_raw.classes == val_raw.classes, (
        "TRAIN et VAL doivent avoir les mêmes classes."
    )

    assert train_raw.classes == test_dataset.classes, (
        "TRAIN et TEST doivent avoir les mêmes classes."
    )

    classes = train_raw.classes
    num_classes = len(classes)

    pool_samples = train_raw.samples + val_raw.samples

    pool_labels = np.array([
        label for _, label in pool_samples
    ])

    print("\nClasses :", classes)

    print(
        f"Pool train+val : {len(pool_samples)} images"
    )

    print(
        f"Test           : {len(test_dataset)} images"
    )

    # -----------------------------------------------------------
    # Dataset views
    # -----------------------------------------------------------

    pool_train_view = ImageListDataset(
        pool_samples,
        transform=train_transforms
    )

    pool_eval_view = ImageListDataset(
        pool_samples,
        transform=eval_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------------
    # K-Fold
    # -----------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []
    fold_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(pool_samples, pool_labels),
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}/{K_FOLDS}")
        print("=" * 70)

        train_subset = Subset(
            pool_train_view,
            train_idx
        )

        train_eval_subset = Subset(
            pool_eval_view,
            train_idx
        )

        val_subset = Subset(
            pool_eval_view,
            val_idx
        )

        # -------------------------------------------------------
        # Class balancing
        # -------------------------------------------------------

        fold_train_labels = pool_labels[train_idx]

        class_counts = np.array([
            np.sum(fold_train_labels == c)
            for c in range(num_classes)
        ])

        class_weights = (
            len(fold_train_labels)
            / (
                num_classes *
                np.maximum(class_counts, 1)
            )
        )

        sample_weights = np.array([
            class_weights[label]
            for label in fold_train_labels
        ])

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weights,
                dtype=torch.double
            ),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        train_eval_loader = DataLoader(
            train_eval_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        print(
            f"Train : {len(train_subset)} | "
            f"Val : {len(val_subset)}"
        )

        print(
            f"Class counts : {class_counts.tolist()}"
        )

        # -------------------------------------------------------
        # Model
        # -------------------------------------------------------

        model = build_model(
            num_classes=num_classes,
            device=device,
            dropout_rate=0.35
        )

        # Loss pondérée
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device
            )
        )

        # Au départ : uniquement FC
        optimizer = optim.AdamW(
            model.fc.parameters(),
            lr=LR_HEAD,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_val_f1 = -float("inf")

        epochs_no_improve = 0

        fold_model_path = os.path.join(
            WEIGHTS_DIR,
            f"fold{fold}_best.pth"
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_balanced_accuracy": [],
            "val_auc": [],
            "bias": [],
            "variance": [],
            "learning_rate": []
        }

        # -------------------------------------------------------
        # Training
        # -------------------------------------------------------

        for epoch in range(NUM_EPOCHS):

            # -----------------------------------------------
            # Fine-tuning progressif
            # -----------------------------------------------

            if epoch + 1 == FINE_TUNE_EPOCH:

                print("\n>>> Activation du fine-tuning de layer4...")

                unfreeze_layer4(model)

                # Optimizer avec deux learning rates :
                # layer4 faible LR + classifier plus fort LR.
                optimizer = optim.AdamW(
                    [
                        {
                            "params": model.layer4.parameters(),
                            "lr": LR_FINE_TUNE
                        },
                        {
                            "params": model.fc.parameters(),
                            "lr": LR_HEAD
                        }
                    ],
                    weight_decay=WEIGHT_DECAY
                )

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=1,
                    min_lr=1e-6
                )

            # -----------------------------------------------
            # TRAIN
            # -----------------------------------------------

            model.train()

            running_train_loss = 0.0
            train_samples = 0

            for inputs, labels in train_loader:

                inputs = inputs.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                # Évite les gradients explosifs.
                torch.nn.utils.clip_grad_norm_(
                    get_trainable_parameters(model),
                    max_norm=1.0
                )

                optimizer.step()

                running_train_loss += (
                    loss.item() *
                    inputs.size(0)
                )

                train_samples += inputs.size(0)

            train_loss = (
                running_train_loss /
                max(train_samples, 1)
            )

            # -----------------------------------------------
            # TRAIN CLEAN
            # -----------------------------------------------

            _, train_metrics, _, _, _ = evaluate(
                model,
                train_eval_loader,
                device,
                criterion=None,
                num_classes=num_classes
            )

            # -----------------------------------------------
            # VALIDATION
            # -----------------------------------------------

            val_loss, val_metrics, _, _, _ = evaluate(
                model,
                val_loader,
                device,
                criterion=criterion,
                num_classes=num_classes
            )

            scheduler.step(val_loss)

            train_error = 1.0 - train_metrics["accuracy"]
            val_error = 1.0 - val_metrics["accuracy"]

            # Mesure pratique du gap train/validation.
            variance = max(
                0.0,
                val_error - train_error
            )

            # Ici on ne prétend pas connaître exactement
            # le Bayes error. On utilise l'erreur train
            # comme proxy observable.
            bias = train_error

            current_lrs = [
                group["lr"]
                for group in optimizer.param_groups
            ]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            history["train_acc"].append(
                train_metrics["accuracy"]
            )

            history["val_acc"].append(
                val_metrics["accuracy"]
            )

            history["val_precision"].append(
                val_metrics["precision"]
            )

            history["val_recall"].append(
                val_metrics["recall"]
            )

            history["val_f1"].append(
                val_metrics["f1"]
            )

            history["val_balanced_accuracy"].append(
                val_metrics["balanced_accuracy"]
            )

            history["val_auc"].append(
                val_metrics["auc"]
                if val_metrics["auc"] is not None
                else 0.0
            )

            history["bias"].append(bias)
            history["variance"].append(variance)

            history["learning_rate"].append(
                max(current_lrs)
            )

            auc_text = (
                f"{val_metrics['auc']:.4f}"
                if val_metrics["auc"] is not None
                else "N/A"
            )

            print(
                f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
                f"TrainLoss {train_loss:.4f} | "
                f"TrainAcc {train_metrics['accuracy']:.4f} | "
                f"ValLoss {val_loss:.4f} | "
                f"ValAcc {val_metrics['accuracy']:.4f} | "
                f"Precision {val_metrics['precision']:.4f} | "
                f"Recall {val_metrics['recall']:.4f} | "
                f"F1 {val_metrics['f1']:.4f} | "
                f"AUC {auc_text} | "
                f"Bias {bias:.4f} | "
                f"Variance {variance:.4f}"
            )

            # -----------------------------------------------
            # CHECKPOINT
            # -----------------------------------------------

            # F1 comme critère principal.
            # En cas d'égalité, on préfère la plus faible val loss.
            is_better = (
                val_metrics["f1"] > best_val_f1
                or (
                    np.isclose(
                        val_metrics["f1"],
                        best_val_f1
                    )
                    and val_loss < best_val_loss
                )
            )

            if is_better:

                best_val_f1 = val_metrics["f1"]
                best_val_loss = val_loss

                epochs_no_improve = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classes": classes,
                        "num_classes": num_classes,
                        "fold": fold,
                        "best_val_f1": best_val_f1,
                        "best_val_loss": best_val_loss,
                    },
                    fold_model_path
                )

                print(
                    f"  ✓ Nouveau meilleur modèle sauvegardé "
                    f"(F1={best_val_f1:.4f})"
                )

            else:

                epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:

                    print(
                        f"  Early stopping après "
                        f"{epoch + 1} epochs."
                    )

                    break

        # -------------------------------------------------------
        # Courbes
        # -------------------------------------------------------

        plot_history(
            history,
            fold,
            PLOTS_DIR
        )

        # Sauvegarde historique JSON
        history_path = os.path.join(
            PLOTS_DIR,
            f"fold{fold}_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

        # -------------------------------------------------------
        # Reload best model
        # -------------------------------------------------------

        checkpoint = torch.load(
            fold_model_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        final_val_loss, final_val_metrics, _, _, _ = evaluate(
            model,
            val_loader,
            device,
            criterion=criterion,
            num_classes=num_classes
        )

        print("\nRESULTAT FOLD")
        print(
            f"Accuracy          : "
            f"{final_val_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision         : "
            f"{final_val_metrics['precision']:.4f}"
        )

        print(
            f"Recall            : "
            f"{final_val_metrics['recall']:.4f}"
        )

        print(
            f"F1                : "
            f"{final_val_metrics['f1']:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{final_val_metrics['balanced_accuracy']:.4f}"
        )

        if final_val_metrics["specificity"] is not None:
            print(
                f"Specificity       : "
                f"{final_val_metrics['specificity']:.4f}"
            )

        if final_val_metrics["auc"] is not None:
            print(
                f"AUC               : "
                f"{final_val_metrics['auc']:.4f}"
            )

        fold_results.append(final_val_metrics)
        fold_model_paths.append(fold_model_path)

    # ===========================================================
    # 8. RÉSUMÉ K-FOLD
    # ===========================================================

    print("\n")
    print("=" * 70)
    print("RÉSUMÉ CROSS-VALIDATION")
    print("=" * 70)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "specificity",
        "auc"
    ]

    summary = {}

    for metric in metric_names:

        values = [
            result[metric]
            for result in fold_results
            if result[metric] is not None
        ]

        if values:

            mean_value = float(np.mean(values))
            std_value = float(np.std(values))

            summary[metric] = {
                "mean": mean_value,
                "std": std_value
            }

            print(
                f"{metric:20s}: "
                f"{mean_value:.4f} ± {std_value:.4f}"
            )

    with open(
        os.path.join(
            WEIGHTS_DIR,
            "cross_validation_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ===========================================================
    # 9. ENSEMBLE SUR TEST
    # ===========================================================

    print("\n")
    print("=" * 70)
    print("ÉVALUATION FINALE — ENSEMBLE K-FOLD")
    print("=" * 70)

    summed_probs = None
    test_labels_all = None

    # Poids basés sur le F1 validation.
    ensemble_weights = np.array([
        max(result["f1"], 1e-8)
        for result in fold_results
    ])

    ensemble_weights /= ensemble_weights.sum()

    print(
        "Poids ensemble :",
        [
            f"{w:.4f}"
            for w in ensemble_weights
        ]
    )

    for fold_idx, fold_path in enumerate(
        fold_model_paths
    ):

        model = build_model(
            num_classes,
            device
        )

        checkpoint = torch.load(
            fold_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        fold_probs = []
        fold_labels = []

        with torch.no_grad():

            for inputs, labels in test_loader:

                inputs = inputs.to(
                    device,
                    non_blocking=True
                )

                outputs = model(inputs)

                probs = torch.softmax(
                    outputs,
                    dim=1
                ).cpu().numpy()

                fold_probs.append(probs)
                fold_labels.append(
                    labels.numpy()
                )

        fold_probs = np.concatenate(
            fold_probs,
            axis=0
        )

        fold_labels = np.concatenate(
            fold_labels,
            axis=0
        )

        weight = ensemble_weights[fold_idx]

        if summed_probs is None:

            summed_probs = (
                weight *
                fold_probs
            )

            test_labels_all = fold_labels

        else:

            summed_probs += (
                weight *
                fold_probs
            )

    avg_probs = summed_probs

    test_preds = np.argmax(
        avg_probs,
        axis=1
    )

    # -----------------------------------------------------------
    # Test metrics
    # -----------------------------------------------------------

    test_metrics = compute_metrics(
        test_labels_all,
        test_preds,
        avg_probs,
        num_classes=num_classes
    )

    print("\nTEST FINAL")
    print("-" * 50)

    for key, value in test_metrics.items():

        if value is not None:

            print(
                f"{key:20s}: {value:.4f}"
            )

    # ===========================================================
    # 10. MATRICE DE CONFUSION
    # ===========================================================

    cm = confusion_matrix(
        test_labels_all,
        test_preds
    )

    print("\nMatrice de confusion :")
    print(cm)

    # Brute
    plt.figure(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(
        values_format="d",
        ax=plt.gca()
    )

    plt.title(
        f"Confusion Matrix - Ensemble {K_FOLDS}-Fold"
    )

    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            WEIGHTS_DIR,
            "confusion_matrix_ensemble.png"
        ),
        dpi=200
    )

    plt.close()

    # Normalisée
    cm_normalized = (
        cm.astype(float) /
        np.maximum(
            cm.sum(axis=1, keepdims=True),
            1
        )
    )

    plt.figure(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=classes
    )

    disp.plot(
        values_format=".2f",
        ax=plt.gca()
    )

    plt.title(
        f"Normalized Confusion Matrix - Ensemble {K_FOLDS}-Fold"
    )

    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            WEIGHTS_DIR,
            "confusion_matrix_ensemble_normalized.png"
        ),
        dpi=200
    )

    plt.close()

    # ===========================================================
    # 11. SAUVEGARDE DES RÉSULTATS FINAUX
    # ===========================================================

    final_results = {
        "classes": classes,
        "num_classes": num_classes,
        "k_folds": K_FOLDS,
        "epochs": NUM_EPOCHS,
        "fine_tune_epoch": FINE_TUNE_EPOCH,
        "batch_size": BATCH_SIZE,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "ensemble_weights": ensemble_weights.tolist()
    }

    with open(
        os.path.join(
            WEIGHTS_DIR,
            "final_results.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2
        )

    print("\n")
    print("=" * 70)
    print("PIPELINE TERMINÉ")
    print("=" * 70)

    print(
        f"Accuracy finale : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"F1 final        : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Résultats       : {WEIGHTS_DIR}/"
    )


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":
    main()