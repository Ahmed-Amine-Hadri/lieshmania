import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Paths
    # ---------------------------------------------------------
    TRAIN_DIR = "data/proceeded/train"
    VAL_DIR = "data/proceeded/val"
    TEST_DIR = "data/proceeded/test"
    WEIGHTS_DIR = "weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)  # reproductibilité, important avec un dataset aussi petit

    BAYES_ERROR = 0.02

    # ---------------------------------------------------------
    # 2. Transforms
    # IMPORTANT : Resize obligatoire avant ToTensor(), sinon le DataLoader
    # plante dès que deux images n'ont pas exactement la même taille.
    # ---------------------------------------------------------
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    IMG_SIZE = 224

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ---------------------------------------------------------
    # 3. Load Data & Handle Imbalance
    # ---------------------------------------------------------
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transforms)

    # Copie du train set SANS augmentation, uniquement pour le diagnostic
    # bias/variance : on veut comparer train vs dev dans les mêmes
    # conditions (pas d'augmentation), sinon train_error est artificiellement
    # gonflé par le flip/rotation/color jitter et la variance calculée
    # (val_error - train_error) est faussée -> souvent écrasée à 0.
    train_eval_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=val_test_transforms)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    class_counts = [len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in train_dataset.classes]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---------------------------------------------------------
    # 4. Modèle : linear probing (backbone gelé) + Dropout
    # On garde le backbone gelé : avec ~220 images au total, dégeler
    # layer4 fait overfitter quasi instantanément (on l'a vu : train
    # Acc = 1.0000 dès l'epoch 11 dans la version précédente).
    # ---------------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    DROPOUT_RATE = 0.3  # plus élevé que la V1 pour compenser le très petit dataset
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(num_ftrs, len(train_dataset.classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # ---------------------------------------------------------
    # 5. Training & Validation avec Early Stopping sur val_loss
    # On sélectionne le meilleur modèle sur val_loss plutôt que val_acc :
    # sur un dev set de ~30 images, val_acc reste souvent identique sur
    # plusieurs epochs (ex: 0.9118 répété), donc peu discriminant.
    # ---------------------------------------------------------
    num_epochs = 30
    patience = 6
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_model_path = os.path.join(WEIGHTS_DIR, "best_resnet18_leishmania.pth")

    print("\nStarting Training & Dev Evaluation...")
    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_loss_train = 0.0
        running_corrects_train = 0
        train_preds, train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss_train / len(train_dataset)
        train_acc = running_corrects_train.double() / len(train_dataset)
        train_prec = precision_score(train_labels, train_preds, average='binary', zero_division=0)

        # --- VALIDATION ---
        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = running_loss_val / len(val_dataset)
        val_acc = running_corrects_val.double() / len(val_dataset)
        val_prec = precision_score(val_labels, val_preds, average='binary', zero_division=0)

        scheduler.step(val_loss)

        # --- DIAGNOSTIC TRAIN (SANS augmentation) ---
        # On ré-évalue le modèle sur les mêmes images de train, mais avec
        # val_test_transforms (pas de flip/rotation/color jitter), pour que
        # train_error et val_error soient comparables sur un pied d'égalité.
        # Sans ça, train_error est artificiellement gonflé par l'augmentation
        # et la variance (val_error - train_error) est faussée (souvent
        # écrasée à 0 par le max(0.0, ...)).
        model.eval()
        running_corrects_train_clean = 0
        with torch.no_grad():
            for inputs, labels in train_eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects_train_clean += torch.sum(preds == labels.data)

        train_acc_clean = running_corrects_train_clean.double() / len(train_eval_dataset)
        train_error_clean = 1.0 - train_acc_clean.item()
        val_error = 1.0 - val_acc.item()

        avoidable_bias = max(0.0, train_error_clean - BAYES_ERROR)
        variance = max(0.0, val_error - train_error_clean)

        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"TRAIN (augmenté, pendant l'entraînement) | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Loss: {train_loss:.4f}")
        print(f"TRAIN (propre, pour diagnostic)          | Acc: {train_acc_clean:.4f} | Error: {train_error_clean:.4f}")
        print(f"DEV   | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Loss: {val_loss:.4f} | Error: {val_error:.4f}")
        print(f"DIAG  | Avoidable Bias: {avoidable_bias:.4f} | Variance: {variance:.4f}")

        # --- Early stopping sur val_loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)
            print(f"  -> Nouveau meilleur modèle (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping déclenché après {epoch+1} epochs (pas d'amélioration depuis {patience} epochs).")
                break

    # ---------------------------------------------------------
    # 6. Final Test Set Evaluation & Confusion Matrix
    # ---------------------------------------------------------
    print("\n=============================================")
    print("Loading Best Model for Final Test Evaluation...")
    print("=============================================")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    running_corrects_test = 0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects_test += torch.sum(preds == labels.data)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = running_corrects_test.double() / len(test_dataset)
    test_prec = precision_score(test_labels, test_preds, average='binary', zero_division=0)
    test_error = 1.0 - test_acc.item()

    print(f"TEST RESULTS | Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Error: {test_error:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nMatrice de confusion brute :")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Matrice de Confusion - Jeu de Test (Leishmania)")
    plt.xlabel("Prédiction du Modèle")
    plt.ylabel("Vérité Terrain (Réalité)")
    plt.savefig(os.path.join(WEIGHTS_DIR, "confusion_matrix.png"))
    plt.show()

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()