"""
Leishmania segmentation training script
-----------------------------------------
- U-Net with a pretrained (ImageNet) encoder -> transfer learning
- Encoder frozen so only the decoder trains -> fast convergence in few epochs
- Images are split into non-overlapping patches -> smaller tensors, faster steps,
  more training samples out of a small dataset
- Mixed precision (AMP) on GPU -> extra speed
- 5 epochs by default

Expected layout (matches your repo):
data/raw/segmentation/images/pos_web_000.jpg, neg_web_000.jpg, ...
data/raw/segmentation/masks/pos_web_000.png,  neg_web_000.png, ...

Install once:
    pip install torch torchvision segmentation-models-pytorch albumentations opencv-python tqdm

Run:
    python src/unet_segmentation_train.py
"""

import os
import glob
import time
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMAGES_DIR = os.path.join("data", "raw", "segmentation", "images")
MASKS_DIR = os.path.join("data", "raw", "segmentation", "masks")
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

PATCH_SIZE = 128          # smaller patch -> ~4x fewer pixels per sample than 256, big CPU win
VAL_SPLIT = 0.15
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
ENCODER = "timm-mobilenetv3_small_100"  # much lighter than mobilenet_v2, built for CPU/edge speed
ENCODER_WEIGHTS = "imagenet"
FREEZE_ENCODER = True           # transfer learning: only decoder trains -> fast
NUM_WORKERS = 0                 # on Windows, worker spawn overhead often outweighs the benefit for small in-RAM datasets
SEED = 42

# CPU-speed controls: caps total patches trained/validated on per epoch.
# This is the single biggest lever for wall-clock time on CPU -- lower these
# further if an epoch is still too slow, raise them if you want more data used.
MAX_TRAIN_PATCHES = 600
MAX_VAL_PATCHES = 120

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # fixed patch size -> lets cuDNN pick fastest kernels
    print(f"[info] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(os.cpu_count())  # use every CPU core for matmul/conv ops
    print(f"[info] running on CPU with {os.cpu_count()} threads")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------
# 1. Gather image/mask pairs (pos_web_* and neg_web_*)
# --------------------------------------------------------------------------
def gather_pairs(images_dir, masks_dir):
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "pos_web_*.jpg"))
        + glob.glob(os.path.join(images_dir, "neg_web_*.jpg"))
    )

    pairs = []
    missing = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. pos_web_000
        mask_path = os.path.join(masks_dir, stem + ".png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            missing.append(stem)

    if missing:
        print(f"[warn] {len(missing)} images have no matching mask, skipped "
              f"(e.g. {missing[:5]})")

    print(f"[info] found {len(pairs)} image/mask pairs")
    return pairs


# --------------------------------------------------------------------------
# 2. Split into non-overlapping patches
# --------------------------------------------------------------------------
def extract_patches(image, mask, patch_size):
    """image: HxWx3 uint8, mask: HxW uint8 (0/255). Returns list of (patch_img, patch_mask)."""
    h, w = mask.shape[:2]

    # If the image is smaller than one patch, just resize up to patch size.
    if h < patch_size or w < patch_size:
        image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        return [(image, mask)]

    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            img_p = image[y:y + patch_size, x:x + patch_size]
            msk_p = mask[y:y + patch_size, x:x + patch_size]
            patches.append((img_p, msk_p))

    # capture the remainder on the right/bottom edge too (one extra patch each side)
    if h % patch_size != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append((image[y:y + patch_size, x:x + patch_size],
                             mask[y:y + patch_size, x:x + patch_size]))
    if w % patch_size != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, patch_size):
            patches.append((image[y:y + patch_size, x:x + patch_size],
                             mask[y:y + patch_size, x:x + patch_size]))

    return patches


def build_patch_bank(pairs, patch_size):
    """Loads every image/mask pair once and pre-cuts it into patches held in RAM."""
    bank = []
    for img_path, mask_path in tqdm(pairs, desc="Extracting patches"):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"[warn] could not read {img_path} or {mask_path}, skipping")
            continue

        for img_p, msk_p in extract_patches(image, mask, patch_size):
            bank.append((img_p, msk_p))

    print(f"[info] built {len(bank)} patches from {len(pairs)} images")
    return bank


# --------------------------------------------------------------------------
# 3. Dataset
# --------------------------------------------------------------------------
def get_train_transform(patch_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(patch_size):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class PatchDataset(Dataset):
    def __init__(self, patch_bank, transform):
        self.bank = patch_bank
        self.transform = transform

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, idx):
        image, mask = self.bank[idx]
        mask = (mask > 127).astype(np.float32)  # binarize to 0/1

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0).float()  # 1xHxW
        return image_t, mask_t


# --------------------------------------------------------------------------
# 4. Model
# --------------------------------------------------------------------------
def build_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,  # raw logits, loss handles sigmoid
    )

    if FREEZE_ENCODER:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[info] encoder frozen (transfer learning: decoder-only training)")

    return model.to(DEVICE)


# --------------------------------------------------------------------------
# 5. Loss / metric
# --------------------------------------------------------------------------
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()


def criterion(logits, targets):
    return 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)


@torch.no_grad()
def dice_score(logits, targets, eps=1e-7):
    """Per-sample dice. Returns a tensor of shape (B,) -- NOT averaged here,
    so callers can separate out empty-mask ("trivial background") samples
    from real lesion-containing samples."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice  # per-sample tensor


# --------------------------------------------------------------------------
# 6. Train / validate loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, scaler, train=True):
    model.train() if train else model.eval()

    total_loss, n_batches = 0.0, 0
    all_dice_sum, all_dice_n = 0.0, 0          # every sample, incl. empty-mask ones
    fg_dice_sum, fg_dice_n = 0.0, 0            # only samples that actually contain a lesion
    torch.set_grad_enabled(train)

    for images, masks in tqdm(loader, desc="train" if train else "val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        per_sample_dice = dice_score(logits, masks)               # (B,)
        has_lesion = masks.sum(dim=(1, 2, 3)) > 0                  # (B,) bool

        all_dice_sum += per_sample_dice.sum().item()
        all_dice_n += per_sample_dice.numel()

        if has_lesion.any():
            fg_dice_sum += per_sample_dice[has_lesion].sum().item()
            fg_dice_n += int(has_lesion.sum().item())

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice_all = all_dice_sum / max(all_dice_n, 1)
    avg_dice_fg = fg_dice_sum / fg_dice_n if fg_dice_n > 0 else float("nan")
    return avg_loss, avg_dice_all, avg_dice_fg, fg_dice_n, all_dice_n


def main():
    pairs = gather_pairs(IMAGES_DIR, MASKS_DIR)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No image/mask pairs found. Check {IMAGES_DIR} and {MASKS_DIR}."
        )

    # split by IMAGE first (not by patch) to avoid leaking patches from the
    # same image into both train and val
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"[info] {len(train_pairs)} train images / {len(val_pairs)} val images")

    train_bank = build_patch_bank(train_pairs, PATCH_SIZE)
    val_bank = build_patch_bank(val_pairs, PATCH_SIZE)

    # cap how many patches actually get trained/validated on -- this is what
    # keeps a CPU epoch from ballooning to thousands of steps.
    # Bias sampling towards lesion-containing patches: with a random sample,
    # mostly-empty datasets end up training on almost nothing but background.
    def biased_sample(bank, max_n):
        if not max_n or len(bank) <= max_n:
            return bank
        fg = [p for p in bank if (p[1] > 127).any()]
        bg = [p for p in bank if not (p[1] > 127).any()]
        # aim for roughly half lesion-containing patches, capped by availability
        n_fg = min(len(fg), max_n // 2)
        n_bg = max_n - n_fg
        sampled = random.sample(fg, n_fg) + random.sample(bg, min(n_bg, len(bg)))
        random.shuffle(sampled)
        return sampled

    train_bank = biased_sample(train_bank, MAX_TRAIN_PATCHES)
    val_bank = biased_sample(val_bank, MAX_VAL_PATCHES)
    print(f"[info] using {len(train_bank)} train patches / {len(val_bank)} val patches this run")

    # sanity check: how much of the data is actually lesion-containing vs empty?
    def fg_fraction(bank):
        n_fg = sum(1 for _, m in bank if (m > 127).any())
        return n_fg, len(bank)

    tr_fg, tr_n = fg_fraction(train_bank)
    va_fg, va_n = fg_fraction(val_bank)
    print(f"[info] train patches with a lesion: {tr_fg}/{tr_n} "
          f"({100 * tr_fg / max(tr_n, 1):.1f}%)")
    print(f"[info] val patches with a lesion:   {va_fg}/{va_n} "
          f"({100 * va_fg / max(va_n, 1):.1f}%)")
    if va_fg < 10:
        print("[warn] very few lesion-containing val patches -- val_dice_fg will be noisy/unreliable")

    train_ds = PatchDataset(train_bank, get_train_transform(PATCH_SIZE))
    val_ds = PatchDataset(val_bank, get_val_transform(PATCH_SIZE))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_dice_fg = -1.0
    best_path = os.path.join(WEIGHTS_DIR, "unet_best.pth")

    print(f"[info] training on {DEVICE} for {EPOCHS} epochs")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_dice_all, train_dice_fg, tr_fg_n, _ = run_epoch(
            model, train_loader, optimizer, scaler, train=True)
        val_loss, val_dice_all, val_dice_fg, va_fg_n, _ = run_epoch(
            model, val_loader, optimizer, scaler, train=False)
        dt = time.time() - t0

        print(f"[epoch {epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_dice_all={train_dice_all:.4f} "
              f"train_dice_fg={train_dice_fg:.4f} (n={tr_fg_n}) | "
              f"val_loss={val_loss:.4f} val_dice_all={val_dice_all:.4f} "
              f"val_dice_fg={val_dice_fg:.4f} (n={va_fg_n}) | {dt:.1f}s")

        # select "best" using the honest, lesion-only score -- not the
        # trivially-inflated all-patches score
        if va_fg_n > 0 and val_dice_fg > best_val_dice_fg:
            best_val_dice_fg = val_dice_fg
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (val_dice_fg={val_dice_fg:.4f}), saved to {best_path}")

    final_path = os.path.join(WEIGHTS_DIR, "unet_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[done] best val_dice_fg (lesion-only, the honest metric) = {best_val_dice_fg:.4f}")
    print(f"[done] weights in {WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()