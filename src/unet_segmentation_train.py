import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. Dataset Split (90:10) ---
def split_dataset(image_dir, mask_dir, test_size=0.10):
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
    all_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
    
    # Quick safety check
    assert len(all_images) == len(all_masks), f"Mismatch! Found {len(all_images)} images and {len(all_masks)} masks."
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=test_size, random_state=42
    )
    return train_imgs, val_imgs, train_masks, val_masks

# --- 2. Dynamic Patching Dataset ---
class SegmentationPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames, 
                 patch_size=256, stride=128, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches_info = []
        
        print("Calculating patch coordinates...")
        for img_name, mask_name in zip(image_filenames, mask_filenames):
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            with Image.open(img_path) as img:
                w, h = img.size
                # Sliding window to calculate (x,y) for every patch
                for y in range(0, h - self.patch_size + 1, self.stride):
                    for x in range(0, w - self.patch_size + 1, self.stride):
                        self.patches_info.append((img_path, mask_path, x, y))

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches_info[idx]
        
        # Crop the exact patch dynamically
        img = Image.open(img_path).convert('RGB')
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        img_array = np.array(img_patch)
        
        mask = Image.open(mask_path).convert('L')
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_array = np.array(mask_patch)
        mask_array = (mask_array > 127).astype(np.float32) # Binarize the mask

        # Apply geometric augmentations to BOTH simultaneously
        if self.transform:
            augmented = self.transform(image=img_array, mask=mask_array)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_array).float()

        mask_tensor = mask_tensor.unsqueeze(0) # Shape: (1, H, W)
        return img_tensor, mask_tensor

# --- 3. Combined Loss Function ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig.view(-1) * targets.view(-1)).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.sum() + targets.sum() + smooth)  
        return bce_loss + dice_loss

# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # 1. Setup Folders
    IMG_DIR = './data/raw/segmentation/images/'
    MASK_DIR = './data/raw/segmentation/masks/'
    WEIGHTS_DIR = './weights/'
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(WEIGHTS_DIR, 'unet_segmentation_best.pth')

    # 2. Strict Geometric Augmentations
    train_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 3. Initialize Splits & Datasets
    print("Initializing Data Split...")
    train_imgs, val_imgs, train_masks, val_masks = split_dataset(IMG_DIR, MASK_DIR, test_size=0.10)

    # Train uses stride 128 (50% overlap). Val uses stride 256 (0% overlap).
    train_dataset = SegmentationPatchDataset(IMG_DIR, MASK_DIR, train_imgs, train_masks, 
                                            patch_size=256, stride=128, transform=train_transform)
    val_dataset = SegmentationPatchDataset(IMG_DIR, MASK_DIR, val_imgs, val_masks, 
                                          patch_size=256, stride=256, transform=val_transform)

    print(f"Extracted Patches - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Initialize Pre-trained Model
    print("Initializing U-Net ResNet34 Model...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    num_epochs = 30

    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Save Weights
        if epoch_val_loss < best_val_loss:
            print(f"Validation improved ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving weights...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), SAVE_PATH)