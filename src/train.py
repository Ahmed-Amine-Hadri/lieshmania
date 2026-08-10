import torch
from torchvision import transforms, datasets

# 1. Augmentation pipeline for TRAINING (applied dynamically per epoch)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),         # Flip horizontally
    transforms.RandomVerticalFlip(p=0.5),           # Flip vertically
    transforms.RandomRotation(degrees=30),         # Rotate up to 30 degrees
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.1
    ),                                              # Simulate camera/lighting changes
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Clean transformation for VALIDATION (NO augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets directly from your processed directory
train_dataset = datasets.ImageFolder(root='data/proceeded/train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='data/proceeded/val', transform=val_transforms)