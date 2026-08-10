import os
import random
from PIL import Image, ImageOps

def resize_and_pad(image, target_size=(224, 224)):
    """
    Redimensionne une image à target_size en conservant son ratio initial,
    puis ajoute un padding (bandes neutres) si nécessaire pour éviter la déformation
    et gérer les images de basse résolution sans perte.
    """
    # Conserver le ratio d'origine
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Créer une image de fond neutre (gris moyen ou noir) et y centrer l'image
    delta_w = target_size[0] - image.size[0]
    delta_h = target_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # Appliquer un padding avec des pixels de couleur neutre (ex: noir)
    return ImageOps.expand(image, padding, fill=(0, 0, 0))

def process_and_split_dataset(source_folder, target_base_folder, class_label, source_id, train_ratio=0.7, val_ratio=0.15):
    train_dir = os.path.join(target_base_folder, 'train', class_label)
    val_dir = os.path.join(target_base_folder, 'val', class_label)
    test_dir = os.path.join(target_base_folder, 'test', class_label)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_exts)]
    
    if not images:
        print(f"⚠️ Warning: No images found in {source_folder}")
        return

    random.seed(42)
    random.shuffle(images)
    
    n = len(images)
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))
    
    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    def process_and_save(file_list, destination_dir):
        for i, filename in enumerate(file_list):
            src_path = os.path.join(source_folder, filename)
            new_filename = f"{class_label}_{source_id}_{i+1:03d}.jpg"
            dest_path = os.path.join(destination_dir, new_filename)
            
            try:
                with Image.open(src_path) as img:
                    img = img.convert('RGB')
                    # Utilisation du padding intelligent au lieu d'un crop strict
                    processed_img = resize_and_pad(img, target_size=(224, 224))
                    processed_img.save(dest_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Skipping {filename} due to loading error: {e}")

    process_and_save(train_images, train_dir)
    process_and_save(val_images, val_dir)
    process_and_save(test_images, test_dir)
    
    print(f"✅ Processed '{class_label}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test.")

RAW_NEGATIVE_DIR = "data/raw/leishmania classifier/negative"
RAW_POSITIVE_DIR = "data/raw/leishmania classifier/positive"
PROCESSED_DATA_DIR = "data/proceeded"

if __name__ == "__main__":
    print("Starting robust data preprocessing and splitting...")
    process_and_split_dataset(RAW_NEGATIVE_DIR, PROCESSED_DATA_DIR, "negative", "isic", 0.7, 0.15)
    process_and_split_dataset(RAW_POSITIVE_DIR, PROCESSED_DATA_DIR, "positive", "mixed", 0.7, 0.15)
    print("Data processing complete!")