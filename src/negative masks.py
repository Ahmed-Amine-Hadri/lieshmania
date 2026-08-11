import os
import numpy as np
from PIL import Image

def generate_negative_masks(negative_images_dir, output_masks_dir):
    """
    Creates solid black masks for negative images.
    """
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)
        
    # Get all negative images
    image_files = [f for f in os.listdir(negative_images_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))]
    
    print(f"Found {len(image_files)} negative images. Generating masks...")
    
    for img_name in image_files:
        img_path = os.path.join(negative_images_dir, img_name)
        
        # 1. Open the original image just to get its width and height
        with Image.open(img_path) as img:
            w, h = img.size
            
        # 2. Create a NumPy array of all zeros (black) with the same dimensions
        # dtype=np.uint8 ensures it is formatted as an 8-bit grayscale image
        black_mask_array = np.zeros((h, w), dtype=np.uint8)
        
        # 3. Convert the array back into an image
        mask_image = Image.fromarray(black_mask_array, mode='L')
        
        # 4. Save the mask (use .png to avoid compression artifacts on the pure black)
        base_name = os.path.splitext(img_name)[0]
        mask_filename = f"{base_name}.png"
        mask_image.save(os.path.join(output_masks_dir, mask_filename))
        
    print("All negative masks generated successfully!")

# --- How to use it ---
# Set these to your actual folder paths# Update this path to match your exact folder names
NEGATIVE_IMGS_FOLDER = './data/raw/leishmania classifier/negative/'

# Keep this as the output destination
MASKS_OUTPUT_FOLDER = './data/raw/segmentation/masks/'

generate_negative_masks(NEGATIVE_IMGS_FOLDER, MASKS_OUTPUT_FOLDER)