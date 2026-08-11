import json
import os
from PIL import Image, ImageDraw

def convert_coco_json_to_masks(json_file_path, output_mask_dir):
    """
    Reads a COCO JSON file and draws binary masks (PNGs) based on the polygon coordinates.
    """
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
        
    print(f"Loading JSON file: {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    # Map image IDs to their filenames and dimensions
    images_info = {img['id']: img for img in data['images']}
    
    # Group the annotations (polygons) by the image they belong to
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
    print(f"Found {len(images_info)} images with annotations. Drawing masks...")
    
    # Draw the masks
    for img_id, img_data in images_info.items():
        filename = img_data['file_name']
        width = img_data['width']
        height = img_data['height']
        
        # Create a completely black background image (0 = black)
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # If the image has polygons, draw them in white (255 = white)
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                for segmentation in ann['segmentation']:
                    # COCO stores polygons as a flat list: [x1, y1, x2, y2, ...]
                    # We convert this to a list of tuples for PIL: [(x1, y1), (x2, y2), ...]
                    polygon_points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                    
                    # Draw the filled white polygon
                    draw.polygon(polygon_points, outline=255, fill=255)
                    
        # Save the mask as a PNG using the original photo's name
        base_name = os.path.splitext(filename)[0]
        mask_filename = f"{base_name}.png"
        mask_img.save(os.path.join(output_mask_dir, mask_filename))
        
    print(f"Success! Masks saved to {output_mask_dir}")

# --- Execute the Conversion ---
# 1. Update this to the exact name of your downloaded JSON file
JSON_FILE = './src/labels_my-project-name_2026-08-11-07-09-37.json'
OUTPUT_DIR = './data/raw/segmentation/masks/'

convert_coco_json_to_masks(JSON_FILE, OUTPUT_DIR)