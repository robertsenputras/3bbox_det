import os
import shutil
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import random

# ── CONFIG ──────────────────────────────────────────────────────────────
data_path = "data/dl_challenge"     # contains subfolders like <hash1>, <hash2>, …
rgb_file = "rgb.jpg"               # image file name in each hash folder
seg_file = "mask.npy"              # mask file name in each hash folder (shape: (N_instances, H, W))
output_dir = "yolo_dataset"
train_ratio = 0.8                  # 80% for training, 20% for validation
min_area = 100                     # minimum area in pixels
min_box_size = 10                  # minimum bounding box size

def verify_mask(mask):
    """Verify that the mask is valid and contains instances."""
    if mask is None or mask.shape[0] == 0:
        return False
    
    # Check if any instance mask contains non-zero pixels
    return np.any(mask > 0)

def mask_to_segments(mask):
    """Convert binary mask to normalized segment coordinates."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        # Simplify contour to reduce number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:  # Only add polygons with at least 3 points
            # Convert to normalized coordinates
            points = approx.reshape(-1, 2)
            segments.append(points)
    
    return segments

def create_yolo_dataset(data_dir="data/dl_challenge", output_dir="yolo_dataset", train_ratio=0.8):
    """Create YOLO format dataset with images and labels."""
    # Create directory structure
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Get all hash folders
    hash_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    # Shuffle and split
    random.shuffle(hash_folders)
    split_idx = int(len(hash_folders) * train_ratio)
    train_folders = hash_folders[:split_idx]
    val_folders = hash_folders[split_idx:]
    
    def process_split(folders, split):
        print(f"Processing {split} split...")
        processed_images = 0
        
        for hash_folder in tqdm(folders, desc=f"{split} folders"):
            folder_path = os.path.join(data_dir, hash_folder)
            img_path = os.path.join(folder_path, rgb_file)
            mask_path = os.path.join(folder_path, seg_file)
            
            # Skip if files don't exist
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                continue
            
            # Read image and mask
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            try:
                masks = np.load(mask_path)
            except:
                continue
            
            # Verify mask
            if not verify_mask(masks):
                continue
            
            h, w = img.shape[:2]
            processed_images += 1
            
            # Copy image with a new name
            new_img_name = f"{processed_images:06d}.jpg"
            dst_img_path = os.path.join(output_dir, split, "images", new_img_name)
            shutil.copy2(img_path, dst_img_path)
            
            # Create label file
            label_path = os.path.join(output_dir, split, "labels", f"{processed_images:06d}.txt")
            
            with open(label_path, 'w') as f:
                # Process each instance mask
                for inst_mask in masks:
                    bin_mask = (inst_mask > 0).astype(np.uint8)
                    segments = mask_to_segments(bin_mask)
                    
                    if not segments:
                        continue
                    
                    for segment in segments:
                        # Normalize coordinates
                        segment = segment.astype(float)
                        segment[:, 0] /= w
                        segment[:, 1] /= h
                        
                        # Format: class_id x1 y1 x2 y2 ... xn yn
                        coords = ' '.join([f"{x:.6f}" for x in segment.reshape(-1)])
                        f.write(f"0 {coords}\n")
        
        return processed_images
    
    # Process train and validation splits
    n_train = process_split(train_folders, "train")
    n_val = process_split(val_folders, "val")
    
    print(f"\nDataset created successfully in {output_dir}")
    print(f"Training images: {n_train}")
    print(f"Validation images: {n_val}")

if __name__ == "__main__":
    create_yolo_dataset() 