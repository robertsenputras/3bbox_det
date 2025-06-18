import os
import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage import measure

# ── CONFIG ──────────────────────────────────────────────────────────────
data_path = "data/dl_challenge"     # contains subfolders like <hash1>, <hash2>, …
rgb_file = "rgb.jpg"
seg_file = "mask.npy"          # shape: (N_instances, H, W)
output_dir = "coco_dataset"
train_ratio = 0.8  # 80% for training, 20% for validation
min_area = 100  # minimum area in pixels
min_box_size = 10  # minimum bounding box size

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    # Find contours
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    
    for contour in contours:
        # Flip XY to XY coordinates and flatten
        contour = np.flip(contour, axis=1)
        # Approximate contour to reduce number of points
        epsilon = 0.005 * cv2.arcLength(contour.astype(np.float32), True)
        approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)
        if len(approx) > 4:  # Only include polygons with more than 4 points
            polygons.append(approx.flatten().tolist())
    
    return polygons

def verify_mask(mask, min_area=100, min_box_size=10):
    """Verify if a mask is valid"""
    if not np.any(mask):  # Check if mask is empty
        return False
        
    # Check area
    area = np.sum(mask)
    if area < min_area:
        return False
        
    # Check bounding box size
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not (np.any(rows) and np.any(cols)):
        return False
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Check if bounding box is too small
    if (rmax - rmin < min_box_size) or (cmax - cmin < min_box_size):
        return False
        
    return True

def create_coco_annotations(hash_folders, split_name):
    print(f"\nProcessing {split_name} split:")
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "object", "supercategory": "none"}
        ]
    }
    
    image_id = 0
    ann_id = 0
    total_instances = 0
    skipped_instances = 0
    
    for hash_val in tqdm(sorted(hash_folders), desc="Processing images", unit="img"):
        folder = os.path.join(data_path, hash_val)
        if not os.path.isdir(folder):
            continue
        
        # Load image and mask
        src_img_path = os.path.join(folder, rgb_file)
        mask_path = os.path.join(folder, seg_file)
        
        if not (os.path.exists(src_img_path) and os.path.exists(mask_path)):
            tqdm.write(f"⚠️  Skipping {folder}, missing files")
            continue
        
        img = cv2.imread(src_img_path)
        masks = np.load(mask_path)
        
        if img is None or masks.shape[0] == 0:
            tqdm.write(f"⚠️  Skipping {folder}, invalid data")
            continue
        
        H, W = img.shape[:2]
        valid_instances = []
        
        # Process each instance mask
        for inst in range(masks.shape[0]):
            total_instances += 1
            bin_mask = (masks[inst] > 0).astype(np.uint8)
            
            if not verify_mask(bin_mask, min_area, min_box_size):
                skipped_instances += 1
                continue
            
            # Get both RLE and polygon representations
            fortran_mask = np.asfortranarray(bin_mask)
            rle = maskUtils.encode(fortran_mask)
            rle['counts'] = rle['counts'].decode('ascii')
            polygons = mask_to_polygon(bin_mask)
            
            if not polygons:  # Skip if no valid polygons found
                skipped_instances += 1
                continue
            
            # Calculate bounding box and area
            bbox = maskUtils.toBbox(rle).tolist()
            area = float(maskUtils.area(rle))
            
            valid_instances.append({
                "segmentation": polygons,  # Use polygons instead of RLE
                "area": area,
                "bbox": bbox,
                "category_id": 0,
                "iscrowd": 0
            })
        
        if valid_instances:
            image_id += 1
            # Copy image
            new_img_name = f"{image_id:06d}.jpg"
            dst_img_path = os.path.join(output_dir, "images", split_name, new_img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Add image info
            coco["images"].append({
                "id": image_id,
                "file_name": new_img_name,
                "height": H,
                "width": W
            })
            
            # Add annotations
            for instance in valid_instances:
                ann_id += 1
                instance["id"] = ann_id
                instance["image_id"] = image_id
                coco["annotations"].append(instance)
            
            tqdm.write(f"✓ Image {new_img_name}: {len(valid_instances)} valid instances")
        else:
            tqdm.write(f"⚠️  {hash_val}: No valid instances")
    
    # Save annotations
    output_json = os.path.join(output_dir, "annotations", f"instances_{split_name}.json")
    print(f"\nSaving annotations to {output_json}")
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    
    print(f"\nSummary for {split_name}:")
    print(f"- Total instances processed: {total_instances}")
    print(f"- Skipped instances: {skipped_instances}")
    print(f"- Valid instances: {len(coco['annotations'])}")
    print(f"- Images with annotations: {len(coco['images'])}")
    
    return len(coco["images"]), len(coco["annotations"])

# Clean up existing dataset
print("\nCleaning up existing dataset...")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create directory structure
print("Creating COCO dataset structure...")
os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

# Split dataset
print("\nSplitting dataset...")
hash_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
train_folders, val_folders = train_test_split(hash_folders, train_size=train_ratio, random_state=42)

print(f"\nSplit dataset into:")
print(f"- Training: {len(train_folders)} folders")
print(f"- Validation: {len(val_folders)} folders")

# Create datasets
n_train_images, n_train_anns = create_coco_annotations(train_folders, "train")
print(f"\n✅ Train set: {n_train_images} images and {n_train_anns} annotations")

n_val_images, n_val_anns = create_coco_annotations(val_folders, "val")
print(f"\n✅ Validation set: {n_val_images} images and {n_val_anns} annotations")

print("\nDone! Dataset structure:")
print("coco_dataset/")
print("├── images/")
print("│   ├── train/")
print("│   │   └── [images]")
print("│   └── val/")
print("│       └── [images]")
print("└── annotations/")
print("    ├── instances_train.json")
print("    └── instances_val.json")
