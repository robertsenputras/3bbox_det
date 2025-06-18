import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import random
from tqdm import tqdm

def visualize_coco_dataset(coco_root, split='val', num_samples=5):
    """
    Visualize COCO dataset annotations
    Args:
        coco_root: Path to COCO dataset root
        split: 'train' or 'val'
        num_samples: Number of random images to visualize
    """
    # Initialize COCO API
    ann_file = os.path.join(coco_root, 'annotations', f'instances_{split}.json')
    coco = COCO(ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Randomly sample images if num_samples is less than total images
    if num_samples < len(img_ids):
        img_ids = random.sample(img_ids, num_samples)
    
    # Create output directory for visualization
    output_dir = os.path.join(coco_root, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nVisualizing {len(img_ids)} images from {split} set...")
    
    # Process each image
    for img_id in tqdm(img_ids):
        # Load image info and image
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(coco_root, 'images', split, img_info['file_name'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Draw each annotation
        for ann in anns:
            # Get bbox coordinates
            bbox = ann['bbox']
            x, y, w, h = [int(b) for b in bbox]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color for bbox
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw segmentation mask
            if 'segmentation' in ann:
                # Handle RLE
                if isinstance(ann['segmentation'], dict):
                    rle = ann['segmentation']
                    mask = maskUtils.decode(rle)
                    mask = mask.astype(np.uint8) * 255
                    
                    # Apply mask overlay
                    mask_color = vis_img.copy()
                    mask_color[mask > 0] = [0, 0, 255]  # Red color for mask
                    vis_img = cv2.addWeighted(vis_img, 0.7, mask_color, 0.3, 0)
            
            # Add annotation ID
            cv2.putText(vis_img, f"ID: {ann['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{split}_{img_info['file_name']}")
        cv2.imwrite(output_path, vis_img)
        
        # Display image dimensions and annotation count
        print(f"\nImage: {img_info['file_name']}")
        print(f"Dimensions: {img.shape}")
        print(f"Number of annotations: {len(anns)}")
        
        # Display bbox coordinates for each annotation
        for ann in anns:
            bbox = ann['bbox']
            print(f"Annotation ID {ann['id']}: bbox = {bbox}")

if __name__ == '__main__':
    coco_root = 'coco_dataset'
    
    # Visualize both train and val sets
    for split in ['train', 'val']:
        visualize_coco_dataset(coco_root, split=split, num_samples=5)
        print(f"\nVisualization complete for {split} set!")
        print(f"Check {coco_root}/visualization/ for the output images.") 