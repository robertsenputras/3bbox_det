import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from ultralytics import YOLO

from pipeline import FrustumDetectionPipeline

def create_frustum_data(image_path, depth_path, calib_path, output_dir, yolo_weights):
    """
    Create frustum point cloud data from image and depth
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map
        calib_path: Path to calibration file
        output_dir: Output directory for frustum data
        yolo_weights: Path to YOLOv11-seg weights
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    yolo_model = YOLO(yolo_weights)
    
    # Load data
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(depth_path)
    camera_matrix = np.loadtxt(calib_path)
    
    # Get 2D detections
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.data.cpu().numpy()
    
    # Initialize Frustum-PointNet pipeline for point cloud processing
    frustum_pipeline = FrustumDetectionPipeline(weights_path=None)
    
    # Process each detection
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # Get frustum points
        points = frustum_pipeline.get_frustum_points(box, depth_map, camera_matrix)
        
        # Get points mask from segmentation
        points_mask = mask[points[:, 1].astype(int), points[:, 0].astype(int)]
        
        # Compute 3D box parameters
        # For training data, you need ground truth 3D boxes
        # This is just a placeholder - you need to replace with actual ground truth
        center = np.mean(points[points_mask > 0.5], axis=0)
        size = np.array([2.0, 2.0, 4.0])  # Default car size
        heading = 0.0  # Default heading
        box_params = np.concatenate([center, size, [heading]])
        
        # Save data
        base_name = output_dir / f"{image_path.stem}_{i}"
        np.save(f"{base_name}_frustum.npy", points)
        np.save(f"{base_name}_box.npy", box_params)
        np.save(f"{base_name}_mask.npy", points_mask)

def prepare_dataset(data_root, output_root, yolo_weights, split_ratio=0.8):
    """
    Prepare complete dataset
    Args:
        data_root: Root directory containing images, depth maps, and calibration
        output_root: Root directory for output
        yolo_weights: Path to YOLOv11-seg weights
        split_ratio: Train/val split ratio
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # Create output directories
    train_dir = output_root / 'train'
    val_dir = output_root / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(data_root.glob('images/*.jpg'))
    num_train = int(len(image_files) * split_ratio)
    
    # Process training data
    print("Processing training data...")
    for image_file in tqdm(image_files[:num_train]):
        depth_file = data_root / 'depth' / f"{image_file.stem}.npy"
        calib_file = data_root / 'calib' / f"{image_file.stem}.txt"
        create_frustum_data(image_file, depth_file, calib_file, train_dir, yolo_weights)
    
    # Process validation data
    print("Processing validation data...")
    for image_file in tqdm(image_files[num_train:]):
        depth_file = data_root / 'depth' / f"{image_file.stem}.npy"
        calib_file = data_root / 'calib' / f"{image_file.stem}.txt"
        create_frustum_data(image_file, depth_file, calib_file, val_dir, yolo_weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Root directory of raw data')
    parser.add_argument('--output_root', required=True, help='Root directory for processed data')
    parser.add_argument('--yolo_weights', default='yolo11m-seg.pt', help='Path to YOLOv11-seg weights')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/val split ratio')
    args = parser.parse_args()
    
    prepare_dataset(args.data_root, args.output_root, args.yolo_weights, args.split_ratio) 