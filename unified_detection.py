import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import os
import sys

# Add paths for frustum_pointnets_pytorch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRUSTUM_DIR = os.path.join(BASE_DIR, 'frustum_detection')
sys.path.append(FRUSTUM_DIR)
sys.path.append(os.path.join(FRUSTUM_DIR, 'frustum_pointnets_pytorch'))

from frustum_detection.pipeline import FrustumDetectionPipeline
from frustum_detection.test_3d import visualize_results
from frustum_detection.dataset import FrustumDataset
from frustum_pointnets_pytorch.models.model_util import NUM_HEADING_BIN, g_mean_size_arr

class UnifiedDetector:
    def __init__(self, config):
        """
        Initialize the unified 2D-3D detector
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.device = config['model']['device']
        
        # Initialize Frustum-PointNet pipeline
        print(f"Initializing Frustum-PointNet {config['model']['frustum_version']}...")
        self.frustum_pipeline = FrustumDetectionPipeline(
            model_version=config['model']['frustum_version'],
            weights_path=config['model']['frustum_weights'],
            device=self.device
        )
        
        # Store visualization parameters
        self.axis_size = config.get('visualization', {}).get('axis_size', 0.1)
    
    def process_dataset_item(self, data_dict):
        """
        Process a single item from the FrustumDataset
        Args:
            data_dict: Dictionary containing dataset item
        Returns:
            boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
            scores: List of confidence scores (all 1.0 for ground truth)
            point_clouds: List of (N, 4) arrays of points with intensity
        """
        # Get point cloud data
        point_cloud = data_dict['point_cloud']  # (N, 4, num_points)
        
        # Convert to numpy if needed
        if isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.cpu().numpy()
        
        # Process each object's point cloud
        point_clouds = []
        if len(point_cloud.shape) == 3:
            for i in range(point_cloud.shape[0]):
                pc = point_cloud[i]  # (4, num_points)
                # Transpose to get (num_points, 4)
                pc = np.transpose(pc, (1, 0))
                
                # Transform coordinates to match visualization system
                transformed_points = np.column_stack([
                    pc[:, 1],  # X = original Y
                    pc[:, 2],  # Y = original Z
                    -pc[:, 0],  # Z = negative original X
                    pc[:, 3]   # Keep intensity as is
                ])
                point_clouds.append(transformed_points)
        else:
            # Single point cloud case
            pc = np.transpose(point_cloud, (1, 0))
            transformed_points = np.column_stack([
                pc[:, 1],  # X = original Y
                pc[:, 2],  # Y = original Z
                -pc[:, 0],  # Z = negative original X
                pc[:, 3]   # Keep intensity as is
            ])
            point_clouds.append(transformed_points)
        
        # Extract box parameters
        centers = data_dict['box3d_center']      # (N, 3)
        size_residuals = data_dict['size_residual']  # (N, 3)
        angle_classes = data_dict['angle_class']     # (N,)
        angle_residuals = data_dict['angle_residual']  # (N,)
        
        # Convert to numpy for processing
        centers = centers.cpu().numpy()
        size_residuals = size_residuals.cpu().numpy()
        angle_classes = angle_classes.cpu().numpy()
        angle_residuals = angle_residuals.cpu().numpy()
        
        boxes_3d = []
        scores = []
        
        # Process each object
        for i in range(len(centers)):
            # Get center and transform it to match visualization system
            center = centers[i]
            transformed_center = np.array([
                center[1],      # X = original Y
                center[2],      # Y = original Z
                -center[0]      # Z = negative original X
            ])
            
            # Get size (add residual to mean size)
            size = size_residuals[i] + g_mean_size_arr[0]  # Using first class mean size
            # Transform size to match visualization system
            transformed_size = np.array([
                size[1],    # length in X = original Y
                size[2],    # width in Y = original Z
                size[0]     # height in Z = original X
            ])
            
            # Get heading angle and transform it
            angle_class = angle_classes[i]
            angle_residual = angle_residuals[i]
            heading_angle = (angle_class * (2 * np.pi / NUM_HEADING_BIN) + 
                           angle_residual)
            # Adjust heading angle for coordinate transformation
            transformed_heading = heading_angle - np.pi/2  # Rotate 90 degrees to match new coordinate system
            
            # Create box parameters [x, y, z, l, w, h, heading]
            box = np.concatenate([transformed_center, transformed_size, [transformed_heading]])
            boxes_3d.append(box)
            scores.append(1.0)  # Ground truth boxes have score 1.0
        
        return boxes_3d, scores, point_clouds

def get_scene_list(data_root, train_val_split=0.8, is_training=True):
    """Get list of scenes for training or validation"""
    data_root = Path(data_root)
    all_scenes = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    # Split into train and validation
    num_scenes = len(all_scenes)
    num_train = int(num_scenes * train_val_split)
    
    # Use same random seed for consistent splits
    np.random.seed(42)
    scene_indices = np.random.permutation(num_scenes)
    train_indices = scene_indices[:num_train]
    val_indices = scene_indices[num_train:]
    
    # Return appropriate scene list
    if is_training:
        return [all_scenes[i] for i in train_indices]
    else:
        return [all_scenes[i] for i in val_indices]

def process_dataset(detector, config):
    """Process all items in the dataset"""
    # Get scene list
    if config['dataset']['scene_list'] is not None:
        # Use specified scene list
        scene_list = [Path(config['dataset']['data_path']) / scene 
                     for scene in config['dataset']['scene_list']]
    else:
        # Use train/val split
        scene_list = get_scene_list(
            config['dataset']['data_path'],
            train_val_split=config['dataset']['train_val_split'],
            is_training=config['dataset']['is_training']
        )
    
    print(f"Processing {len(scene_list)} scenes...")
    
    # Create dataset with scene list
    dataset = FrustumDataset(
        config['dataset']['data_path'],
        scene_list=scene_list,
        num_points=config['dataset']['num_points']
    )
    
    # Create output directory
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each item
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        data_dict = dataset[i]
        
        # Process dataset item
        boxes_3d, scores, point_clouds = detector.process_dataset_item(data_dict)
        
        # Ensure we have a proper point cloud
        if point_clouds[0].size == 0 or len(point_clouds[0].shape) != 2 or point_clouds[0].shape[1] != 4:
            print(f"Warning: Invalid point cloud shape {point_clouds[0].shape} for item {i}, skipping...")
            continue
        
        # Visualize and save results
        output_path = output_dir / f"result_{i:06d}.png"
        try:
            visualize_results(point_clouds, boxes_3d, scores, config)
            print(f"Saved visualization to {output_path}")
        except Exception as e:
            print(f"Error visualizing results for item {i}: {str(e)}")
            print(f"Point cloud shape: {point_clouds[0].shape}")
            print(f"Number of boxes: {len(boxes_3d)}")
            print(f"Number of scores: {len(scores)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Dataset Processing for 3D Object Detection')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = UnifiedDetector(config)
    
    # Process dataset
    if not config['dataset']['data_path']:
        raise ValueError("dataset.data_path must be specified in config")
    process_dataset(detector, config)

if __name__ == '__main__':
    main() 