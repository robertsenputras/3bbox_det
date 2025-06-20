import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
import yaml
import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'frustum_pointnets_pytorch'))

from dataset import FrustumDataset
from frustum_pointnets_pytorch.models.frustum_pointnets_v1 import FrustumPointNetv1
from frustum_pointnets_pytorch.models.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_mean_size_arr
from frustum_pointnets_pytorch.models.model_util import get_box3d_corners_helper as get_box3d_corners

def get_scene_list(data_root, train_val_split=0.8, is_training=False):
    """Get list of scenes for training or validation"""
    data_root = Path(data_root)
    all_scenes = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    # Split into train and validation
    num_scenes = len(all_scenes)
    num_train = int(num_scenes * train_val_split)
    
    # Use same random seed as training for consistent splits
    np.random.seed(42)
    scene_indices = np.random.permutation(num_scenes)
    train_indices = scene_indices[:num_train]
    val_indices = scene_indices[num_train:]
    
    # Return appropriate scene list
    if is_training:
        return [all_scenes[i] for i in train_indices]
    else:
        return [all_scenes[i] for i in val_indices]

def visualize_input_pointcloud(point_cloud):
    """
    Visualize the input point cloud
    Args:
        point_cloud: (N, 4) array of points with intensity
    """
    print("\nInput Point Cloud Visualization:")
    print(f"Point cloud shape: {point_cloud.shape}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Use XYZ coordinates
    
    # Color point cloud by intensity
    if point_cloud.shape[1] > 3:
        colors = np.zeros((len(point_cloud), 3))
        colors[:, 0] = point_cloud[:, 3]  # Map intensity to red channel
        colors = colors / (colors.max() + 1e-6)  # Normalize, avoid division by zero
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,      # 1 meter scale
        origin=[0, 0, 0]  # at origin
    )

    # Visualize point cloud with axis
    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="Input Point Cloud",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

def visualize_results(point_cloud, boxes_3d, scores, config):
    """
    Visualize detection results
    Args:
        point_cloud: (N, 4) array of points
        boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
        config: Configuration dictionary
    """
    print("\nDetection Results Visualization:")
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Number of boxes: {len(boxes_3d)}")
    print(f"Scores: {scores}")
    print(f"First box: {boxes_3d[0] if len(boxes_3d) > 0 else 'No boxes'}")
    
    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detection Results")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # Color point cloud by intensity if available
    if point_cloud.shape[1] > 3:
        colors = np.zeros((len(point_cloud), 3))
        colors[:, 0] = point_cloud[:, 3]  # Map intensity to red channel
        colors = colors / colors.max()
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add point cloud to visualization
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # 1 meter size
        origin=[0, 0, 0]  # at origin
    )
    vis.add_geometry(coord_frame)
    
    # Add 3D boxes
    boxes_added = 0

    for box, score in zip(boxes_3d, scores):
        if score < config.get('min_score', 0.5):  # Add default threshold if not in config
            continue
        
        # Get box corners
        center = box[:3]
        size = box[3:6]
        heading = box[6]
        corners = get_box3d_corners(center, heading, size)
        corners = corners.reshape(-1, 3)  # Reshape to (8, 3) for visualization
        
        # Define box edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]
        
        # Create line set for box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set all lines to red color
        colors = [[1, 0, 0] for _ in range(len(lines))]  # All lines red
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        vis.add_geometry(line_set)
        boxes_added += 1
    
    print(f"Total boxes added to visualization: {boxes_added}")
    
    # Set view control and render options
    opt = vis.get_render_option()
    opt.background_color = np.array(config.get('background_color', [0.1, 0.1, 0.1]))
    opt.point_size = config.get('point_size', 2.0)
    opt.line_width = config.get('line_width', 5.0)  # Thicker lines like in bbox_pred_viz
    
    # Set camera viewpoint
    vc = vis.get_view_control()
    vc.set_zoom(0.8)
    vc.set_lookat([0, 0, 0])
    vc.set_up([0, 0, 1])  # Set Z axis as up direction
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def get_boxes_from_predictions(predictions):
    """
    Convert model predictions to 3D boxes and scores
    Args:
        predictions: dict containing model predictions
    Returns:
        boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
    """
    print("\nDebug predictions:")
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape {v.shape}, range [{v.min().item():.3f}, {v.max().item():.3f}]")
    
    # Get box centers
    centers = predictions['box3d_center'].detach().cpu().numpy()  # (bs, 3)
    
    # Get heading information
    heading_scores = torch.softmax(predictions['heading_scores'], dim=1).detach().cpu().numpy()  # (bs, NH)
    heading_residuals = predictions['heading_residual'].detach().cpu().numpy()  # (bs, NH)
    heading_class = np.argmax(heading_scores, axis=1)  # (bs,)
    heading_angles = heading_class * (2 * np.pi / NUM_HEADING_BIN) + \
                    np.array([heading_residuals[i, heading_class[i]] for i in range(len(heading_class))])  # (bs,)
    
    # Get size information
    size_scores = torch.softmax(predictions['size_scores'], dim=1).detach().cpu().numpy()  # (bs, NS)
    size_residuals = predictions['size_residual'].detach().cpu().numpy()  # (bs, NS, 3)
    size_class = np.argmax(size_scores, axis=1)  # (bs,)
    
    # Get predicted sizes
    mean_sizes = g_mean_size_arr[size_class]  # (bs, 3)
    size_residuals_for_class = np.array([size_residuals[i, size_class[i]] for i in range(len(size_class))])  # (bs, 3)
    sizes = mean_sizes + size_residuals_for_class  # (bs, 3)
    
    print("\nDebug box calculations:")
    print(f"Centers shape: {centers.shape}, range [{centers.min():.3f}, {centers.max():.3f}]")
    print(f"Sizes shape: {sizes.shape}, range [{sizes.min():.3f}, {sizes.max():.3f}]")
    print(f"Heading angles shape: {heading_angles.shape}, range [{heading_angles.min():.3f}, {heading_angles.max():.3f}]")
    
    # Combine into boxes_3d format [x, y, z, l, w, h, heading]
    boxes_3d = np.concatenate([centers, sizes, heading_angles[:, np.newaxis]], axis=1)  # (bs, 7)
    
    # Get confidence scores from segmentation logits
    seg_scores = torch.softmax(predictions['logits'], dim=2)  # [bs, N, 2]
    seg_conf = seg_scores[:, :, 1].mean(dim=1).detach().cpu().numpy()  # Average foreground score
    
    # Get heading confidence (max of softmaxed scores)
    heading_conf = np.max(heading_scores, axis=1)
    
    # Get size confidence (max of softmaxed scores)
    size_conf = np.max(size_scores, axis=1)
    
    # Combine scores - using geometric mean instead of minimum
    scores = np.power(seg_conf * heading_conf * size_conf, 1/3)
    
    print(f"\nConfidence Scores Breakdown:")
    print(f"Segmentation confidence: {seg_conf}")
    print(f"Heading confidence: {heading_conf}")
    print(f"Size confidence: {size_conf}")
    print(f"Final combined score: {scores}")
    
    print(f"\nFinal outputs:")
    print(f"Boxes shape: {boxes_3d.shape}")
    print(f"Scores shape: {scores.shape}, range [{scores.min():.3f}, {scores.max():.3f}]")
    
    return boxes_3d, scores

def visualize_combined(input_point_cloud, frustum_point_cloud, boxes_3d, scores, config):
    """
    Visualize both input point cloud and detection results in one window
    Args:
        input_point_cloud: (N, 4) array of input points with intensity
        frustum_point_cloud: (M, 4) array of frustum points with intensity
        boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
        config: Configuration dictionary
    """
    print("\nCombined Visualization:")
    print(f"Input point cloud shape: {input_point_cloud.shape}")
    print(f"Frustum point cloud shape: {frustum_point_cloud.shape}")
    print(f"Number of boxes: {len(boxes_3d)}")
    
    # Print some statistics to debug coordinate systems
    print("\nPoint Cloud Statistics:")
    print(f"Input PC range - X: [{input_point_cloud[:,0].min():.3f}, {input_point_cloud[:,0].max():.3f}]")
    print(f"Input PC range - Y: [{input_point_cloud[:,1].min():.3f}, {input_point_cloud[:,1].max():.3f}]")
    print(f"Input PC range - Z: [{input_point_cloud[:,2].min():.3f}, {input_point_cloud[:,2].max():.3f}]")
    
    if len(boxes_3d) > 0:
        print("\nFirst Box Statistics:")
        print(f"Center: {boxes_3d[0][:3]}")
        print(f"Size: {boxes_3d[0][3:6]}")
        print(f"Heading: {boxes_3d[0][6]}")
    
    # Create input point cloud (blue)
    input_pcd = o3d.geometry.PointCloud()
    # Transform coordinates to match the visualization system
    input_points_transformed = input_point_cloud[:, :3].copy()
    # Rotate coordinates to match the visualization system
    # X -> -Z (blue points down)
    # Y -> X (red points up-right)
    # Z -> Y (green points right)
    input_points_transformed = np.column_stack([
        input_point_cloud[:, 1],  # X = original Y
        input_point_cloud[:, 2],  # Y = original Z
        -input_point_cloud[:, 0]  # Z = negative original X
    ])
    input_pcd.points = o3d.utility.Vector3dVector(input_points_transformed)
    input_pcd.paint_uniform_color([0, 0, 1])  # Blue for input point cloud
    
    # Create frustum point cloud (green)
    frustum_pcd = o3d.geometry.PointCloud()
    # Transform coordinates for frustum point cloud
    frustum_points_transformed = np.column_stack([
        frustum_point_cloud[:, 1],  # X = original Y
        frustum_point_cloud[:, 2],  # Y = original Z
        -frustum_point_cloud[:, 0]  # Z = negative original X
    ])
    frustum_pcd.points = o3d.utility.Vector3dVector(frustum_points_transformed)
    frustum_pcd.paint_uniform_color([0, 1, 0])  # Green for frustum point cloud
    
    # Create coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,      # 1 meter scale
        origin=[0, 0, 0]  # at origin
    )
    
    # Create list of geometries to visualize
    geometries = [input_pcd, frustum_pcd, axis]
    
    # Add 3D boxes
    boxes_added = 0
    for box, score in zip(boxes_3d, scores):
        if score < config.get('min_score', 0.5):  # Add default threshold if not in config
            continue
        
        # Get box corners
        center = box[:3]
        size = box[3:6]
        heading = box[6]
        corners = get_box3d_corners(center, heading, size)
        corners = corners.reshape(-1, 3)  # Reshape to (8, 3) for visualization
        
        print(f"\nBox {boxes_added} corners:")
        print(f"Corner ranges - X: [{corners[:,0].min():.3f}, {corners[:,0].max():.3f}]")
        print(f"Corner ranges - Y: [{corners[:,1].min():.3f}, {corners[:,1].max():.3f}]")
        print(f"Corner ranges - Z: [{corners[:,2].min():.3f}, {corners[:,2].max():.3f}]")
        
        # Define box edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]
        
        # Create line set for box
        line_set = o3d.geometry.LineSet()
        # Transform coordinates for box corners
        corners_transformed = np.column_stack([
            corners[:, 1],  # X = original Y
            corners[:, 2],  # Y = original Z
            -corners[:, 0]  # Z = negative original X
        ])
        line_set.points = o3d.utility.Vector3dVector(corners_transformed)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set all lines to red color
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red for boxes
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        geometries.append(line_set)
        boxes_added += 1
    
    print(f"\nTotal boxes added to visualization: {boxes_added}")
    
    # Visualize everything
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Combined Visualization",
        width=1280,
        height=960,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get validation scene list
    val_scenes = get_scene_list(config['data_root'], config['train_val_split'], is_training=False)
    print(f"Found {len(val_scenes)} validation scenes")
    
    # Initialize detector
    detector = FrustumPointNetv1(
        n_classes=config.get('num_classes', 3),
        n_channel=config.get('num_channels', 3)
    ).to(config['device'])
    
    # Load weights
    if config['weights_path']:
        weights = torch.load(config['weights_path'], map_location=config['device'])
        if 'model_state_dict' in weights:
            detector.load_state_dict(weights['model_state_dict'])
        else:
            detector.load_state_dict(weights)  # Handle case where weights are saved directly
    detector.eval()
    
    # Create dataset and dataloader
    dataset = FrustumDataset(
        data_path=config['data_root'],
        scene_list=val_scenes,
        num_points=config['num_points']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            # Move data to device
            data_dict = {k: v.to(config['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Get input point cloud
            input_points = data_dict['point_cloud'][0].cpu().numpy().transpose(1, 0)  # (N, 4)
            
            # Run detection
            predictions = detector(data_dict)
            
            # Convert predictions to boxes and scores
            boxes_3d, scores = get_boxes_from_predictions(predictions)
            
            # Get frustum point cloud
            frustum_points = data_dict['point_cloud'][0].cpu().numpy().transpose(1, 0)  # (N, 4)
            
            # Visualize combined results
            visualize_combined(input_points, frustum_points, boxes_3d, scores, config)
            # raise Exception("Stop here")
            # Break after first batch if in debug mode
            if config.get('debug', False):
                break

if __name__ == '__main__':
    main() 