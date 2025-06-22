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
from typing import List, Tuple
import colorsys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'frustum_pointnets_pytorch'))

from dataset import FrustumDataset
from frustum_pointnets_pytorch.models.frustum_pointnets_v1 import FrustumPointNetv1
from frustum_pointnets_pytorch.models.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_mean_size_arr
from frustum_pointnets_pytorch.models.model_util import get_box3d_corners_helper as get_box3d_corners

def custom_collate_fn(batch):
    """Custom collate function to handle batches with different numbers of objects
    Args:
        batch: List of dictionaries from __getitem__
    Returns:
        Collated batch with padded tensors
    """
    # Find maximum number of objects in this batch
    max_objects = max([b['point_cloud'].size(0) for b in batch])
    
    # Initialize the collated batch
    collated_batch = {}
    
    # Handle each key in the batch
    for key in batch[0].keys():
        if key == 'num_objects':
            # Just stack the number of objects
            collated_batch[key] = torch.stack([b[key] for b in batch])
            continue
            
        if key == 'one_hot':
            # One hot is already fixed size
            collated_batch[key] = torch.stack([b[key] for b in batch])
            continue
            
        if isinstance(batch[0][key], torch.Tensor):
            # Get the shape of the tensor
            tensor_shape = batch[0][key].shape
            
            # Create padded tensors for each item in batch
            if len(tensor_shape) == 1:  # For 1D tensors like rot_angle
                padded = [torch.cat([b[key], 
                                   torch.zeros(max_objects - len(b[key]), 
                                             dtype=b[key].dtype, 
                                             device=b[key].device)]) 
                         for b in batch]
            elif len(tensor_shape) == 2:  # For 2D tensors
                if key in ['size_residual', 'box3d_center']:  # Special case for Nx3 tensors
                    padded = [torch.cat([b[key], 
                                       torch.zeros(max_objects - b[key].size(0), 3,
                                                 dtype=b[key].dtype,
                                                 device=b[key].device)], 0)
                             for b in batch]
                else:  # For other 2D tensors
                    padded = [torch.cat([b[key],
                                       torch.zeros(max_objects - b[key].size(0),
                                                 b[key].size(1),
                                                 dtype=b[key].dtype,
                                                 device=b[key].device)], 0)
                             for b in batch]
            elif len(tensor_shape) == 3:  # For 3D tensors like point_cloud
                padded = [torch.cat([b[key],
                                   torch.zeros(max_objects - b[key].size(0),
                                             b[key].size(1),
                                             b[key].size(2),
                                             dtype=b[key].dtype,
                                             device=b[key].device)], 0)
                         for b in batch]
                
            # Stack the padded tensors
            collated_batch[key] = torch.stack(padded)
    
    return collated_batch

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

def calculate_rotated_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate 3D IoU between two rotated boxes using corner representation
    Args:
        box1: [x, y, z, l, w, h, heading]
        box2: [x, y, z, l, w, h, heading]
    Returns:
        iou: 3D IoU value considering rotation
    """
    # Convert boxes to corner representation
    center1, size1, heading1 = box1[:3], box1[3:6], box1[6]
    center2, size2, heading2 = box2[:3], box2[3:6], box2[6]
    
    # Convert to torch tensors and add batch dimension
    center1 = torch.from_numpy(center1).float().unsqueeze(0).unsqueeze(0)
    center2 = torch.from_numpy(center2).float().unsqueeze(0).unsqueeze(0)
    size1 = torch.from_numpy(size1).float().unsqueeze(0).unsqueeze(0)
    size2 = torch.from_numpy(size2).float().unsqueeze(0).unsqueeze(0)
    heading1 = torch.tensor(heading1).float().unsqueeze(0).unsqueeze(0)
    heading2 = torch.tensor(heading2).float().unsqueeze(0).unsqueeze(0)
    
    # Get corners
    corners1 = get_box3d_corners(center1, heading1, size1)
    corners2 = get_box3d_corners(center2, heading2, size2)
    corners1 = corners1.squeeze(0).squeeze(0).numpy()  # (8, 3)
    corners2 = corners2.squeeze(0).squeeze(0).numpy()  # (8, 3)
    
    # Project boxes onto XY plane for 2D IoU first
    corners1_xy = corners1[:, :2]  # (8, 2)
    corners2_xy = corners2[:, :2]  # (8, 2)
    
    # Calculate z-axis overlap
    z1_min, z1_max = np.min(corners1[:, 2]), np.max(corners1[:, 2])
    z2_min, z2_max = np.min(corners2[:, 2]), np.max(corners2[:, 2])
    z_intersection = np.maximum(0, np.minimum(z1_max, z2_max) - np.maximum(z1_min, z2_min))
    z_union = (z1_max - z1_min) + (z2_max - z2_min) - z_intersection
    
    # If no z-overlap, return 0
    if z_intersection == 0:
        return 0.0
    
    # Calculate areas using shoelace formula
    def polygon_area(corners):
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        return abs(area) / 2.0
    
    area1 = polygon_area(corners1_xy[:4])  # Use first 4 corners (bottom face)
    area2 = polygon_area(corners2_xy[:4])
    
    # Calculate intersection area using Sutherland-Hodgman algorithm
    def clip_polygon(subject_polygon, clip_polygon):
        def inside(p, cp1, cp2):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
        
        def compute_intersection(p1, p2, cp1, cp2):
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [p1[0] - p2[0], p1[1] - p2[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = p1[0] * p2[1] - p1[1] * p2[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
        
        output_list = subject_polygon
        cp1 = clip_polygon[-1]
        
        for j in range(len(clip_polygon)):
            cp2 = clip_polygon[j]
            input_list = output_list
            output_list = []
            if not input_list:
                break
            s = input_list[-1]
            
            for i in range(len(input_list)):
                e = input_list[i]
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output_list.append(compute_intersection(s, e, cp1, cp2))
                    output_list.append(e)
                elif inside(s, cp1, cp2):
                    output_list.append(compute_intersection(s, e, cp1, cp2))
                s = e
            cp1 = cp2
        return output_list
    
    # Calculate intersection polygon
    intersection_polygon = clip_polygon(corners1_xy[:4].tolist(), corners2_xy[:4].tolist())
    if not intersection_polygon:
        return 0.0
    
    intersection_area = polygon_area(np.array(intersection_polygon))
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    # Final 3D IoU
    iou = (intersection_area * z_intersection) / (union_area * z_union)
    return iou

def nms_3d(boxes: List[np.ndarray], scores: List[float], config: dict) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    """
    Perform 3D Non-Maximum Suppression with multiple thresholds
    Args:
        boxes: List of boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
        config: Configuration dictionary with NMS parameters
    Returns:
        kept_boxes: List of boxes after NMS
        kept_scores: List of scores after NMS
        suppressed_boxes: List of suppressed boxes (for visualization)
        suppressed_scores: List of suppressed scores
    """
    if not boxes:
        return [], [], [], []
    
    # Get NMS parameters from config
    iou_threshold = config.get('nms_threshold', 0.3)
    score_threshold = config.get('score_threshold', 0.1)
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Filter by score threshold first
    score_mask = scores > score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    
    # Sort by score
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    kept_boxes = []
    kept_scores = []
    suppressed_boxes = []
    suppressed_scores = []
    
    while len(boxes) > 0:
        # Keep the highest scoring box
        kept_boxes.append(boxes[0])
        kept_scores.append(scores[0])
        
        if len(boxes) == 1:
            break
        
        # Calculate IoU of the highest scoring box with all other boxes
        ious = [calculate_rotated_3d_iou(boxes[0], box) for box in boxes[1:]]
        
        # Identify boxes to suppress
        mask = np.array(ious) <= iou_threshold
        
        # Store suppressed boxes
        suppressed_boxes.extend(boxes[1:][~mask].tolist())
        suppressed_scores.extend(scores[1:][~mask].tolist())
        
        # Keep boxes below threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
    
    return kept_boxes, kept_scores, suppressed_boxes, suppressed_scores

def get_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV color space
    Args:
        n: Number of colors needed
    Returns:
        colors: List of RGB colors
    """
    # Base colors for better distinction
    base_colors = [
        [1, 0, 0],      # Red
        [0, 1, 0],      # Green
        [0, 0, 1],      # Blue
        [1, 1, 0],      # Yellow
        [1, 0, 1],      # Magenta
        [0, 1, 1],      # Cyan
        [1, 0.5, 0],    # Orange
        [0.5, 0, 1],    # Purple
        [0, 1, 0.5],    # Spring Green
        [1, 0.5, 0.5],  # Pink
        [0.5, 1, 0],    # Lime
        [0.5, 0.5, 1],  # Light Blue
        [1, 0.8, 0],    # Gold
    ]
    
    if n <= len(base_colors):
        return base_colors[:n]
    
    # If we need more colors, generate them using HSV color space
    colors = []
    for i in range(n):
        # Use golden ratio to space the hues evenly
        hue = i * 0.618033988749895 % 1
        # Vary saturation and value slightly to create more distinction
        sat = 0.6 + (i % 3) * 0.2  # Varies between 0.6, 0.8, 1.0
        val = 0.9 + (i % 2) * 0.1  # Varies between 0.9 and 1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(list(rgb))
    
    return colors

def visualize_results(point_clouds, boxes_3d, scores, config):
    """
    Visualize detection results with NMS and distinct colors for each frustum
    Args:
        point_clouds: List of (N, 4) arrays of points with intensity
        boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
        config: Configuration dictionary
    """
    print("\nVisualization Details:")
    print(f"Number of point clouds: {len(point_clouds)}")
    print(f"Number of boxes before NMS: {len(boxes_3d)}")
    
    # Perform NMS
    kept_boxes, kept_scores, suppressed_boxes, suppressed_scores = nms_3d(boxes_3d, scores, config)
    print(f"Number of boxes after NMS: {len(kept_boxes)}")
    print(f"Number of suppressed boxes: {len(suppressed_boxes)}")
    
    # Generate distinct colors for each frustum
    num_frustums = len(point_clouds)
    distinct_colors = get_distinct_colors(num_frustums)
    print(f"Generated {len(distinct_colors)} distinct colors for {num_frustums} frustums")
    
    # Create list of geometries to visualize
    geometries = []
    
    # Add point clouds with distinct colors
    for idx, point_cloud in enumerate(point_clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Use distinct color for each frustum
        base_color = distinct_colors[idx]
        
        if point_cloud.shape[1] > 3:
            # Use intensity to modulate the base color
            intensity = point_cloud[:, 3]
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
            intensity = intensity.reshape(-1, 1)
            
            # Create color array by scaling the base color with intensity
            colors = np.tile(base_color, (len(point_cloud), 1))
            colors = colors * (0.3 + 0.7 * intensity)  # Keep some base color even at low intensity
        else:
            # Use base color directly if no intensity
            colors = np.tile(base_color, (len(point_cloud), 1))
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)
    
    # Add coordinate frame with configured size
    axis_size = config.get('visualization', {}).get('axis_size', 1.0)  # Default to 1.0 if not specified
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0, 0, 0]
    )
    geometries.append(axis)
    
    # Helper function to create box geometry
    def create_box_geometry(box, score, color, is_suppressed=False):
        center = box[:3]
        size = box[3:6]
        heading = box[6]
        
        # Convert to tensor format for corner calculation
        center_tensor = torch.from_numpy(center).float().unsqueeze(0).unsqueeze(0)
        heading_tensor = torch.tensor(heading).float().unsqueeze(0).unsqueeze(0)
        size_tensor = torch.from_numpy(size).float().unsqueeze(0).unsqueeze(0)
        
        # Get corners
        corners = get_box3d_corners(center_tensor, heading_tensor, size_tensor)
        corners = corners.squeeze(0).squeeze(0).numpy()
        
        # Define box edges
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        if is_suppressed:
            # Gray color for suppressed boxes
            colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))]
        else:
            # Use the same color as the corresponding frustum, but modulate by score
            box_color = np.array(color) * (0.3 + 0.7 * score)  # Maintain some color even with low score
            colors = [box_color for _ in range(len(lines))]
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    # Add kept boxes with matching colors
    for idx, (box, score) in enumerate(zip(kept_boxes, kept_scores)):
        # Use the same color as the corresponding frustum
        color = distinct_colors[idx % len(distinct_colors)]
        geometries.append(create_box_geometry(box, score, color, False))
    
    # Add suppressed boxes (in gray)
    if config.get('show_suppressed_boxes', True):
        for box, score in zip(suppressed_boxes, suppressed_scores):
            geometries.append(create_box_geometry(box, score, None, True))
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Detection Results with NMS",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

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
    
    # Get box centers (B, M, 3) -> (M, 3)
    centers = predictions['box3d_center'].detach().cpu().numpy()  # (B, M, 3)
    if centers.ndim == 3:
        centers = centers[0]  # Take first batch since we process one at a time
    print(f"Centers shape after processing: {centers.shape}")
    
    # Get heading information
    heading_scores = torch.softmax(predictions['heading_scores'], dim=-1).detach().cpu().numpy()  # (B, M, NH)
    heading_residuals = predictions['heading_residual'].detach().cpu().numpy()  # (B, M, NH)
    if heading_scores.ndim == 3:
        heading_scores = heading_scores[0]  # (M, NH)
        heading_residuals = heading_residuals[0]  # (M, NH)
    
    print(f"Heading scores shape: {heading_scores.shape}")
    print(f"Heading residuals shape: {heading_residuals.shape}")
    
    heading_class = np.argmax(heading_scores, axis=1)  # (M,)
    
    # Get the residual value corresponding to the predicted heading class
    heading_residuals_for_class = np.array([
        heading_residuals[i, heading_class[i]] 
        for i in range(len(heading_class))
    ])  # (M,)
    
    # Calculate heading angles
    heading_angles = heading_class * (2 * np.pi / NUM_HEADING_BIN) + heading_residuals_for_class  # (M,)
    print(f"Heading angles shape: {heading_angles.shape}")
    
    # Get size information
    size_scores = torch.softmax(predictions['size_scores'], dim=-1).detach().cpu().numpy()  # (B, M, NS)
    size_residuals = predictions['size_residual'].detach().cpu().numpy()  # (B, M, 1, 3)
    if size_scores.ndim == 3:
        size_scores = size_scores[0]  # (M, NS)
        size_residuals = size_residuals[0]  # (M, 1, 3)
    
    # Remove the extra dimension from size_residuals
    size_residuals = np.squeeze(size_residuals, axis=1)  # (M, 3)
    
    size_class = np.zeros(size_scores.shape[0], dtype=np.int32)  # (M,) Always class 0 in new format
    print(f"Size residuals shape after squeeze: {size_residuals.shape}")
    
    # Get sizes by adding residuals to mean size
    mean_sizes = g_mean_size_arr[size_class]  # (M, 3)
    sizes = mean_sizes + size_residuals  # (M, 3)
    print(f"Mean sizes shape: {mean_sizes.shape}")
    print(f"Final sizes shape: {sizes.shape}")
    
    # Get confidence scores from segmentation logits
    seg_logits = predictions['logits'].detach().cpu()  # (B, M, num_points, 2)
    if seg_logits.ndim == 4:
        seg_logits = seg_logits[0]  # (M, num_points, 2)
    seg_probs = torch.softmax(seg_logits, dim=-1)  # (M, num_points, 2)
    scores = seg_probs[..., 1].mean(dim=1).numpy()  # (M,) - mean probability of object class
    print(f"Scores shape: {scores.shape}")
    
    # Combine parameters into boxes
    boxes_3d = []
    print(f"\nDebug shapes before box creation:")
    print(f"centers: {centers.shape}")
    print(f"sizes: {sizes.shape}")
    print(f"heading_angles: {heading_angles.shape}")
    
    for i in range(len(centers)):
        center = centers[i]  # (3,)
        size = sizes[i]      # (3,)
        heading = heading_angles[i]  # scalar
        box = np.concatenate([center, size, [heading]])  # (7,)
        boxes_3d.append(box)
    
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
    """Main testing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test.yaml', help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--output_dir', default='test_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['weights'] = args.weights
    config['visualize'] = args.visualize
    config['output_dir'] = args.output_dir
    
    print(f"\nVisualization enabled: {config['visualize']}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get validation scenes
    val_scenes = get_scene_list(config['data_root'], config['train_val_split'], is_training=False)
    
    # Create dataset and dataloader
    dataset = FrustumDataset(
        data_path=config['data_root'],
        scene_list=val_scenes,
        num_points=config['num_points']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Use batch size 2 to avoid shape issues
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    device = torch.device(config['device'])
    model = FrustumPointNetv1().to(device)
    
    # Load weights - handle both direct weights and checkpoint format
    print(f"\nLoading weights from {config['weights']}...")
    weights = torch.load(config['weights'], map_location=device)
    if 'model_state_dict' in weights:
        print("Loading from checkpoint format...")
        model.load_state_dict(weights['model_state_dict'])
    else:
        print("Loading direct model weights...")
        model.load_state_dict(weights)
    model.eval()
    
    # Testing loop
    print("\nStarting testing...")
    all_boxes = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(dataloader, desc="Testing")):
            # Move data to device
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
            
            # Print shapes of input data for debugging
            print(f"\nBatch {batch_idx} input shapes:")
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
            
            # Process each item in the batch separately to avoid shape issues
            batch_boxes = []
            batch_scores = []
            batch_point_clouds = []
            
            # Get predictions for the first (and only) item in batch
            predictions = model(data_dict)
            
            # Convert predictions to boxes and scores
            item_boxes, item_scores = get_boxes_from_predictions(predictions)
            
            # Store results
            batch_boxes.append(item_boxes)
            batch_scores.append(item_scores)
            
            # Get point cloud data for visualization
            point_cloud = data_dict['point_cloud'][0].cpu().numpy()  # (M, 4, num_points)
            for j in range(point_cloud.shape[0]):  # Loop through objects in the scene
                pc = point_cloud[j]  # (4, num_points)
                pc = pc.transpose(1, 0)  # (num_points, 4)
                batch_point_clouds.append(pc)
            
            # Store all results
            all_boxes.extend([box for boxes in batch_boxes for box in boxes])
            all_scores.extend([score for scores in batch_scores for score in scores])
            
            # Visualize if requested
            if config['visualize']:
                # Visualize all point clouds and boxes together
                all_boxes_batch = [box for boxes in batch_boxes for box in boxes]
                all_scores_batch = [score for scores in batch_scores for score in scores]
                visualize_results(batch_point_clouds, all_boxes_batch, all_scores_batch, config)
    
    # Save results
    results = {
        'boxes_3d': np.array(all_boxes),
        'scores': np.array(all_scores)
    }
    np.save(output_dir / 'results.npy', results)
    print(f"\nResults saved to {output_dir / 'results.npy'}")
    print(f"Total boxes detected: {len(all_boxes)}")
    print(f"Score range: [{min(all_scores):.3f}, {max(all_scores):.3f}]")

if __name__ == '__main__':
    main() 