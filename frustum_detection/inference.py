import torch
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
import open3d as o3d
from tqdm import tqdm

from pipeline import FrustumDetectionPipeline
from frustum_pointnets_pytorch.models.model_util import get_box3d_corners_helper as get_box3d_corners

def visualize_results(image, point_cloud, boxes_3d, scores, min_score=0.5):
    """
    Visualize detection results
    Args:
        image: (H, W, 3) RGB image
        point_cloud: (N, 4) array of points
        boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
        scores: List of confidence scores
        min_score: Minimum confidence score to visualize
    """
    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # Color point cloud by intensity
    colors = np.zeros((len(point_cloud), 3))
    colors[:, 0] = point_cloud[:, 3]  # Map intensity to red channel
    colors = colors / colors.max()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add point cloud to visualization
    vis.add_geometry(pcd)
    
    # Add 3D boxes
    for box, score in zip(boxes_3d, scores):
        if score < min_score:
            continue
            
        # Get box corners
        center = box[:3]
        size = box[3:6]
        heading = box[6]
        corners = get_box3d_corners(center, heading, size)
        
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
        
        # Color based on confidence score (green to red)
        color = np.array([score, 1-score, 0])  # RGB
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        
        vis.add_geometry(line_set)
    
    # Set view control
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0
    
    # Set camera viewpoint
    vc = vis.get_view_control()
    vc.set_zoom(0.8)
    vc.set_lookat([0, 0, 0])
    vc.set_up([0, 0, 1])  # Set Z axis as up direction
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    # Initialize detector
    detector = FrustumDetectionPipeline(
        model_version='v1',
        weights_path='first_try_2006_1344.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize YOLO model for 2D detection
    yolo = YOLO('/home/robert/3bbox_det/ckpt/19June_1026.pt')
    
    # Load sample data
    image_path = 'path/to/image.jpg'
    point_cloud_path = 'path/to/pointcloud.npy'
    calib_path = 'path/to/calib.txt'
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load point cloud
    point_cloud = np.load(point_cloud_path)  # (N, 4) array [x, y, z, intensity]
    
    # Load calibration
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    P = np.array([float(x) for x in calib[2].strip().split(' ')[1:]]).reshape(3, 4)
    
    # Run 2D detection
    results = yolo(image)[0]
    boxes_2d = results.boxes.xyxy.cpu().numpy()  # (N, 4) array [x1, y1, x2, y2]
    
    # Run 3D detection
    boxes_3d, scores = detector(boxes_2d, point_cloud, P)
    
    # Visualize results
    visualize_results(image, point_cloud, boxes_3d, scores)

if __name__ == '__main__':
    main() 