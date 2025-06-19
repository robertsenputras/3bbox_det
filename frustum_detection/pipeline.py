import torch
import numpy as np
from pathlib import Path
import sys

# Add Frustum-PointNet to path
sys.path.append(str(Path(__file__).parent.parent / 'frustum_pointnets_pytorch'))

from models.frustum_pointnets_v1 import FrustumPointNetv1
from models.frustum_pointnets_v2 import FrustumPointNetv2

class FrustumDetectionPipeline:
    def __init__(self, model_version='v1', weights_path=None, device='cuda'):
        """
        Initialize the Frustum Detection Pipeline
        Args:
            model_version: 'v1' or 'v2' for different Frustum-PointNet versions
            weights_path: Path to the pretrained weights
            device: Device to run inference on
        """
        self.device = device
        self.model_version = model_version
        
        # Initialize Frustum-PointNet model
        if model_version == 'v1':
            self.model = FrustumPointNetv1()
        else:
            self.model = FrustumPointNetv2()
            
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        self.model = self.model.to(device)
        self.model.eval()

    def project_image_to_rect(self, uv_depth, P):
        """
        Project 2D points to 3D points in camera coordinate
        Args:
            uv_depth: (N, 3) array of point UV and depth
            P: (3, 4) Camera projection matrix
        Returns:
            points: (N, 3) array of points in camera coordinate
        """
        c_u = P[0, 2]
        c_v = P[1, 2]
        f_u = P[0, 0]
        f_v = P[1, 1]
        b_x = P[0, 3] / (-f_u)
        b_y = P[1, 3] / (-f_v)
        
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
        y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
        pts_3d = np.zeros((n, 3))
        pts_3d[:, 0] = x
        pts_3d[:, 1] = y
        pts_3d[:, 2] = uv_depth[:, 2]
        return pts_3d

    def get_frustum_points(self, box_2d, depth_map, P):
        """
        Extract points within the frustum defined by 2D box
        Args:
            box_2d: (4,) array of 2D box coordinates [x1, y1, x2, y2]
            depth_map: (H, W) depth map
            P: (3, 4) Camera projection matrix
        Returns:
            points: (N, 3) array of points in the frustum
        """
        x1, y1, x2, y2 = box_2d
        
        # Get points within 2D box
        v, u = np.meshgrid(range(int(y1), int(y2)), range(int(x1), int(x2)), indexing='ij')
        uv = np.stack([u.flatten(), v.flatten()], axis=1)
        
        # Get corresponding depths
        depths = depth_map[v.flatten(), u.flatten()]
        
        # Create UV-Depth array
        uv_depth = np.concatenate([uv, depths[:, None]], axis=1)
        
        # Project to 3D
        points_3d = self.project_image_to_rect(uv_depth, P)
        return points_3d

    def process_2d_boxes(self, boxes_2d, depth_map, P):
        """
        Process 2D boxes to get frustum point clouds
        Args:
            boxes_2d: (N, 4) array of 2D boxes
            depth_map: (H, W) depth map
            P: (3, 4) Camera projection matrix
        Returns:
            frustum_points: List of (M, 3) arrays of points in each frustum
        """
        frustum_points = []
        for box in boxes_2d:
            points = self.get_frustum_points(box, depth_map, P)
            frustum_points.append(points)
        return frustum_points

    def detect_3d_objects(self, frustum_points):
        """
        Run Frustum-PointNet on point clouds
        Args:
            frustum_points: List of (N, 3) arrays of points in frustums
        Returns:
            boxes_3d: List of 3D bounding boxes
            scores: List of confidence scores
        """
        boxes_3d = []
        scores = []
        
        with torch.no_grad():
            for points in frustum_points:
                # Prepare input for Frustum-PointNet
                points = torch.from_numpy(points).float().to(self.device)
                # Add batch dimension
                points = points.unsqueeze(0)
                
                # Forward pass
                output = self.model(points)
                
                # Process output to get 3D box parameters
                # This will depend on the exact output format of the model
                box_3d = self.process_model_output(output)
                boxes_3d.append(box_3d)
                
        return boxes_3d, scores

    def process_model_output(self, output):
        """
        Process model output to get 3D box parameters
        Args:
            output: Model output
        Returns:
            box_3d: 3D bounding box parameters
        """
        # This needs to be implemented based on the exact output format
        # of the Frustum-PointNet model being used
        raise NotImplementedError

    def nms_3d(self, boxes_3d, scores, iou_threshold=0.5):
        """
        3D Non-Maximum Suppression
        Args:
            boxes_3d: List of 3D boxes
            scores: List of confidence scores
            iou_threshold: IoU threshold for NMS
        Returns:
            filtered_boxes: List of filtered 3D boxes
            filtered_scores: List of filtered scores
        """
        # TODO: Implement 3D NMS
        raise NotImplementedError

    def __call__(self, image_boxes, depth_map, P):
        """
        Run the complete pipeline
        Args:
            image_boxes: (N, 4) array of 2D boxes from image detection
            depth_map: (H, W) depth map
            P: (3, 4) Camera projection matrix
        Returns:
            boxes_3d: List of 3D bounding boxes after NMS
            scores: List of confidence scores
        """
        # Get frustum point clouds for each 2D box
        frustum_points = self.process_2d_boxes(image_boxes, depth_map, P)
        
        # Run Frustum-PointNet on each frustum
        boxes_3d, scores = self.detect_3d_objects(frustum_points)
        
        # Apply 3D NMS
        filtered_boxes, filtered_scores = self.nms_3d(boxes_3d, scores)
        
        return filtered_boxes, filtered_scores 