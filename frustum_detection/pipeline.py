import torch
import numpy as np
from pathlib import Path
import sys
import torch.nn.functional as F

# Add Frustum-PointNet to path
sys.path.append(str(Path(__file__).parent / 'frustum_pointnets_pytorch'))

from frustum_pointnets_pytorch.models.frustum_pointnets_v1 import FrustumPointNetv1
from frustum_pointnets_pytorch.models.model_util import g_mean_size_arr, NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from frustum_pointnets_pytorch.train.provider import compute_box3d_iou
from frustum_pointnets_pytorch.models.model_util import parse_output_to_tensors, get_box3d_corners_helper

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
        
        # Initialize model with 3 channels (x, y, z) to match checkpoint
        self.model = FrustumPointNetv1(n_classes=3, n_channel=3)
            
        if weights_path:
            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location=device)
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        self.model = self.model.to(device)
        self.model.eval()

    def generate_ref_points(self, box_2d, depth_map, P):
        """
        Generate reference points for multi-scale feature extraction
        Args:
            box_2d: (4,) array of 2D box coordinates [x1, y1, x2, y2]
            depth_map: (H, W) depth map
            P: (3, 4) Camera projection matrix
        Returns:
            ref_points: List of reference points at different scales
        """
        cx, cy = (box_2d[0] + box_2d[2]) / 2., (box_2d[1] + box_2d[3]) / 2.
        
        # Define depth ranges for each scale
        z1 = np.array([20, 40, 60, 80])  # Scale 1
        z2 = np.array([10, 30, 50, 70])  # Scale 2
        z3 = np.array([5, 15, 25, 35])   # Scale 3
        z4 = np.array([2, 8, 12, 18])    # Scale 4
        
        # Generate reference points for each scale
        ref_points = []
        for z in [z1, z2, z3, z4]:
            xyz = np.zeros((len(z), 3))
            xyz[:, 0] = cx
            xyz[:, 1] = cy
            xyz[:, 2] = z
            ref_points.append(self.project_image_to_rect(xyz, P))
            
        return ref_points

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

    def get_frustum_points(self, box_2d, point_cloud, P):
        """
        Extract points within the frustum defined by 2D box
        Args:
            box_2d: (4,) array of 2D box coordinates [x1, y1, x2, y2]
            point_cloud: (N, 4) array of points (x, y, z, intensity)
            P: (3, 4) Camera projection matrix
        Returns:
            points: (N, 4) array of points in the frustum
        """
        x1, y1, x2, y2 = box_2d
        
        # Project 3D points to 2D
        pts_3d = point_cloud[:, :3]
        pts_2d = self.project_3d_to_2d(pts_3d, P)
        
        # Get points within 2D box
        mask = (pts_2d[:, 0] >= x1) & (pts_2d[:, 0] < x2) & \
               (pts_2d[:, 1] >= y1) & (pts_2d[:, 1] < y2)
        
        return point_cloud[mask]

    def project_3d_to_2d(self, pts_3d, P):
        """
        Project 3D points to 2D image plane
        Args:
            pts_3d: (N, 3) array of 3D points
            P: (3, 4) Camera projection matrix
        Returns:
            pts_2d: (N, 2) array of 2D points
        """
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
        pts_2d_homo = np.dot(P, pts_3d_homo.T).T
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:]
        return pts_2d

    def detect_3d_objects(self, frustum_points):
        """
        Run Frustum-PointNet on point clouds
        Args:
            frustum_points: List of (N, 4) arrays of points in frustums
        Returns:
            boxes_3d: List of 3D bounding boxes
            scores: List of confidence scores
        """
        boxes_3d = []
        scores = []
        
        with torch.no_grad():
            for points in frustum_points:
                # Only use x, y, z coordinates (drop intensity)
                points = points[:, :3]
                
                # Prepare input for Frustum-PointNet
                points = torch.from_numpy(points).float().to(self.device)
                # Add batch dimension and transpose to (B, C, N)
                points = points.unsqueeze(0).transpose(1, 2)
                
                # Create input dictionary
                data_dict = {
                    'point_cloud': points,
                    'one_hot': torch.ones(1, 3).to(self.device),  # Assuming car class
                }
                
                # Forward pass
                output = self.model(data_dict)
                
                # Process output
                box_3d, score = self.process_model_output(output)
                boxes_3d.append(box_3d)
                scores.append(score)
                
        return boxes_3d, scores

    def process_model_output(self, output):
        """
        Process model output to get 3D box parameters
        Args:
            output: Model output tuple (cls_probs, center_pred, heading_pred, size_pred)
        Returns:
            box_3d: 3D bounding box parameters [x, y, z, l, w, h, heading]
            score: Confidence score
        """
        cls_probs, center_pred, heading_pred, size_pred = output
        
        # Get confidence score
        score = cls_probs[0, 1].item()  # Probability of object class
        
        # Get box center
        center = center_pred[0].cpu().numpy()
        
        # Get heading angle
        heading = heading_pred[0].item()
        
        # Get box size
        size = size_pred[0].cpu().numpy()
        
        # Combine parameters
        box_3d = np.concatenate([center, size, [heading]])
        
        return box_3d, score

    def nms_3d(self, boxes_3d, scores, iou_threshold=0.5):
        """
        3D Non-Maximum Suppression
        Args:
            boxes_3d: List of 3D boxes [x, y, z, l, w, h, heading]
            scores: List of confidence scores
            iou_threshold: IoU threshold for NMS
        Returns:
            filtered_boxes: List of filtered 3D boxes
            filtered_scores: List of filtered scores
        """
        if not boxes_3d:
            return [], []
            
        # Convert to numpy arrays
        boxes_3d = np.array(boxes_3d)
        scores = np.array(scores)
        
        # Sort by score
        order = scores.argsort()[::-1]
        boxes_3d = boxes_3d[order]
        scores = scores[order]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU of the picked box with the rest
            ious = compute_box3d_iou(
                boxes_3d[i:i+1, :3],  # center
                boxes_3d[i:i+1, -1:],  # heading
                boxes_3d[i:i+1, 3:6],  # size
                boxes_3d[1:, :3],
                boxes_3d[1:, -1:],
                boxes_3d[1:, 3:6]
            )[0]
            
            # Remove boxes with IoU > threshold
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]
            
        return boxes_3d[keep], scores[keep]

    def __call__(self, data_dict):
        """
        Run inference on a batch of data
        Args:
            data_dict: Dictionary containing:
                - point_cloud: (B, 4, N) tensor of point clouds
                - one_hot: (B, 3) tensor of one-hot class vectors
                - other fields required by the model
        Returns:
            boxes_3d: List of 3D bounding boxes [x, y, z, l, w, h, heading]
            scores: List of confidence scores
        """
        with torch.no_grad():
            # Forward pass through model
            output = self.model(data_dict)
            
            # Process output to get boxes and scores
            if isinstance(output, tuple):
                losses, metrics = output
                # During training/validation, return empty lists
                return [], []
            else:
                # During inference, process output
                cls_probs, center_preds, heading_preds, size_preds = output
                
                # Get scores from classification probabilities
                scores = cls_probs[:, :, 1]  # Get probability of positive class
                scores = scores.cpu().numpy()
                
                # Convert predictions to boxes
                center_preds = center_preds.cpu().numpy()
                heading_preds = heading_preds.cpu().numpy()
                size_preds = size_preds.cpu().numpy()
                
                boxes_3d = []
                final_scores = []
                
                # Process each batch
                for b in range(cls_probs.shape[0]):
                    # Get indices of objects (where probability > threshold)
                    obj_indices = np.where(scores[b] > 0.5)[0]
                    
                    for idx in obj_indices:
                        # Get box parameters
                        center = center_preds[b, idx]
                        heading = heading_preds[b, idx]
                        size = size_preds[b, idx]
                        
                        # Create box parameters [x, y, z, l, w, h, heading]
                        box = np.concatenate([center, size, [heading]])
                        boxes_3d.append(box)
                        final_scores.append(scores[b, idx])
                
                return boxes_3d, final_scores 