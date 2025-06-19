import torch
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO

from pipeline import FrustumDetectionPipeline
from box_utils import nms_3d

class ObjectDetector3D:
    def __init__(self, 
                 yolo_weights='yolo11m-seg.pt',
                 frustum_weights=None,
                 frustum_version='v1',
                 device='cuda'):
        """
        Initialize the 3D Object Detector
        Args:
            yolo_weights: Path to YOLOv11-seg weights
            frustum_weights: Path to Frustum-PointNet weights
            frustum_version: Version of Frustum-PointNet to use ('v1' or 'v2')
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_weights)
        
        # Initialize Frustum-PointNet pipeline
        self.frustum_pipeline = FrustumDetectionPipeline(
            model_version=frustum_version,
            weights_path=frustum_weights,
            device=device
        )
        
    def get_2d_boxes(self, image):
        """
        Get 2D bounding boxes from YOLOv11-seg
        Args:
            image: (H, W, 3) RGB image
        Returns:
            boxes: (N, 4) array of boxes [x1, y1, x2, y2]
            masks: (N, H, W) array of segmentation masks
            scores: (N,) array of confidence scores
        """
        results = self.yolo_model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        return boxes, masks, scores
    
    def process_depth(self, depth_map, masks):
        """
        Process depth map using segmentation masks
        Args:
            depth_map: (H, W) depth map
            masks: (N, H, W) segmentation masks
        Returns:
            processed_depth: (H, W) processed depth map
        """
        # Initialize processed depth map
        processed_depth = depth_map.copy()
        
        # Apply each mask
        for mask in masks:
            # Get depth values within the mask
            mask_depth = depth_map[mask > 0.5]
            if len(mask_depth) == 0:
                continue
                
            # Simple statistical filtering
            mean_depth = np.mean(mask_depth)
            std_depth = np.std(mask_depth)
            valid_depth = (mask_depth > mean_depth - 2*std_depth) & (mask_depth < mean_depth + 2*std_depth)
            
            # Update depth map
            processed_depth[mask > 0.5] = mask_depth * valid_depth
            
        return processed_depth
    
    def __call__(self, image, depth_map, camera_matrix):
        """
        Run complete 3D object detection pipeline
        Args:
            image: (H, W, 3) RGB image
            depth_map: (H, W) depth map
            camera_matrix: (3, 4) camera projection matrix
        Returns:
            boxes_3d: List of 3D bounding boxes
            scores_3d: List of confidence scores
        """
        # Get 2D boxes and masks from YOLOv11-seg
        boxes_2d, masks, scores_2d = self.get_2d_boxes(image)
        
        # Process depth map using segmentation masks
        processed_depth = self.process_depth(depth_map, masks)
        
        # Run Frustum-PointNet pipeline
        boxes_3d, scores_3d = self.frustum_pipeline(boxes_2d, processed_depth, camera_matrix)
        
        # Combine 2D and 3D scores
        final_scores = scores_2d * scores_3d
        
        # Apply 3D NMS
        keep_indices = nms_3d(boxes_3d, final_scores, iou_threshold=0.5)
        
        filtered_boxes = [boxes_3d[i] for i in keep_indices]
        filtered_scores = final_scores[keep_indices]
        
        return filtered_boxes, filtered_scores

def visualize_results(image, depth_map, boxes_2d, boxes_3d, scores):
    """
    Visualize detection results
    Args:
        image: (H, W, 3) RGB image
        depth_map: (H, W) depth map
        boxes_2d: (N, 4) array of 2D boxes
        boxes_3d: List of 3D boxes
        scores: (N,) array of scores
    Returns:
        vis_image: Visualization image
    """
    # TODO: Implement visualization
    # This should include:
    # 1. 2D boxes on RGB image
    # 2. Projected 3D boxes on RGB image
    # 3. Bird's eye view visualization
    raise NotImplementedError

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--depth', required=True, help='Path to depth map')
    parser.add_argument('--calib', required=True, help='Path to calibration file')
    parser.add_argument('--yolo-weights', default='yolo11m-seg.pt', help='Path to YOLOv11-seg weights')
    parser.add_argument('--frustum-weights', help='Path to Frustum-PointNet weights')
    parser.add_argument('--frustum-version', default='v1', choices=['v1', 'v2'], help='Frustum-PointNet version')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')
    parser.add_argument('--output', help='Path to save visualization')
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector3D(
        yolo_weights=args.yolo_weights,
        frustum_weights=args.frustum_weights,
        frustum_version=args.frustum_version,
        device=args.device
    )
    
    # Load inputs
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(args.depth)  # Assuming .npy format
    camera_matrix = np.loadtxt(args.calib)  # Assuming 3x4 projection matrix
    
    # Run detection
    boxes_3d, scores = detector(image, depth_map, camera_matrix)
    
    # Visualize if output path is provided
    if args.output:
        vis_image = visualize_results(image, depth_map, boxes_2d, boxes_3d, scores)
        cv2.imwrite(args.output, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)) 