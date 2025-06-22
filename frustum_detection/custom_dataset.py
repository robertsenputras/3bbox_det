import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2
import random
import logging

class DLChallengeDataset(Dataset):
    def __init__(self, data_root, scene_list=None, num_points=1024, rotate_to_center=True, 
                 random_flip=False, random_shift=False, is_training=True):
        """
        Dataset for loading dl_challenge data format
        Args:
            data_root: Path to dataset directory (data/dl_challenge)
            scene_list: List of scene directories to use. If None, use all scenes
            num_points: Number of points to sample per frustum
            rotate_to_center: Whether to rotate frustum to face forward
            random_flip: Whether to randomly flip the frustum
            random_shift: Whether to randomly shift points
            is_training: Whether in training mode (no image loading) or testing mode
        """
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.rotate_to_center = rotate_to_center
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.is_training = is_training
        
        # Get data directories
        if scene_list is None:
            self.data_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        else:
            self.data_dirs = scene_list
            
        logging.info(f"Dataset initialized with {len(self.data_dirs)} scenes in {'training' if is_training else 'testing'} mode")
        
    def __len__(self):
        return len(self.data_dirs)
    
    def random_shift_points(self, points, shift_range=0.1):
        """Randomly shift point cloud"""
        shifts = np.random.uniform(-shift_range, shift_range, size=3)
        points[0] += shifts[0]
        points[1] += shifts[1]
        points[2] += shifts[2]
        return points
    
    def random_flip_points(self, points, boxes):
        """
        Randomly flip point cloud and 3D boxes along x-axis
        Args:
            points: (3, H, W) organized point cloud
            boxes: (N, 8, 3) array of box corners
        """
        if random.random() > 0.5:
            # Flip points
            points[0] = -points[0]  # Flip x coordinates
            
            # Flip boxes
            boxes[..., 0] = -boxes[..., 0]  # Flip x coordinates of all corners
            
            # Reorder box corners to maintain correct orientation
            boxes = boxes[:, [4, 5, 6, 7, 0, 1, 2, 3], :]
            
        return points, boxes
    
    def rotate_points_to_center(self, points, boxes):
        """
        Rotate point cloud to align with box orientation
        Args:
            points: (3, H, W) organized point cloud
            boxes: (N, 8, 3) array of box corners
        """
        rotated_points = points.copy()
        rotated_boxes = boxes.copy()
        
        for i in range(len(boxes)):
            # Compute box center and heading from corners
            center = boxes[i].mean(axis=0)
            front_center = (boxes[i, 0] + boxes[i, 1] + boxes[i, 4] + boxes[i, 5]) / 4
            heading = np.arctan2(front_center[1] - center[1], front_center[0] - center[0])
            
            # Create rotation matrix
            cosa = np.cos(-heading)  # Negative heading to rotate to front
            sina = np.sin(-heading)
            rot_matrix = np.array([
                [cosa, -sina, 0],
                [sina, cosa, 0],
                [0, 0, 1]
            ])
            
            # Reshape points for rotation
            points_reshaped = points.reshape(3, -1)
            
            # Center points
            points_centered = points_reshaped - center.reshape(3, 1)
            
            # Rotate points
            points_rotated = np.dot(rot_matrix, points_centered)
            
            # Move back
            points_final = points_rotated + center.reshape(3, 1)
            rotated_points = points_final.reshape(points.shape)
            
            # Rotate box corners
            corners_centered = boxes[i] - center
            corners_rotated = np.dot(corners_centered, rot_matrix.T)
            rotated_boxes[i] = corners_rotated + center
            
        return rotated_points, rotated_boxes
    
    def get_frustum_points(self, points, boxes, seg_masks):
        """
        Extract points within the frustum of each 3D box
        Args:
            points: (3, H, W) organized point cloud
            boxes: (N, 8, 3) array of box corners
            seg_masks: (N, H, W) segmentation masks
        Returns:
            list of frustum point clouds and their masks
        """
        frustums = []
        frustum_masks = []
        
        # Reshape points for easier processing
        points_reshaped = points.reshape(3, -1).T  # (H*W, 3)
        
        for i, (box, mask) in enumerate(zip(boxes, seg_masks)):
            # Get points within segmentation mask
            mask_flat = mask.reshape(-1)
            valid_points = points_reshaped[mask_flat > 0.5]
            
            # Sample points if needed
            if len(valid_points) > 0:
                if len(valid_points) >= self.num_points:
                    # Randomly sample points
                    choice = np.random.choice(len(valid_points), self.num_points, replace=False)
                    sampled_points = valid_points[choice]
                else:
                    # If we have fewer points, randomly duplicate some points
                    choice = np.random.choice(len(valid_points), self.num_points - len(valid_points))
                    extra_points = valid_points[choice]
                    sampled_points = np.concatenate([valid_points, extra_points], axis=0)
                
                frustums.append(sampled_points)
                frustum_masks.append(np.ones(self.num_points))  # All points are part of the object
        
        return frustums, frustum_masks
    
    def __getitem__(self, idx):
        """Get a single data item"""
        data_dir = self.data_dirs[idx]
        
        # Load data
        points = np.load(data_dir / 'pc.npy')  # (3, H, W) organized point cloud
        boxes = np.load(data_dir / 'bbox3d.npy')  # (N, 8, 3) box corners
        seg_masks = np.load(data_dir / 'mask.npy')  # (N, H, W) segmentation masks
        
        # Data augmentation on full point cloud
        if self.random_shift:
            points = self.random_shift_points(points)
            
        if self.random_flip:
            points, boxes = self.random_flip_points(points, boxes)
            
        if self.rotate_to_center:
            points, boxes = self.rotate_points_to_center(points, boxes)
        
        # Get frustum point clouds
        frustums, frustum_masks = self.get_frustum_points(points, boxes, seg_masks)
        
        # Convert to tensors
        frustums = [torch.from_numpy(f).float() for f in frustums]
        frustum_masks = [torch.from_numpy(m).float() for m in frustum_masks]
        boxes = torch.from_numpy(boxes).float()
        
        # Stack frustums and add intensity channel (zeros)
        if len(frustums) > 0:
            point_clouds = torch.stack(frustums)  # (N, num_points, 3)
            # Add intensity channel
            intensities = torch.zeros(point_clouds.shape[:-1] + (1,))  # (N, num_points, 1)
            point_clouds = torch.cat([point_clouds, intensities], dim=-1)  # (N, num_points, 4)
            # Transpose to match model's expected format
            point_clouds = point_clouds.transpose(2, 1)  # (N, 4, num_points)
        else:
            # Create dummy point cloud if no frustums
            point_clouds = torch.zeros((1, 4, self.num_points), dtype=torch.float32)
            frustum_masks = [torch.zeros(self.num_points, dtype=torch.float32)]
        
        # Create return dictionary
        data_dict = {
            'point_cloud': point_clouds,  # (N, 4, num_points)
            'seg': torch.stack(frustum_masks) if len(frustum_masks) > 0 else torch.zeros((1, self.num_points)),  # (N, num_points)
            'box3d_center': boxes.mean(dim=1),  # (N, 3)
            'heading_class': torch.zeros(len(boxes), dtype=torch.long),  # (N,)
            'heading_residual': torch.zeros(len(boxes)),  # (N,)
            'size_class': torch.zeros(len(boxes), dtype=torch.long),  # (N,)
            'size_residual': torch.zeros((len(boxes), 3)),  # (N, 3)
            'rot_angle': torch.zeros(len(boxes)),  # (N,)
            'scene_id': data_dir.name,
            'one_hot': torch.tensor([1, 0, 0], dtype=torch.float32)  # 3D one-hot vector
        }
        
        # Only load image in testing mode
        if not self.is_training:
            # Load image using cv2 and convert from BGR to RGB
            image = cv2.imread(str(data_dir / 'rgb.jpg'))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # If image loading fails, create a dummy image
                H, W = points.shape[1:]
                image = np.zeros((H, W, 3), dtype=np.uint8)
            
            # Convert image to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            data_dict['image'] = image  # (3, H, W) tensor
        
        return data_dict 