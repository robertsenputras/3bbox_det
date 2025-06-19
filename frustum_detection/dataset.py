import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
from frustum_pointnets_pytorch.models.model_util import NUM_HEADING_BIN, g_mean_size_arr

class FrustumDataset(Dataset):
    def __init__(self, data_path, scene_list=None, rotate_to_center=True, random_flip=False, random_shift=False, num_points=1024, max_objects=8):
        """
        Dataset for Frustum-PointNet training
        Args:
            data_path: Path to dataset directory
            scene_list: Optional list of scene directories to use. If None, uses all scenes.
            rotate_to_center: Whether to rotate frustum to face forward
            random_flip: Whether to randomly flip the frustum
            random_shift: Whether to randomly shift points
            num_points: Number of points to sample (default: 1024 as in original implementation)
            max_objects: Maximum number of objects to handle (for padding)
        """
        self.data_path = Path(data_path)
        self.rotate_to_center = rotate_to_center
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.num_points = num_points
        self.max_objects = max_objects
        
        # Constants from the original implementation
        self.num_heading_bin = 12
        self.num_size_cluster = 8
        
        # Load data paths
        if scene_list is None:
            # Use all scenes in data_path
            self.pcl_files = sorted(self.data_path.glob('*pc.npy'))  # (3,H,W) point clouds
            self.box_files = sorted(self.data_path.glob('*bbox3d.npy'))  # (N,8,3) boxes
            self.seg_files = sorted(self.data_path.glob('*mask.npy'))   # (N,H,W) segmentation masks
        else:
            # Use only specified scenes
            self.pcl_files = []
            self.box_files = []
            self.seg_files = []
            for scene in scene_list:
                self.pcl_files.extend(sorted(Path(scene).glob('*pc.npy')))
                self.box_files.extend(sorted(Path(scene).glob('*bbox3d.npy')))
                self.seg_files.extend(sorted(Path(scene).glob('*mask.npy')))
        
        assert len(self.pcl_files) == len(self.box_files) == len(self.seg_files), \
            "Number of point cloud, box, and segmentation files must match"
            
    def __len__(self):
        return len(self.pcl_files)
    
    def corners_to_params(self, corners):
        """Convert 8 corners (8,3) to box parameters (center[3], size[3], heading[1])"""
        # Calculate center as mean of corners
        center = np.mean(corners, axis=0)
        # print(center)
        # print(corners)
        # Calculate size as max - min along each axis
        size = np.max(corners, axis=0) - np.min(corners, axis=0)
        
        # Calculate heading angle from front face
        front_center = np.mean(corners[:4], axis=0)  # assuming first 4 corners are front face
        direction = front_center - center
        heading = np.arctan2(direction[1], direction[0])  # angle from x-axis
        
        # Combine parameters
        params = np.concatenate([center, size, [heading]])  # shape: (7,)
        return params
    
    def random_shift_points(self, points, shift_range=0.1):
        """Randomly shift point cloud"""
        shifts = np.random.uniform(-shift_range, shift_range, size=3)
        points[0] += shifts[0]  # shift x coordinates
        points[1] += shifts[1]  # shift y coordinates
        points[2] += shifts[2]  # shift z coordinates
        return points
    
    def random_flip_points(self, points, boxes):
        """Randomly flip point cloud and boxes along x-axis"""
        if random.random() > 0.5:
            # Flip points
            points[0] = -points[0]  # Flip x coordinates
            # Flip boxes
            boxes[..., 0] = -boxes[..., 0]  # Flip x coordinates of all corners
        return points, boxes
    
    def rotate_points_to_center(self, points, boxes):
        """Rotate point cloud and boxes
        Args:
            points: (3,H,W) array of point cloud
            boxes: (N,8,3) array of box corners
        """
        points = points.astype(np.float32)
        boxes = boxes.astype(np.float32)
        
        rotated_boxes = []
        # Process each box
        for box in boxes:
            # Calculate box center
            center = np.mean(box, axis=0)
            
            # Get forward direction using first four corners
            front_center = np.mean(box[:4], axis=0)
            direction = front_center - center
            
            # Calculate rotation angle
            angle = np.arctan2(direction[1], direction[0])
            
            # Create rotation matrix
            cosa = np.cos(angle)
            sina = np.sin(angle)
            rot_matrix = np.array([
                [cosa, -sina, 0.0],
                [sina, cosa, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # Rotate box
            box_centered = box - center
            box_rotated = np.dot(box_centered, rot_matrix.T)
            rotated_boxes.append(box_rotated)
            
            # Rotate points
            # Reshape points to (3,-1) for rotation
            points_flat = points.reshape(3, -1)
            points_centered = points_flat - center[:, None]  # broadcast center to all points
            points_rotated = np.dot(rot_matrix, points_centered)
            points = points_rotated.reshape(points.shape)
            
        rotated_boxes = np.stack(rotated_boxes)
        return points, rotated_boxes

    def sample_points(self, points, masks, num_points=20000):
        """Sample a fixed number of points
        Args:
            points: (H*W,3) points
            masks: (N,H*W) masks for N objects
            num_points: number of points to sample
        Returns:
            sampled_points: (num_points,3)
            sampled_masks: (N,num_points)
        """
        total_points = len(points)
        
        if total_points >= num_points:
            # Randomly sample points
            choice = np.random.choice(total_points, num_points, replace=False)
            points = points[choice]
            masks = masks[:, choice]
        else:
            # If we have fewer points, randomly duplicate some points
            choice = np.random.choice(total_points, num_points - total_points)
            extra_points = points[choice]
            extra_masks = masks[:, choice]
            points = np.concatenate([points, extra_points], axis=0)
            masks = np.concatenate([masks, extra_masks], axis=1)
            
        return points, masks

    def pad_boxes_and_masks(self, boxes, masks):
        """Pad boxes and masks to max_objects
        Args:
            boxes: (N,7) box parameters
            masks: (N,num_points) segmentation masks
        Returns:
            padded_boxes: (max_objects,7)
            padded_masks: (max_objects,num_points)
            valid_mask: (max_objects,) boolean mask indicating valid objects
        """
        num_objects = len(boxes)
        if num_objects > self.max_objects:
            # If we have more objects than max_objects, randomly select max_objects
            indices = np.random.choice(num_objects, self.max_objects, replace=False)
            boxes = boxes[indices]
            masks = masks[indices]
            valid_mask = np.ones(self.max_objects, dtype=bool)
        else:
            # Pad with zeros
            pad_boxes = np.zeros((self.max_objects - num_objects, 7), dtype=np.float32)
            pad_masks = np.zeros((self.max_objects - num_objects, masks.shape[1]), dtype=np.float32)
            boxes = np.concatenate([boxes, pad_boxes], axis=0)
            masks = np.concatenate([masks, pad_masks], axis=0)
            valid_mask = np.concatenate([np.ones(num_objects, dtype=bool),
                                       np.zeros(self.max_objects - num_objects, dtype=bool)])
        
        return boxes, masks, valid_mask

    def get_heading_angle(self, box_params):
        """Convert heading angle to class label and residual"""
        angle = box_params[6]
        angle = angle % (2 * np.pi)  # normalize to [0, 2π]
        
        # Convert to heading bin class label and residual
        bin_size = 2 * np.pi / self.num_heading_bin
        heading_class = int(angle / bin_size)
        heading_residual = angle - (heading_class * bin_size + bin_size / 2)
        
        return heading_class, heading_residual

    def get_size_class(self, box_params):
        """Convert size to residual"""
        size = box_params[3:6]  # length, width, height
        # For single class, we only compute residual from mean size
        mean_size = g_mean_size_arr[0]  # Use the single class mean size
        size_residual = size - mean_size
        return 0, size_residual  # Always return 0 for class since we only have one

    def __getitem__(self, idx):
        """Get a single data item"""
        # Load data
        points = np.load(self.pcl_files[idx])  # (3,H,W) point cloud
        boxes = np.load(self.box_files[idx])   # (N,8,3) boxes
        masks = np.load(self.seg_files[idx])   # (N,H,W) segmentation masks
        
        # Data augmentation
        if self.rotate_to_center:
            points, boxes = self.rotate_points_to_center(points, boxes)
            
        if self.random_flip:
            points, boxes = self.random_flip_points(points, boxes)
            
        if self.random_shift:
            points = self.random_shift_points(points)
        
        # Convert boxes to parameters format
        box_params = np.array([self.corners_to_params(box) for box in boxes])  # (N,7)
        
        # Reshape points to (3,num_points) format
        points = points.reshape(3, -1)  # Reshape to (3,H*W)
        
        # Reshape masks to match points
        masks = masks.reshape(len(boxes), -1)  # Reshape to (N,H*W)
        
        # Sample fixed number of points
        if points.shape[1] >= self.num_points:
            choice = np.random.choice(points.shape[1], self.num_points, replace=False)
            points = points[:, choice]  # (3,num_points)
            masks = masks[:, choice]    # (N,num_points)
        else:
            choice = np.random.choice(points.shape[1], self.num_points - points.shape[1])
            extra_points = points[:, choice]
            extra_masks = masks[:, choice]
            points = np.concatenate([points, extra_points], axis=1)  # (3,num_points)
            masks = np.concatenate([masks, extra_masks], axis=1)     # (N,num_points)
        
        # Get parameters for the first box (or use defaults if no boxes)
        if len(box_params) > 0:
            heading_class, heading_residual = self.get_heading_angle(box_params[0])
            _, size_residual = self.get_size_class(box_params[0])  # Ignore size class for single class
            center = box_params[0, :3]
            mask = masks[0]
        else:
            # If no boxes, use default values
            heading_class, heading_residual = 0, 0
            size_residual = np.zeros(3, dtype=np.float32)
            center = np.zeros(3, dtype=np.float32)
            mask = np.zeros(self.num_points, dtype=np.float32)
        
        # Create one-hot vector for single class
        # Keep it 3-dimensional to match network expectations, but only use first dimension
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[0] = 1  # Object class
        
        # Ensure all numpy arrays are float32
        points = points.astype(np.float32)
        mask = mask.astype(np.float32)
        center = center.astype(np.float32)
        size_residual = size_residual.astype(np.float32)
        heading_residual = np.float32(heading_residual)
        
        # Convert to tensors
        points = torch.from_numpy(points)
        mask = torch.from_numpy(mask)
        center = torch.from_numpy(center)
        heading_class = torch.tensor(heading_class, dtype=torch.long)
        heading_residual = torch.tensor(heading_residual)
        size_class = torch.tensor(0, dtype=torch.long)  # Always 0 for single class
        size_residual = torch.from_numpy(size_residual)
        one_hot = torch.from_numpy(one_hot)
        rot_angle = torch.zeros(1)
        
        # Print all
        print("heading class", heading_class)
        print("heading residual", heading_residual)
        print("size class", size_class)
        print("size residual", size_residual)
        print("center", center)
        print("mask", mask)
        print("points", points)

        # Create the data dictionary expected by the model
        data_dict = {
            'point_cloud': points,  # (3,num_points)
            'one_hot': one_hot,  # (3,) - maintain original dimension but only use first class
            'seg': mask,  # (num_points,)
            'box3d_center': center,  # (3,)
            'size_class': size_class,  # scalar (always 0)
            'size_residual': size_residual,  # (3,)
            'angle_class': heading_class,  # scalar
            'angle_residual': heading_residual,  # scalar
            'rot_angle': rot_angle  # (1,)
        }
        # print(data_dict)
        
        return data_dict 