import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from frustum_pointnets_pytorch.models.model_util import NUM_HEADING_BIN, g_mean_size_arr

class FrustumDataset(Dataset):
    def __init__(self, data_path, scene_list=None, num_points=1024):
        """
        Dataset for Frustum-PointNet training
        Args:
            data_path: Path to dataset directory
            scene_list: Optional list of scene directories to use. If None, uses all scenes.
            num_points: Number of points to sample (default: 1024 as in original implementation)
        """
        self.data_path = Path(data_path)
        self.num_points = num_points
        
        # Constants from the original implementation
        self.num_heading_bin = NUM_HEADING_BIN
        
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
        
        # Calculate size as max - min along each axis
        size = np.max(corners, axis=0) - np.min(corners, axis=0)
        
        # Calculate heading angle from front face
        front_center = np.mean(corners[:4], axis=0)  # assuming first 4 corners are front face
        direction = front_center - center
        heading = np.arctan2(direction[1], direction[0])  # angle from x-axis
        
        return center, size, heading

    def get_heading_angle(self, heading_angle):
        """Convert heading angle to class label and residual"""
        angle = heading_angle % (2 * np.pi)  # normalize to [0, 2π]

        # Convert to heading bin class label and residual
        bin_size = 2 * np.pi / self.num_heading_bin
        heading_class = int(angle / bin_size)
        heading_residual = angle - (heading_class * bin_size + bin_size / 2)
        
        return heading_class, heading_residual

    def get_size_class(self, size):
        """Convert size to class and residual"""
        # For single class, always return class 0 and compute residual from mean size
        mean_size = g_mean_size_arr[0]  # Use the single class mean size
        size_residual = size - mean_size
        return 0, size_residual

    def voxel_downsample(self, points, voxel_size=0.01):
        """Downsample points using voxel grid
        Args:
            points: (3, N) points
            voxel_size: size of voxel (default: 0.01m = 1cm)
        Returns:
            downsampled points: (3, M) where M <= N
        """
        # Convert to (N, 3) for easier processing
        points = points.T  # (N, 3)
        
        # Compute voxel indices for each point
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Create dictionary to store points in each voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            idx_tuple = tuple(idx)
            if idx_tuple in voxel_dict:
                voxel_dict[idx_tuple].append(points[i])
            else:
                voxel_dict[idx_tuple] = [points[i]]
        
        # Average points in each voxel
        downsampled_points = []
        for points_in_voxel in voxel_dict.values():
            centroid = np.mean(points_in_voxel, axis=0)
            downsampled_points.append(centroid)
        
        # Convert back to (3, N) format
        downsampled_points = np.array(downsampled_points).T
        return downsampled_points

    def process_point_cloud(self, points, masks):
        """Process point cloud using multiple masks and sample points
        Args:
            points: (3, H*W) point cloud
            masks: (N, H, W) N different segmentation masks
        Returns:
            sampled_points: (N, num_points, 4) stacked point clouds with intensity
            sampled_masks: (N, num_points) stacked masks
        """
        # Reshape points and masks
        points = points.reshape(3, -1)  # (3, H*W)
        num_masks = masks.shape[0]
        masks = masks.reshape(num_masks, -1)  # (N, H*W)
        
        all_masked_points = []
        all_sampled_masks = []
        
        # Process each mask separately
        for mask in masks:
            # Filter points using current mask
            masked_indices = np.where(mask > 0)[0]
            
            if len(masked_indices) > 0:
                # Get points for current mask
                masked_points = points[:, masked_indices]  # (3, M) where M is number of masked points
                
                # Downsample points to 1cm resolution
                masked_points = self.voxel_downsample(masked_points, voxel_size=0.005)
                # print("masked_points", masked_points.shape)
                # Sample points
                if masked_points.shape[1] >= self.num_points:
                    choice = np.random.choice(masked_points.shape[1], self.num_points, replace=False)
                else:
                    # If we have fewer points, randomly duplicate some points
                    choice = np.random.choice(masked_points.shape[1], self.num_points, replace=True)
                
                sampled_points = masked_points[:, choice]  # (3, num_points)
                
                # Add intensity channel and reshape to (num_points, 4)
                intensity = np.zeros((self.num_points,))
                sampled_points = np.vstack([sampled_points, intensity[None, :]])  # (4, num_points)
                sampled_points = sampled_points.T  # (num_points, 4)
            else:
                # If no points in mask, create zero points with zero intensity
                sampled_points = np.zeros((self.num_points, 4))  # (num_points, 4)
            
            all_masked_points.append(sampled_points)
            
            # Create mask indicator (all 1s since these are all masked points)
            sampled_mask = np.ones(self.num_points)
            all_sampled_masks.append(sampled_mask)
        # raise Exception("Stop here")  
        # Stack all points and masks
        stacked_points = np.stack(all_masked_points)  # (N, num_points, 4)
        stacked_masks = np.stack(all_sampled_masks)   # (N, num_points)
        
        return stacked_points, stacked_masks

    def __getitem__(self, idx):
        """Get a single data item"""
        # Load data
        points = np.load(self.pcl_files[idx])  # (3,H,W) point cloud
        boxes = np.load(self.box_files[idx])   # (N,8,3) boxes
        masks = np.load(self.seg_files[idx])   # (N,H,W) segmentation masks
        
        # Process all objects
        points_all, masks_all = self.process_point_cloud(points, masks)
        
        # Initialize lists to store processed data for each object
        centers = []
        heading_classes = []
        heading_residuals = []
        size_classes = []
        size_residuals = []
        
        # Process each box
        for i, box in enumerate(boxes):
            # Get box parameters
            center, size, heading_angle = self.corners_to_params(box)
            
            # Get heading class and residual
            heading_class, heading_residual = self.get_heading_angle(heading_angle)
            
            # Get size class and residual
            size_class, size_residual = self.get_size_class(size)
            
            centers.append(center)
            heading_classes.append(heading_class)
            heading_residuals.append(heading_residual)
            size_classes.append(size_class)
            size_residuals.append(size_residual)
        
        # Convert to numpy arrays
        centers = np.stack(centers) if centers else np.zeros((0, 3), dtype=np.float32)
        heading_classes = np.array(heading_classes) if heading_classes else np.zeros(0, dtype=np.int64)
        heading_residuals = np.array(heading_residuals) if heading_residuals else np.zeros(0, dtype=np.float32)
        size_classes = np.array(size_classes) if size_classes else np.zeros(0, dtype=np.int64)
        size_residuals = np.stack(size_residuals) if size_residuals else np.zeros((0, 3), dtype=np.float32)
        
        # Create one-hot vector (3D for network compatibility)
        one_hot = np.zeros(3, dtype=np.float32)  # Keep as 3D for network compatibility
        one_hot[0] = 1  # First class only
        
        # Convert to tensors
        points_all = torch.from_numpy(points_all.astype(np.float32))  # (N, num_points, 4)
        points_all = points_all.transpose(1, 2)  # Change to (N, 4, num_points) for network
        masks_all = torch.from_numpy(masks_all.astype(np.float32))  # (N, num_points)
        centers = torch.from_numpy(centers.astype(np.float32))  # (N, 3)
        heading_classes = torch.tensor(heading_classes, dtype=torch.long)  # (N,)
        heading_residuals = torch.tensor(heading_residuals, dtype=torch.float32)  # (N,)
        size_classes = torch.tensor(size_classes, dtype=torch.long)  # (N,)
        size_residuals = torch.from_numpy(size_residuals.astype(np.float32))  # (N, 3)
        one_hot = torch.from_numpy(one_hot)  # Keep as 3D vector
        rot_angle = torch.zeros(points_all.size(0))  # (N,) No rotation angle since we removed augmentation
        
        # print("points_all", points_all.shape)
        # print("masks_all", masks_all.shape)
        # print("centers", centers.shape)
        # print("heading_classes", heading_classes.shape)
        # print("heading_residuals", heading_residuals.shape)
        # print("size_classes", size_classes.shape)
        # print("size_residuals", size_residuals.shape)
        # Create the data dictionary matching provider_fpointnet.py format
        data_dict = {
            'point_cloud': points_all,  # (N, 4, num_points)
            'one_hot': one_hot,    # (3,) - 3D one-hot vector for network compatibility
            'seg': masks_all,      # (N, num_points)
            'box3d_center': centers,  # (N, 3)
            'size_class': size_classes,  # (N,)
            'size_residual': size_residuals,  # (N, 3)
            'angle_class': heading_classes,  # (N,)
            'angle_residual': heading_residuals,  # (N,)
            'rot_angle': rot_angle,  # (N,)
            'num_objects': torch.tensor(len(boxes), dtype=torch.long)  # Scalar indicating number of objects
        }
        
        return data_dict 