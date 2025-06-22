''' Provider class and helper functions for Frustum PointNets.
Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import _pickle as pickle
import sys
import os
import numpy as np
import torch
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from box_util import box3d_iou
from model_util_old import g_type2class, g_class2type, g_type2onehotclass
from model_util_old import g_type_mean_size
from model_util_old import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import ipdb

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

def debug_print(*args, **kwargs):
    """Helper function for debug printing"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        print(*args, **kwargs)

# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 1  # Only one class for object detection
NUM_OBJECT_POINT = 512

# Single class mappings
g_type2class = {'Car': 0}  # Using 'Car' as the default object type
g_class2type = {0: 'Car'}
g_type2onehotclass = {'Car': 0}

# Single class mean size (default car size in meters)
g_type_mean_size = {'Car': np.array([3.88, 1.63, 1.53])}  # [length, width, height]

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
g_mean_size_arr[0,:] = g_type_mean_size['Car']

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.
    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    
    # Handle both scalar and array inputs
    if isinstance(angle, np.ndarray):
        if to_label_format:
            angle[angle > np.pi] -= 2 * np.pi
    else:
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    if isinstance(pred_cls, np.ndarray):
        # Handle array input - ensure pred_cls is within valid range
        pred_cls = np.clip(pred_cls, 0, NUM_SIZE_CLUSTER-1)
        mean_size = g_type_mean_size['Car']  # Use car mean size since we have single class
        return mean_size + residual
    else:
        # Handle single value case - ensure pred_cls is within valid range
        pred_cls = min(max(pred_cls, 0), NUM_SIZE_CLUSTER-1)
        mean_size = g_type_mean_size['Car']
        return mean_size + residual


def collate_fn(batch):
    """
    Custom collate function to handle batches with different numbers of objects
    Args:
        batch: List of dictionaries from __getitem__
    Returns:
        Collated batch with padded tensors
    """
    # Initialize the collated batch
    collated_batch = {}
    
    # Handle each key in the batch
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            tensors = [b[key] for b in batch]
            
            # Handle different tensor shapes
            if len(tensors[0].shape) == 1:  # Scalar tensors
                try:
                    collated_batch[key] = torch.stack(tensors, dim=0)
                except:
                    print(f"Error stacking scalar tensors for key {key}")
                    print(f"Shapes: {[t.shape for t in tensors]}")
                    raise
                    
            elif len(tensors[0].shape) == 2:  # 2D tensors (e.g., point_cloud)
                if key == 'point_cloud':
                    # Ensure all point clouds have the same number of points
                    n_points = tensors[0].shape[1]
                    aligned_tensors = []
                    for tensor in tensors:
                        if tensor.shape[1] != n_points:
                            # Randomly sample or pad points if necessary
                            if tensor.shape[1] > n_points:
                                indices = torch.randperm(tensor.shape[1])[:n_points]
                                tensor = tensor[:, indices]
                            else:
                                padding = torch.zeros((tensor.shape[0], n_points - tensor.shape[1], tensor.shape[2]), 
                                                    dtype=tensor.dtype, device=tensor.device)
                                tensor = torch.cat([tensor, padding], dim=1)
                        aligned_tensors.append(tensor)
                    try:
                        collated_batch[key] = torch.stack(aligned_tensors, dim=0)
                    except:
                        print(f"Error stacking point clouds for key {key}")
                        print(f"Shapes after alignment: {[t.shape for t in aligned_tensors]}")
                        raise
                else:
                    # For other 2D tensors, simply stack them
                    try:
                        collated_batch[key] = torch.stack(tensors, dim=0)
                    except:
                        print(f"Error stacking 2D tensors for key {key}")
                        print(f"Shapes: {[t.shape for t in tensors]}")
                        raise
                        
            else:  # Handle higher dimensional tensors if needed
                try:
                    collated_batch[key] = torch.stack(tensors, dim=0)
                except:
                    print(f"Error stacking tensors for key {key}")
                    print(f"Shapes: {[t.shape for t in tensors]}")
                    raise
            
            # Create valid mask based on point_cloud
            if key == 'point_cloud':
                # Create mask where sum across point dimensions is non-zero
                valid_mask = (collated_batch[key].abs().sum(dim=-1) > 0).float()
                collated_batch['valid_mask'] = valid_mask
    
    return collated_batch


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False,
                 max_objects=5):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
            max_objects: int, maximum number of objects to handle per scene
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.max_objects = max_objects
        
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(ROOT_DIR,
                                                 'kitti/frustum_carpedcyc_%s.pickle' % (split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)

    def __len__(self):
        return len(self.input_list)

    def pad_or_truncate(self, data_dict, n_objects):
        """Helper function to pad or truncate data to max_objects"""
        if n_objects > self.max_objects:
            # Truncate to max_objects
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    if key == 'point_cloud':  # Special handling for point cloud
                        data_dict[key] = data_dict[key][:self.max_objects]
                    else:
                        data_dict[key] = data_dict[key][:self.max_objects]
        elif n_objects < self.max_objects:
            # Pad to max_objects
            pad_size = self.max_objects - n_objects
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    if key == 'point_cloud':  # Special handling for point cloud
                        # Create zero padding for point clouds
                        pad_shape = list(data_dict[key].shape)
                        pad_shape[0] = pad_size
                        padding = torch.zeros(pad_shape, dtype=data_dict[key].dtype, device=data_dict[key].device)
                        data_dict[key] = torch.cat([data_dict[key], padding], dim=0)
                    else:
                        # Pad with zeros or appropriate values
                        pad_shape = list(data_dict[key].shape)
                        pad_shape[0] = pad_size
                        pad_tensor = torch.zeros(pad_shape, dtype=data_dict[key].dtype, device=data_dict[key].device)
                        data_dict[key] = torch.cat([data_dict[key], pad_tensor], dim=0)
        return data_dict

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.from_rgb_detection:
            data_dict = {
                'point_cloud': torch.from_numpy(point_set).float(),  # (n_points,4)
                'rot_angle': torch.FloatTensor([rot_angle]),  # (1,)
                'prob': torch.FloatTensor([self.prob_list[index]])  # (1,)
            }
            if self.one_hot:
                data_dict['one_hot'] = torch.from_numpy(one_hot_vec).float()  # (3,)
            return data_dict

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            if np.random.random() > 0.5:
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)

        # Convert to tensors with proper shapes
        data_dict = {
            'point_cloud': torch.from_numpy(point_set).float(),  # (n_points,4)
            'seg': torch.from_numpy(seg).float(),  # (n_points,)
            'box3d_center': torch.from_numpy(box3d_center).float(),  # (3,)
            'angle_class': torch.tensor(angle_class, dtype=torch.long),  # scalar
            'angle_residual': torch.tensor(angle_residual, dtype=torch.float),  # scalar
            'size_class': torch.tensor(size_class, dtype=torch.long),  # scalar
            'size_residual': torch.from_numpy(size_residual).float(),  # (3,)
            'rot_angle': torch.FloatTensor([rot_angle]),  # (1,)
            'heading_class': torch.tensor(angle_class, dtype=torch.long),  # scalar
            'heading_residual': torch.tensor(angle_residual, dtype=torch.float),  # scalar
            'size_class_label': torch.tensor(size_class, dtype=torch.long),  # scalar
            'size_residual_label': torch.from_numpy(size_residual).float()  # (3,)
        }
        
        if self.one_hot:
            data_dict['one_hot'] = torch.from_numpy(one_hot_vec).float()  # (3,)
            
        return data_dict

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_box3d_corners_helper(centers, headings, sizes):
    """ 
    Input: 
        centers: (N,3) or (3,) numpy array or torch tensor - box centers
        headings: (N,) or scalar numpy array or torch tensor - heading angles
        sizes: (N,3) or (3,) numpy array or torch tensor - box l,w,h
    Output: 
        corners: (N,8,3) or (8,3) torch tensor for box corners
    Note: We assume the box center is at the origin before rotation and translation.
          The order of corners is compatible with box3d_iou.py
    """
    device = centers.device if isinstance(centers, torch.Tensor) else torch.device('cuda')
    dtype = torch.float32
    
    # Convert inputs to tensors if needed
    if isinstance(centers, np.ndarray):
        centers = torch.from_numpy(centers).to(dtype).to(device)
    if isinstance(headings, (np.ndarray, float, int)):
        headings = torch.tensor(headings, dtype=dtype, device=device)
    if isinstance(sizes, np.ndarray):
        sizes = torch.from_numpy(sizes).to(dtype).to(device)
    
    # Ensure all inputs are at least 2D
    if centers.dim() == 1:
        centers = centers.unsqueeze(0)  # (1,3)
    if headings.dim() == 0:
        headings = headings.unsqueeze(0)  # (1,)
    if sizes.dim() == 1:
        sizes = sizes.unsqueeze(0)  # (1,3)
        
    # Ensure consistent batch dimension
    N = centers.shape[0]
    if sizes.shape[0] != N:
        sizes = sizes.expand(N, -1)  # Expand to match batch size
    if headings.shape[0] != N:
        headings = headings.expand(N)  # Expand to match batch size
        
    # Get box dimensions
    l = sizes[:, 0].view(N, 1)  # (N,1) - length
    w = sizes[:, 1].view(N, 1)  # (N,1) - width
    h = sizes[:, 2].view(N, 1)  # (N,1) - height
    
    # Create template box corners (before rotation)
    x_corners = l/2 * torch.tensor([ 1,  1, -1, -1,  1,  1, -1, -1], device=device, dtype=dtype)  # (N,8)
    y_corners = h/2 * torch.tensor([ 1,  1,  1,  1, -1, -1, -1, -1], device=device, dtype=dtype)  # (N,8)
    z_corners = w/2 * torch.tensor([ 1, -1, -1,  1,  1, -1, -1,  1], device=device, dtype=dtype)  # (N,8)
    
    # Stack corners into (N,3,8)
    corners_template = torch.stack([x_corners, y_corners, z_corners], dim=1)  # (N,3,8)
    
    # Create rotation matrices
    cos_t = torch.cos(headings)
    sin_t = torch.sin(headings)
    zeros = torch.zeros_like(cos_t)
    ones = torch.ones_like(cos_t)
    
    # (N,3,3) rotation matrices
    R = torch.stack([
        torch.stack([cos_t, zeros, sin_t], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin_t, zeros, cos_t], dim=1)
    ], dim=1)
    
    # Rotate corners
    corners_rotated = torch.bmm(R, corners_template)  # (N,3,8)
    
    # Translate corners
    corners = corners_rotated + centers.unsqueeze(-1)  # (N,3,8)
    
    # Transpose to get (N,8,3)
    corners = corners.transpose(1, 2)  # (N,8,3)
    
    # Remove batch dimension if input was single box
    if N == 1 and centers.shape[0] == 1:
        corners = corners.squeeze(0)  # (8,3)
        
    return corners


def compute_box3d_iou(center_pred,
                     heading_logits, heading_residuals,
                     size_logits, size_residuals,
                     center_label,
                     heading_class_label, heading_residual_label,
                     size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.
    
    Input:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    batch_size = heading_logits.shape[0]
    
    # Get predicted heading angle and size
    heading_class = np.clip(np.argmax(heading_logits, 1), 0, NUM_HEADING_BIN-1)
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(batch_size)])
    size_class = np.clip(np.argmax(size_logits, 1), 0, NUM_SIZE_CLUSTER-1)
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    
    for i in range(batch_size):
        # Convert to tensors
        center_pred_i = torch.tensor(center_pred[i], dtype=dtype, device=device)
        center_label_i = torch.tensor(center_label[i], dtype=dtype, device=device)
        
        # Get heading angle
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
        heading_angle = torch.tensor(heading_angle, dtype=dtype, device=device)
        
        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        heading_angle_label = torch.tensor(heading_angle_label, dtype=dtype, device=device)
        
        # Get box size
        box_size = class2size(size_class[i], size_residual[i])
        box_size = torch.tensor(box_size, dtype=dtype, device=device)
        
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        box_size_label = torch.tensor(box_size_label, dtype=dtype, device=device)
        
        try:
            # Compute corners
            corners_3d = get_box3d_corners_helper(center_pred_i, heading_angle, box_size)
            corners_3d_label = get_box3d_corners_helper(center_label_i, heading_angle_label, box_size_label)
            
            # Convert to numpy for IoU computation
            corners_3d = corners_3d.detach().cpu().numpy()
            corners_3d_label = corners_3d_label.detach().cpu().numpy()
            
            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
            
            debug_print(f"Batch {i} - IoU3D: {iou_3d:.4f}, IoU2D: {iou_2d:.4f}")
            debug_print(f"Shapes: center_pred={center_pred_i.shape}, heading={heading_angle.shape}, size={box_size.shape}")
            
        except Exception as e:
            debug_print(f"Warning: Error computing IoU for batch item {i}. Using 0. Error: {str(e)}")
            debug_print(f"Shapes: center_pred={center_pred_i.shape}, heading={heading_angle.shape}, size={box_size.shape}")
            iou_3d, iou_2d = 0, 0
            
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)

    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset(1024, split='val',
                             rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data['box3d_center'], \
               'angle_class: ', data['angle_class'], 'angle_res:', data['angle_residual'], \
               'size_class: ', data['size_class'], 'size_residual:', data['size_residual'], \
               'real_size:', g_type_mean_size[g_class2type[data['size_class']]] + data['size_residual']))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data['point_cloud'][:, 0]))
        print((data['box3d_center'], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_box3d_corners_helper(torch.tensor(data['box3d_center']), torch.tensor(data['angle_class']), torch.tensor(data['size_class'] + data['size_residual']))
        ps = data['point_cloud']
        seg = data['seg']
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))