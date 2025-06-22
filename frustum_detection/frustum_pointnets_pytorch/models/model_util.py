import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Global Constants
# -----------------
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 1  # Single class for object detection
NUM_OBJECT_POINT = 512

# Class mappings
g_type2class = {'Object': 0}
g_class2type = {0: 'Object'}
g_type2onehotclass = {'Object': 0}

# Mean size for single class (example values - adjust based on your data)
g_type_mean_size = {'Object': np.array([0.1, 0.1, 0.1])}  # [length, width, height]
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
g_mean_size_arr[0,:] = g_type_mean_size['Object']

def ensure_tensor(x, dtype=torch.float32, device=None):
    """Convert input to tensor with specified dtype and device."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x.to(dtype)

def parse_output_to_tensors(box_pred, logits, mask, stage1_center, bs=None, n_obj=None):
    """
    Parse network outputs to separate tensors.
    Args:
        box_pred: (bs*N,59) or (bs,N,59) - box prediction
        logits: (bs*N,n_points,2) or (bs,N,n_points,2) - segmentation logits
        mask: (bs*N,n_points) or (bs,N,n_points) - segmentation mask
        stage1_center: (bs*N,3) or (bs,N,3) - initial center prediction
        bs: Optional batch size for reshaping
        n_obj: Optional number of objects per batch for reshaping
    Returns:
        Dictionary containing parsed outputs with shape (bs,N,...)
    """
    device = box_pred.device
    dtype = box_pred.dtype

    # Reshape if needed
    if bs is not None and n_obj is not None:
        box_pred = box_pred.view(bs, n_obj, -1)
        logits = logits.view(bs, n_obj, *logits.shape[1:])
        mask = mask.view(bs, n_obj, -1)
        stage1_center = stage1_center.view(bs, n_obj, 3)
    else:
        if len(box_pred.shape) == 2:
            box_pred = box_pred.unsqueeze(0)
            logits = logits.unsqueeze(0)
            mask = mask.unsqueeze(0)
            stage1_center = stage1_center.unsqueeze(0)
    
    bs, n_obj = box_pred.shape[:2]
    c = 0

    # Parse box prediction
    center_boxnet = box_pred[:, :, c:c+3]  # (bs,N,3)
    c += 3

    heading_scores = box_pred[:, :, c:c+NUM_HEADING_BIN]  # (bs,N,NH)
    c += NUM_HEADING_BIN
    heading_residual_normalized = box_pred[:, :, c:c+NUM_HEADING_BIN]  # (bs,N,NH)
    heading_residual = heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    size_scores = box_pred[:, :, c:c+NUM_SIZE_CLUSTER]  # (bs,N,NS)
    c += NUM_SIZE_CLUSTER
    size_residual_normalized = box_pred[:, :, c:c+3*NUM_SIZE_CLUSTER]  # (bs,N,NS*3)
    size_residual_normalized = size_residual_normalized.view(bs, n_obj, NUM_SIZE_CLUSTER, 3)
    
    # Convert mean size array to tensor
    mean_size_arr = ensure_tensor(g_mean_size_arr, dtype=dtype, device=device)
    size_residual = size_residual_normalized * mean_size_arr.view(1, 1, NUM_SIZE_CLUSTER, 3)

    return {
        'center_boxnet': center_boxnet,
        'heading_scores': heading_scores,
        'heading_residual_normalized': heading_residual_normalized,
        'heading_residual': heading_residual,
        'size_scores': size_scores,
        'size_residual_normalized': size_residual_normalized,
        'size_residual': size_residual,
        'logits': logits,
        'mask': mask,
        'stage1_center': stage1_center
    }

def point_cloud_masking(pts, mask):
    '''
    Input:
        pts: (batch_size, 3, npoints)
        mask: (batch_size, npoints)
    Output:
        pts_masked: (batch_size, 3, NUM_OBJECT_POINT)
        mask_xyz_mean: (batch_size, 3)
        mask: (batch_size, npoints) - updated mask if no points were masked
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]

    pts_masked = []
    mask_xyz_mean = []

    for i in range(bs):
        # Get masked points
        pts_masked_i = pts[i, :, mask[i] > 0.5]  # (3, num_masked_points)
        
        # If no points are masked, use all points
        if pts_masked_i.shape[1] == 0:
            pts_masked_i = pts[i]
            mask[i] = torch.ones_like(mask[i])
        
        # Sample or pad to NUM_OBJECT_POINT
        num_masked = pts_masked_i.shape[1]
        if num_masked >= NUM_OBJECT_POINT:
            # Randomly sample NUM_OBJECT_POINT points
            choice = torch.randperm(num_masked)[:NUM_OBJECT_POINT]
            pts_masked_i = pts_masked_i[:, choice]
        else:
            # Pad by repeating points
            padding_choice = torch.randint(0, num_masked, (NUM_OBJECT_POINT - num_masked,))
            pts_masked_i = torch.cat([pts_masked_i, pts_masked_i[:, padding_choice]], dim=1)
        
        # Calculate mean of masked points (using original points before padding)
        mask_xyz_mean_i = torch.mean(pts_masked_i[:, :min(num_masked, NUM_OBJECT_POINT)], dim=1)  # (3,)
        
        pts_masked.append(pts_masked_i)
        mask_xyz_mean.append(mask_xyz_mean_i)

    # Stack results - now all tensors should be same size (3, NUM_OBJECT_POINT)
    pts_masked = torch.stack(pts_masked, dim=0)  # (bs, 3, NUM_OBJECT_POINT)
    mask_xyz_mean = torch.stack(mask_xyz_mean, dim=0)  # (bs, 3)

    return pts_masked, mask_xyz_mean, mask

def gather_object_pts(pts, mask, n_pts=NUM_OBJECT_POINT):
    '''
    :param pts: (bs,c,1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    '''
    bs = pts.shape[0]
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)  # (bs, 512)
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))

    for i in range(bs):
        pos_indices = torch.where(mask[i, :] > 0.5)[0]  # (653,)
        if len(pos_indices) > 0:
            if len(pos_indices) > n_pts:
                choice = np.random.choice(len(pos_indices),
                                          n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices),
                                          n_pts - len(pos_indices), replace=True)
                choice = np.concatenate(
                    (np.arange(len(pos_indices)), choice))
            np.random.shuffle(choice)  # (512,)
            indices[i, :] = pos_indices[choice]
            object_pts[i,:,:] = pts[i,:,indices[i,:]]
        ###else?
    return object_pts, indices

def get_box3d_corners_helper(centers, headings, sizes):
    """
    Convert box parameters to corner coordinates.
    Args:
        centers: (bs,N,3) or (N,3) - box centers
        headings: (bs,N) or (N,) - heading angles
        sizes: (bs,N,3) or (N,3) - box sizes (l,w,h)
    Returns:
        corners: (bs,N,8,3) or (N,8,3) - 3D box corners
    """
    device = centers.device
    dtype = centers.dtype
    
    # Ensure inputs are proper tensors
    centers = ensure_tensor(centers, dtype=dtype, device=device)
    headings = ensure_tensor(headings, dtype=dtype, device=device)
    sizes = ensure_tensor(sizes, dtype=dtype, device=device)
    
    # Add batch dimension if needed
    if len(centers.shape) == 2:
        centers = centers.unsqueeze(0)
        headings = headings.unsqueeze(0)
        sizes = sizes.unsqueeze(0)
    
    bs, n_obj = centers.shape[:2]
    
    # Get box dimensions
    l = sizes[..., 0].view(bs, n_obj, 1)  # (bs,N,1)
    w = sizes[..., 1].view(bs, n_obj, 1)  # (bs,N,1)
    h = sizes[..., 2].view(bs, n_obj, 1)  # (bs,N,1)

    # Create corner coordinates relative to center
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=2)  # (bs,N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=2)  # (bs,N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=2)  # (bs,N,8)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=3)  # (bs,N,8,3)

    # Create rotation matrices
    cos_angle = torch.cos(headings).view(bs, n_obj, 1, 1)  # (bs,N,1,1)
    sin_angle = torch.sin(headings).view(bs, n_obj, 1, 1)  # (bs,N,1,1)
    zeros = torch.zeros_like(cos_angle, device=device, dtype=dtype)  # (bs,N,1,1)
    ones = torch.ones_like(cos_angle, device=device, dtype=dtype)   # (bs,N,1,1)

    R = torch.cat([
        torch.cat([cos_angle, zeros, sin_angle], dim=3),   # (bs,N,1,3)
        torch.cat([zeros, ones, zeros], dim=3),            # (bs,N,1,3)
        torch.cat([-sin_angle, zeros, cos_angle], dim=3)   # (bs,N,1,3)
    ], dim=2)  # (bs,N,3,3)

    # Rotate and translate corners
    corners = torch.matmul(corners, R.transpose(2, 3))  # (bs,N,8,3)
    corners = corners + centers.unsqueeze(2)  # (bs,N,8,3)

    # Remove batch dimension if input was unbatched
    if len(centers.shape) == 2:
        corners = corners.squeeze(0)

    return corners

def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    device = center.device
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(\
            np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)).float().to(device) # (NH,)
    headings = heading_residual + heading_bin_centers.view(1,-1) # (bs,NH)

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().to(device) # (1,3)
    sizes = mean_sizes.view(1,1,3) + size_residual # (bs,1,3)
    sizes = sizes.view(bs,1,NUM_SIZE_CLUSTER,3)\
                .repeat(1,NUM_HEADING_BIN,1,1).float() # (bs,NH,1,3)
    headings = headings.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER) # (bs,NH,1)
    centers = center.view(bs,1,1,3).repeat(1,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,1) # (bs,NH,1,3)
    
    N = bs*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N,3),
                                        headings.view(N),
                                        sizes.view(N,3))
    
    return corners_3d.view((bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3))

def huber_loss(error, delta=1.0):
    """
    Compute the Huber loss.
    Args:
        error: Tensor of any shape - error between predictions and targets
        delta: Float scalar - Huber loss parameter (default: 1.0)
    Returns:
        loss: Reduced Huber loss (always non-negative)
    """
    device = error.device
    dtype = error.dtype
    
    # Ensure error is properly handled
    abs_error = torch.abs(error)
    delta = ensure_tensor(delta, dtype=dtype, device=device)
    
    # Compute quadratic and linear terms
    quadratic = torch.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    # Compute final loss (always non-negative since we use abs_error)
    loss = 0.5 * quadratic.pow(2) + delta * linear
    
    return torch.mean(loss)

class FrustumPointNetLoss(nn.Module):
    def __init__(self):
        super(FrustumPointNetLoss, self).__init__()

    def forward(self, predictions, targets, valid_mask=None):
        """
        Compute all losses for Frustum PointNet.
        Args:
            predictions: Dictionary containing network predictions
            targets: Dictionary containing ground truth
            valid_mask: (bs,N) boolean mask for valid objects
        Returns:
            total_loss: Scalar total loss (guaranteed non-negative)
            loss_dict: Dictionary of individual losses (all non-negative)
        """
        device = predictions['logits'].device
        dtype = predictions['logits'].dtype
        bs, n_obj = predictions['logits'].shape[:2]

        print("\n=== FrustumPointNetLoss Debug Information ===")
        print(f"Batch size: {bs}, Objects per batch: {n_obj}")

        # If no valid_mask provided, assume all objects are valid
        if valid_mask is None:
            valid_mask = torch.ones((bs, n_obj), dtype=torch.bool, device=device)
        print(f"Number of valid objects: {valid_mask.sum().item()}")

        # Compute segmentation loss (non-negative)
        seg_loss = F.cross_entropy(
            predictions['logits'][valid_mask].permute(0,2,1),  # (N,2,n_points)
            targets['seg'][valid_mask].long(),  # (N,n_points)
            reduction='mean'
        )
        print(f"\nSegmentation Loss: {seg_loss.item():.6f}")

        # Compute center loss (non-negative due to norm)
        center_loss = torch.mean(
            torch.norm(predictions['box3d_center'] - targets['box3d_center'], dim=2)[valid_mask]
        )
        print(f"Center Loss: {center_loss.item():.6f}")
        
        stage1_center_loss = torch.mean(
            torch.norm(predictions['stage1_center'] - targets['box3d_center'], dim=2)[valid_mask]
        )
        print(f"Stage1 Center Loss: {stage1_center_loss.item():.6f}")

        # Compute heading loss (non-negative)
        heading_class_loss = F.cross_entropy(
            predictions['heading_scores'][valid_mask],  # (N,NH)
            targets['angle_class'][valid_mask].long(),  # (N,)
            reduction='mean',
            label_smoothing=0.1  # Add label smoothing to improve generalization
        )
        
        # Compute heading residual loss with periodic consideration
        heading_residual_normalized_label = targets['angle_residual'] / (np.pi/NUM_HEADING_BIN)
        heading_residual_pred = torch.gather(
            predictions['heading_residual_normalized'][valid_mask],  # (N,NH)
            1,
            targets['angle_class'][valid_mask].long().unsqueeze(1)  # (N,1)
        ).squeeze(1)  # (N,)
        
        # Use periodic loss for heading residual
        heading_diff = heading_residual_pred - heading_residual_normalized_label[valid_mask]
        # Normalize to [-0.5, 0.5] range within bin
        heading_diff = torch.where(heading_diff > 0.5, heading_diff - 1.0, heading_diff)
        heading_diff = torch.where(heading_diff < -0.5, heading_diff + 1.0, heading_diff)
        heading_residual_normalized_loss = huber_loss(heading_diff, delta=1.0)

        # Reduce multiplier from 20x to 10x
        heading_residual_weight = 10.0  # Changed from 20.0

        # Compute size loss (non-negative)
        size_class_loss = F.cross_entropy(
            predictions['size_scores'][valid_mask],  # (N,NS)
            targets['size_class'][valid_mask].long(),  # (N,)
            reduction='mean'
        )
        print(f"Size Class Loss: {size_class_loss.item():.6f}")
        
        # Compute size residual loss
        size_residual_pred = torch.gather(
            predictions['size_residual_normalized'][valid_mask],  # (N,NS,3)
            1,
            targets['size_class'][valid_mask].long().unsqueeze(1).unsqueeze(2).expand(-1,-1,3)  # (N,1,3)
        ).squeeze(1)  # (N,3)
        
        # Print size residual debug info
        print("\nSize Residual Debug:")
        print(f"Pred range: [{size_residual_pred.min().item():.6f}, {size_residual_pred.max().item():.6f}]")
        print(f"Label range: [{targets['size_residual'][valid_mask].min().item():.6f}, {targets['size_residual'][valid_mask].max().item():.6f}]")
        
        # Compute size residual loss using absolute difference
        size_diff = size_residual_pred - targets['size_residual'][valid_mask]
        size_residual_normalized_loss = huber_loss(size_diff, delta=1.0)
        print(f"Size Residual Loss: {size_residual_normalized_loss.item():.6f}")

        # Compute corner loss
        corners_3d = get_box3d_corners_helper(
            predictions['box3d_center'][valid_mask],
            predictions['heading_residual'][valid_mask].gather(
                1,
                targets['angle_class'][valid_mask].long().unsqueeze(1)
            ).squeeze(1),
            predictions['size_residual'][valid_mask].gather(
                1,
                targets['size_class'][valid_mask].long().unsqueeze(1).unsqueeze(2).expand(-1,-1,3)
            ).squeeze(1)
        )
        
        gt_corners_3d = get_box3d_corners_helper(
            targets['box3d_center'][valid_mask],
            targets['angle_residual'][valid_mask],
            targets['size_residual'][valid_mask]
        )
        
        # Print corners debug info
        print("\nCorners Debug:")
        print(f"Pred corners range: [{corners_3d.min().item():.6f}, {corners_3d.max().item():.6f}]")
        print(f"GT corners range: [{gt_corners_3d.min().item():.6f}, {gt_corners_3d.max().item():.6f}]")
        
        # Compute corner loss using absolute difference
        corners_diff = corners_3d - gt_corners_3d
        corners_loss = huber_loss(corners_diff, delta=1.0)
        print(f"Corners Loss: {corners_loss.item():.6f}")

        # Verify all losses are non-negative
        assert torch.all(seg_loss >= 0), f"Segmentation loss should be non-negative, got {seg_loss}"
        assert torch.all(center_loss >= 0), f"Center loss should be non-negative, got {center_loss}"
        assert torch.all(stage1_center_loss >= 0), f"Stage1 center loss should be non-negative, got {stage1_center_loss}"
        assert torch.all(heading_class_loss >= 0), f"Heading class loss should be non-negative, got {heading_class_loss}"
        assert torch.all(size_class_loss >= 0), f"Size class loss should be non-negative, got {size_class_loss}"
        assert torch.all(heading_residual_normalized_loss >= 0), f"Heading residual loss should be non-negative, got {heading_residual_normalized_loss}"
        assert torch.all(size_residual_normalized_loss >= 0), f"Size residual loss should be non-negative, got {size_residual_normalized_loss}"
        assert torch.all(corners_loss >= 0), f"Corners loss should be non-negative, got {corners_loss}"

        # Total loss (weighted sum of non-negative terms)
        total_loss = seg_loss + \
                    0.5 * center_loss + \
                    stage1_center_loss + \
                    heading_class_loss + \
                    size_class_loss + \
                    heading_residual_normalized_loss * heading_residual_weight + \
                    size_residual_normalized_loss * 20 + \
                    corners_loss

        print("\nWeighted Losses:")
        print(f"Segmentation Loss: {seg_loss.item():.6f}")
        print(f"Center Loss (0.5x): {(0.5 * center_loss).item():.6f}")
        print(f"Stage1 Center Loss: {stage1_center_loss.item():.6f}")
        print(f"Heading Class Loss: {heading_class_loss.item():.6f}")
        print(f"Size Class Loss: {size_class_loss.item():.6f}")
        print(f"Heading Residual Loss (10x): {(heading_residual_normalized_loss * heading_residual_weight).item():.6f}")
        print(f"Size Residual Loss (20x): {(size_residual_normalized_loss * 20).item():.6f}")
        print(f"Corners Loss: {corners_loss.item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")
        print("="*40)

        # Final sanity check
        assert torch.all(total_loss >= 0), f"Total loss should be non-negative, got {total_loss}"

        loss_dict = {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'center_loss': center_loss,
            'heading_class_loss': heading_class_loss,
            'size_class_loss': size_class_loss,
            'heading_residual_normalized_loss': heading_residual_normalized_loss,
            'size_residual_normalized_loss': size_residual_normalized_loss,
            'stage1_center_loss': stage1_center_loss,
            'corners_loss': corners_loss
        }

        return total_loss, loss_dict
