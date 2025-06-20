import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 1  # Only one class for object detection
NUM_OBJECT_POINT = 512

# Single class mappings
g_type2class = {'Object': 0}
g_class2type = {0: 'Object'}
g_type2onehotclass = {'Object': 0}

# Single class mean size (example values - adjust these based on your data)
g_type_mean_size = {'Object': np.array([0.2, 0.2, 0.2])}  # [length, width, height]

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
g_mean_size_arr[0,:] = g_type_mean_size['Object']

def parse_output_to_tensors(box_pred, logits, mask, stage1_center):
    '''
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residual_normalized:(bs,12),-1 to 1
        heading_residual:(bs,12)
        size_scores:(bs,1)
        size_residual_normalized:(bs,1,3)
        size_residual:(bs,1,3)
    '''
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]
    c = 3

    # heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residual_normalized = box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residual = heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residual_normalized = box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residual_normalized = size_residual_normalized.view(bs, NUM_SIZE_CLUSTER, 3)
    size_residual = size_residual_normalized * \
                     torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs,1,1).cuda()
    
    return center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual

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
    Input: 
        centers: (N,3) or (3,) numpy array or torch tensor
        headings: (N,) or scalar numpy array or torch tensor
        sizes: (N,3) or (3,) numpy array or torch tensor
    Output: 
        corners: (N,8,3) or (8,3) numpy array
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(centers, np.ndarray):
        centers = torch.from_numpy(centers).float().cuda()
    if isinstance(headings, (np.ndarray, float, int)):
        headings = torch.tensor(headings).float().cuda()
    if isinstance(sizes, np.ndarray):
        sizes = torch.from_numpy(sizes).float().cuda()
    
    # Handle single box case
    if len(centers.shape) == 1:
        centers = centers.unsqueeze(0)
    if isinstance(headings, (float, int)) or (isinstance(headings, torch.Tensor) and len(headings.shape) == 0):
        headings = headings.unsqueeze(0)
    if len(sizes.shape) == 1:
        sizes = sizes.unsqueeze(0)

    N = centers.shape[0]
    l = sizes[:,0].view(N,1)
    w = sizes[:,1].view(N,1)
    h = sizes[:,2].view(N,1)

    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
    corners = torch.cat([x_corners.view(N,1,8), y_corners.view(N,1,8),\
                            z_corners.view(N,1,8)], dim=1) # (N,3,8)

    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()
    row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.view(N,1,3), row2.view(N,1,3), \
                      row3.view(N,1,3)], axis=1) # (N,3,3)

    corners_3d = torch.bmm(R, corners) # (N,3,8)
    corners_3d +=centers.view(N,3,1).repeat(1,1,8) # (N,3,8)
    corners_3d = torch.transpose(corners_3d,1,2) # (N,8,3)
    
    # Convert back to numpy - detach first if requires grad
    corners_3d = corners_3d.detach().cpu().numpy()
    
    # If input was single box, squeeze the first dimension
    if N == 1:
        corners_3d = corners_3d.squeeze(0)
    
    return corners_3d

def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(\
            np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)).float().cuda() # (NH,)
    headings = heading_residual + heading_bin_centers.view(1,-1) # (bs,NH)

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().cuda() # (1,3)
    sizes = mean_sizes.view(1,1,3) + size_residual # (bs,1,3)
    sizes = sizes.view(bs,1,NUM_SIZE_CLUSTER,3)\
                .repeat(1,NUM_HEADING_BIN,1,1).float() # (bs,NH,1,3)
    headings = headings.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER) # (bs,NH,1)
    centers = center.view(bs,1,1,3).repeat(1,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,1) # (bs,NH,1,3)
    
    N = bs*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N,3),
                                        headings.view(N),
                                        sizes.view(N,3))
    
    # Convert corners_3d back to torch tensor since helper returns numpy array
    corners_3d = torch.from_numpy(corners_3d).float().cuda()
    
    # Reshape using view with a tuple for the target shape
    return corners_3d.view((bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3))

def huber_loss(error, delta=1.0):#(32,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

class FrustumPointNetLoss(nn.Module):
    def __init__(self):
        super(FrustumPointNetLoss, self).__init__()

    def forward(self, logits, mask_label, \
                 center, center_label, stage1_center, \
                 heading_scores, heading_residual_normalized, heading_residual, \
                 heading_class_label, heading_residual_label, \
                 size_scores, size_residual_normalized, size_residual,
                 size_class_label, size_residual_label,
                 corner_loss_weight=10.0, box_loss_weight=1.0):
        '''
        Binary classification for object detection and 3D box estimation
        '''
        device = logits.device
        
        print("\n=== Model Outputs ===")
        print(f"Logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Center shape: {center.shape}, values: {center[0]}")
        print(f"Stage1 Center shape: {stage1_center.shape}, values: {stage1_center[0]}")
        print(f"Heading scores shape: {heading_scores.shape}, values: {heading_scores[0]}")
        print(f"Heading residual shape: {heading_residual.shape}, values: {heading_residual[0]}")
        print(f"Size scores shape: {size_scores.shape}, values: {size_scores[0]}")
        print(f"Size residual shape: {size_residual.shape}, values: {size_residual[0]}")
        
        print("\n=== Ground Truth ===")
        print(f"Mask label shape: {mask_label.shape}, unique values: {torch.unique(mask_label)}")
        print(f"Center label shape: {center_label.shape}, values: {center_label[0]}")
        print(f"Heading class label shape: {heading_class_label.shape}, values: {heading_class_label[0]}")
        print(f"Heading residual label shape: {heading_residual_label.shape}, values: {heading_residual_label[0]}")
        print(f"Size class label shape: {size_class_label.shape}, values: {size_class_label[0]}")
        print(f"Size residual label shape: {size_residual_label.shape}, values: {size_residual_label[0]}")
        
        # Binary classification loss (object vs. not object)
        logits = F.log_softmax(logits.view(-1,2), dim=1)  # (bs*n_points, 2)
        mask_label = mask_label.view(-1).long()  # (bs*n_points,)
        mask_loss = F.nll_loss(logits, mask_label)
        print(f"\nMask Loss: {mask_loss.item():.4f}")
        
        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)
        print(f"Center Loss: {center_loss.item():.4f}")
        print(f"Center distances: min={center_dist.min().item():.4f}, max={center_dist.max().item():.4f}, mean={center_dist.mean().item():.4f}")
        
        stage1_center_dist = torch.norm(center - stage1_center, dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
        print(f"Stage1 Center Loss: {stage1_center_loss.item():.4f}")
        
        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), heading_class_label.long())
        print(f"Heading Class Loss: {heading_class_loss.item():.4f}")
        print(f"Predicted heading class: {torch.argmax(heading_scores, dim=1)[0]}, GT: {heading_class_label[0]}")
        
        hcls_onehot = torch.eye(NUM_HEADING_BIN, device=device)[heading_class_label.long()]
        heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
        heading_residual_normalized_dist = torch.sum(heading_residual_normalized * hcls_onehot, dim=1)
        heading_residual_normalized_loss = huber_loss(heading_residual_normalized_dist - heading_residual_normalized_label, delta=1.0)
        print(f"Heading Residual Loss: {heading_residual_normalized_loss.item():.4f}")
        
        # Size loss (simplified for single class)
        size_residual_normalized_dist = size_residual_normalized.squeeze(1)
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().to(device)
        size_residual_label_normalized = size_residual_label / mean_size_arr_expand[0]
        size_normalized_dist = torch.norm(size_residual_label_normalized - size_residual_normalized_dist, dim=1)
        size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
        print(f"Size Residual Loss: {size_residual_normalized_loss.item():.4f}")

        # Corner Loss
        corners_3d = get_box3d_corners(center, heading_residual, size_residual)  # This now returns a tensor
        pred_heading_class = torch.argmax(heading_scores, dim=1)
        pred_corners = corners_3d[torch.arange(corners_3d.shape[0]), pred_heading_class, 0]
        
        # Convert heading_residual_label to tensor if it's not already
        gt_heading = heading_residual_label.view(-1,1)
        gt_size = size_residual_label
        
        # Create tensors for both normal and flipped corners
        gt_corners = torch.from_numpy(get_box3d_corners_helper(center_label, gt_heading.squeeze(1), gt_size)).float().to(device)
        gt_corners_flipped = torch.from_numpy(get_box3d_corners_helper(center_label, gt_heading.squeeze(1) + np.pi, gt_size)).float().to(device)
        
        # Calculate corner loss using torch operations
        corners_dist = torch.min(
            torch.mean(torch.norm(pred_corners - gt_corners, dim=2), dim=1),
            torch.mean(torch.norm(pred_corners - gt_corners_flipped, dim=2), dim=1)
        )
        corner_loss = huber_loss(corners_dist, delta=1.0)
        print(f"Corner Loss: {corner_loss.item():.4f}")
        print(f"Corner distances: min={corners_dist.min().item():.4f}, max={corners_dist.max().item():.4f}, mean={corners_dist.mean().item():.4f}")
        
        # Weighted sum of all losses
        total_loss = mask_loss + box_loss_weight * (center_loss + \
                    heading_class_loss + \
                    heading_residual_normalized_loss * 20 + \
                    size_residual_normalized_loss * 20 + \
                    stage1_center_loss + \
                    corner_loss_weight * corner_loss)
        
        print(f"\n=== Final Weighted Losses ===")
        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"Mask Loss (w=1.0): {mask_loss.item():.4f}")
        print(f"Center Loss (w={box_loss_weight}): {(box_loss_weight * center_loss).item():.4f}")
        print(f"Heading Class Loss (w={box_loss_weight}): {(box_loss_weight * heading_class_loss).item():.4f}")
        print(f"Heading Residual Loss (w={box_loss_weight * 20}): {(box_loss_weight * heading_residual_normalized_loss * 20).item():.4f}")
        print(f"Size Residual Loss (w={box_loss_weight * 20}): {(box_loss_weight * size_residual_normalized_loss * 20).item():.4f}")
        print(f"Stage1 Center Loss (w={box_loss_weight}): {(box_loss_weight * stage1_center_loss).item():.4f}")
        print(f"Corner Loss (w={box_loss_weight * corner_loss_weight}): {(box_loss_weight * corner_loss * corner_loss_weight).item():.4f}")
        print("="*50)
        
        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'center_loss': box_loss_weight * center_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
            'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'stage1_center_loss': box_loss_weight * stage1_center_loss,
            'corners_loss': box_loss_weight * corner_loss * corner_loss_weight,
        }
        return losses
