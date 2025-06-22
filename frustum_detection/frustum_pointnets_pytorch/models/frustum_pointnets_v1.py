import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from torch.nn import init
from .model_util import (NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT,
                      point_cloud_masking, parse_output_to_tensors, FrustumPointNetLoss,
                      g_type2class, g_class2type, g_type2onehotclass,
                      g_type_mean_size, g_mean_size_arr)
from ..train.provider import compute_box3d_iou

# PointNet module components
class PointNetInstanceSeg(nn.Module):
    def __init__(self,n_classes=3,n_channel=3):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:1 for single class
        :param one_hot_vec:[bs,3] - Note: Input is still 3D one-hot vector
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        # Fix: Input channels should match the original one-hot dimension (3)
        self.dconv1 = nn.Conv1d(64 + 1024 + 3, 512, 1)  # Keep original 3-dim one-hot
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec): # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :param one_hot_vec: [bs,3]: original 3D one-hot vector
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        # Fix: Keep original one-hot dimensions
        expand_one_hot_vec = one_hot_vec.view(bs, -1, 1)  # bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,1027,1
        expand_global_feat_repeat = expand_global_feat.repeat(1, 1, n_pts)  # bs,1027,n
        concat_feat = torch.cat([out2, expand_global_feat_repeat], 1)  # bs,1091,n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=3):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:1 for single class
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512+n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts,one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        # Initialize batch norm layers with specific settings for small batches
        self.bn1 = nn.BatchNorm1d(128, momentum=0.01, eps=1e-3)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.01, eps=1e-3)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.01, eps=1e-3)
        self.fcbn1 = nn.BatchNorm1d(256, momentum=0.01, eps=1e-3)
        self.fcbn2 = nn.BatchNorm1d(128, momentum=0.01, eps=1e-3)

        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        if bs == 1:
            # For batch size 1, duplicate the input to avoid batch norm issues
            pts = pts.repeat(2, 1, 1)
            one_hot_vec = one_hot_vec.repeat(2, 1)
            
        x = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = self.dropout(x)
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.dropout(x)
        x = self.fc3(x)  # bs,3

        if bs == 1:
            # If we duplicated the batch, take only the first result
            x = x[0:1]
            
        return x

class FrustumPointNetv1(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.InsSeg = PointNetInstanceSeg(n_classes=3, n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)

    def forward(self, data_dicts):
        """
        Forward pass of the network
        Args:
            data_dicts: Dictionary containing:
                - point_cloud: (bs,N,4,n_points) - batched point clouds
                - one_hot: (bs,3) - one hot vectors
                Other fields are optional for inference
        Returns:
            Dictionary containing network outputs
        """
        point_cloud = data_dicts['point_cloud']  # (bs,N,4,n_points)
        one_hot = data_dicts['one_hot']  # (bs,3)
        
        bs = point_cloud.shape[0]
        n_obj = point_cloud.shape[1]
        n_pts = point_cloud.shape[3]
        
        # Reshape inputs
        point_cloud = point_cloud.view(bs * n_obj, 4, n_pts)  # (bs*N,4,n_pts)
        point_cloud = point_cloud[:,:self.n_channel,:]  # Use only specified channels
        
        # Expand one_hot for each object
        one_hot = one_hot.unsqueeze(1).expand(-1, n_obj, -1)  # (bs,N,3)
        one_hot = one_hot.reshape(bs * n_obj, -1)  # (bs*N,3)
        
        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(point_cloud, one_hot)  # (bs*N,n_pts,2)
        
        # Get mask prediction from logits
        mask_prob = F.softmax(logits, dim=2)  # (bs*N,n_pts,2)
        mask = (mask_prob[:, :, 1] > mask_prob[:, :, 0]).float()  # (bs*N,n_pts)
        
        # Mask Point Centroid
        object_pts_xyz = point_cloud[:, :3, :]  # (bs*N,3,n_pts)
        mask_expanded = mask.unsqueeze(1)  # (bs*N,1,n_pts)
        
        # Calculate masked points mean (centroid)
        mask_sum = torch.clamp(mask_expanded.sum(2, keepdim=True), min=1)  # (bs*N,1,1)
        mask_xyz_mean = (mask_expanded * object_pts_xyz).sum(2, keepdim=True) / mask_sum  # (bs*N,3,1)
        
        # Get masked points
        mask_expanded = mask_expanded.expand_as(object_pts_xyz)  # (bs*N,3,n_pts)
        object_pts_xyz_masked = object_pts_xyz * mask_expanded  # (bs*N,3,n_pts)
        
        # Center using mask centroid
        object_pts_xyz_masked = object_pts_xyz_masked - mask_xyz_mean.expand_as(object_pts_xyz_masked)
        
        # T-Net and Object Estimation
        center_delta = self.STN(object_pts_xyz_masked, one_hot)  # (bs*N,3)
        stage1_center = center_delta + mask_xyz_mean.squeeze(2)  # (bs*N,3)
        
        # Get final centered points
        object_pts_xyz_new = object_pts_xyz_masked - center_delta.unsqueeze(2).expand_as(object_pts_xyz_masked)
        
        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot)  # (bs*N,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        
        # Parse outputs with original batch size and number of objects
        outputs = parse_output_to_tensors(box_pred, logits, mask, stage1_center, bs, n_obj)
        
        # Add box center to outputs
        outputs['box3d_center'] = outputs['center_boxnet'] + outputs['stage1_center']
        
        return outputs

class FrustumPointNetLoss(nn.Module):
    def forward(self, end_points, labels, valid_mask=None):
        """
        Args:
            end_points: dict from network forward pass
            labels: dict of ground truth labels
            valid_mask: (bs,N) tensor indicating which objects are valid (not padding)
        Returns:
            loss: pytorch scalar tensor
            loss_dict: dict of losses for logging
        """
        # If no valid_mask provided, assume all objects are valid
        if valid_mask is None:
            valid_mask = torch.ones_like(end_points['box3d_center'][:,:,0])
        
        # Get batch size and number of objects
        bs = end_points['logits'].shape[0]
        n_obj = end_points['logits'].shape[1]
        
        # Compute segmentation loss
        logits = end_points['logits'].view(-1, 2)  # (bs*N*n_pts,2)
        seg_label = labels['seg_label'].view(-1)  # (bs*N*n_pts,)
        valid_pts = (seg_label != -1)
        seg_loss = F.cross_entropy(logits[valid_pts], seg_label[valid_pts])
        
        # Compute center loss
        center_loss = torch.zeros(1).cuda()
        valid_obj_mask = valid_mask.view(-1)  # (bs*N,)
        if valid_obj_mask.sum() > 0:
            center_dist = torch.norm(end_points['box3d_center'].view(-1,3)[valid_obj_mask] - 
                                  labels['box3d_center'].view(-1,3)[valid_obj_mask], dim=1)
            center_loss = huber_loss(center_dist, delta=2.0)
            center_loss = center_loss.mean()
        
        # Compute heading loss
        heading_class_loss = torch.zeros(1).cuda()
        heading_residual_normalized_loss = torch.zeros(1).cuda()
        if valid_obj_mask.sum() > 0:
            heading_class_label = labels['heading_class_label'].view(-1)[valid_obj_mask]  # (n_valid,)
            heading_residual_label = labels['heading_residual_label'].view(-1)[valid_obj_mask]  # (n_valid,)
            
            heading_scores = end_points['heading_scores'].view(-1, NUM_HEADING_BIN)[valid_obj_mask]  # (n_valid,NH)
            heading_residual_normalized = end_points['heading_residual_normalized'].view(-1, NUM_HEADING_BIN)[valid_obj_mask]  # (n_valid,NH)
            
            heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)
            
            # Only compute heading residual loss for valid objects with correct heading class
            heading_class_pred = torch.argmax(heading_scores, dim=1)  # (n_valid,)
            valid_residual = (heading_class_pred == heading_class_label)
            if valid_residual.sum() > 0:
                heading_residual_normalized_loss = huber_loss(
                    heading_residual_normalized[valid_residual, heading_class_label[valid_residual]],
                    heading_residual_label[valid_residual],
                    delta=1.0).mean()
        
        # Compute size loss
        size_class_loss = torch.zeros(1).cuda()
        size_residual_normalized_loss = torch.zeros(1).cuda()
        if valid_obj_mask.sum() > 0:
            size_class_label = labels['size_class_label'].view(-1)[valid_obj_mask]  # (n_valid,)
            size_residual_label = labels['size_residual_label'].view(-1,3)[valid_obj_mask]  # (n_valid,3)
            
            size_scores = end_points['size_scores'].view(-1, NUM_SIZE_CLUSTER)[valid_obj_mask]  # (n_valid,NS)
            size_residual_normalized = end_points['size_residual_normalized'].view(-1, NUM_SIZE_CLUSTER, 3)[valid_obj_mask]  # (n_valid,NS,3)
            
            size_class_loss = F.cross_entropy(size_scores, size_class_label)
            
            # Only compute size residual loss for valid objects with correct size class
            size_class_pred = torch.argmax(size_scores, dim=1)  # (n_valid,)
            valid_residual = (size_class_pred == size_class_label)
            if valid_residual.sum() > 0:
                size_residual_normalized_loss = huber_loss(
                    size_residual_normalized[valid_residual, size_class_label[valid_residual]],
                    size_residual_label[valid_residual],
                    delta=1.0).mean()
        
        # Compute corner loss
        corner_loss = torch.zeros(1).cuda()
        if valid_obj_mask.sum() > 0:
            corners_3d_pred = get_box3d_corners_helper(
                end_points['box3d_center'].view(-1,3)[valid_obj_mask],
                end_points['heading_residual'].view(-1,NUM_HEADING_BIN)[valid_obj_mask],
                end_points['size_residual'].view(-1,NUM_SIZE_CLUSTER,3)[valid_obj_mask])  # (n_valid,8,3)
            corners_3d_gt = labels['corners_3d'].view(-1,8,3)[valid_obj_mask]  # (n_valid,8,3)
            corners_3d_gt_flip = get_box3d_corners_helper(
                labels['box3d_center'].view(-1,3)[valid_obj_mask],
                labels['heading_residual_label'].view(-1)[valid_obj_mask] + np.pi,
                labels['size_residual_label'].view(-1,3)[valid_obj_mask])  # (n_valid,8,3)
            
            corners_dist = torch.min(
                torch.norm(corners_3d_pred - corners_3d_gt, dim=2).mean(dim=1),
                torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=2).mean(dim=1))
            corner_loss = huber_loss(corners_dist, delta=1.0).mean()
        
        # Weight losses
        loss = 0.5 * seg_loss + \
               center_loss + \
               0.1 * heading_class_loss + \
               heading_residual_normalized_loss + \
               0.1 * size_class_loss + \
               size_residual_normalized_loss + \
               corner_loss
        
        loss_dict = {
            'total_loss': loss,
            'seg_loss': seg_loss,
            'center_loss': center_loss,
            'heading_class_loss': heading_class_loss,
            'heading_residual_normalized_loss': heading_residual_normalized_loss,
            'size_class_loss': size_class_loss, 
            'size_residual_normalized_loss': size_residual_normalized_loss,
            'corner_loss': corner_loss
        }
        
        return loss, loss_dict

if __name__ == '__main__':
    from ..train.provider import FrustumDataset
    dataset = FrustumDataset(npoints=1024, split='val',
        rotate_to_center=True, random_flip=False, random_shift=False, one_hot=True,
        overwritten_data_path='kitti/frustum_caronly_val.pickle',
        gen_ref = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = FrustumPointNetv1().cuda()
    for batch, data_dicts in enumerate(dataloader):
        data_dicts_var = {key: value.squeeze().cuda() for key, value in data_dicts.items()}
        result = model(data_dicts_var)

        print()
        for key,value in result.items():
            print(key,value.shape)
        print()
        input()
    '''
    total_loss tensor(50.4213, device='cuda:0', grad_fn=<AddBackward0>)
    mask_loss tensor(5.5672, device='cuda:0', grad_fn=<MulBackward0>)
    heading_class_loss tensor(2.5698, device='cuda:0', grad_fn=<MulBackward0>)
    size_class_loss tensor(0.9636, device='cuda:0', grad_fn=<MulBackward0>)
    heading_residual_normalized_loss tensor(9.8356, device='cuda:0', grad_fn=<MulBackward0>)
    size_residual_normalized_loss tensor(4.0655, device='cuda:0', grad_fn=<MulBackward0>)
    stage1_center_loss tensor(4.0655, device='cuda:0', grad_fn=<MulBackward0>)
    corners_loss tensor(26.4575, device='cuda:0', grad_fn=<MulBackward0>)
    
    seg_acc 16.07421875
    iou2d 0.066525616
    iou3d 0.045210287
    iou3d_acc 0.0
    '''
    '''
    data_dicts = {
        'point_cloud': torch.zeros(size=(32,1024,4),dtype=torch.float32).transpose(2, 1),
        'rot_angle': torch.zeros(32).float(),
        'box3d_center': torch.zeros(32,3).float(),
        'size_class': torch.zeros(32),
        'size_residual': torch.zeros(32,3).float(),
        'angle_class': torch.zeros(32).long(),
        'angle_residual': torch.zeros(32).float(),
        'one_hot': torch.zeros(32,3).float(),
        'seg': torch.zeros(32,1024).float()
    }
    data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
    model = FrustumPointNetv1().cuda()
    losses, metrics= model(data_dicts_var)

    print()
    for key,value in losses.items():
        print(key,value)
    print()
    for key,value in metrics.items():
        print(key,value)
    '''