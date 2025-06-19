import numpy as np
import torch

def rotate_pc_along_y(pc, rot_angle):
    """
    Rotate point cloud along y axis
    Args:
        pc: (N, 3) point cloud
        rot_angle: rotation angle in radians
    Returns:
        rotated pc
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, 0, sinval],
                      [0, 1, 0],
                      [-sinval, 0, cosval]])
    return np.dot(pc, rotmat)

def compute_box_3d(center, size, heading_angle):
    """
    Compute 8 corners of 3D bounding box
    Args:
        center: (3,) array of box center
        size: (3,) array of box size (l,w,h)
        heading_angle: heading angle in radians
    Returns:
        corners_3d: (8,3) array of vertices
    """
    l, w, h = size
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotate
    corners_3d = rotate_pc_along_y(corners_3d.T, heading_angle).T
    
    # Translate
    corners_3d = corners_3d.T + center
    return corners_3d

def box3d_iou(corners1, corners2):
    """
    Compute 3D bounding box IoU
    Args:
        corners1: (8,3) array of vertices for the 1st box
        corners2: (8,3) array of vertices for the 2nd box
    Returns:
        iou: 3D IoU scalar
    """
    # Compute volume of first box
    a = np.linalg.norm(corners1[1] - corners1[0])
    b = np.linalg.norm(corners1[2] - corners1[1])
    c = np.linalg.norm(corners1[4] - corners1[0])
    vol1 = a * b * c
    
    # Compute volume of second box
    a = np.linalg.norm(corners2[1] - corners2[0])
    b = np.linalg.norm(corners2[2] - corners2[1])
    c = np.linalg.norm(corners2[4] - corners2[0])
    vol2 = a * b * c
    
    # TODO: Compute intersection volume
    # This is a simplified IoU calculation
    # For accurate IoU, we need to implement actual 3D intersection volume computation
    
    # For now, use center distance as a proxy
    center1 = np.mean(corners1, axis=0)
    center2 = np.mean(corners2, axis=0)
    dist = np.linalg.norm(center1 - center2)
    
    # If centers are far, IoU is 0
    if dist > (a + b + c):
        return 0.0
    
    # Approximate IoU based on center distance
    intersection_vol = max(0, min(vol1, vol2) * (1 - dist/(a + b + c)))
    iou = intersection_vol / (vol1 + vol2 - intersection_vol)
    
    return iou

def nms_3d(boxes, scores, iou_threshold):
    """
    3D Non-Maximum Suppression
    Args:
        boxes: (N, 8, 3) array of box corners
        scores: (N,) array of scores
        iou_threshold: IoU threshold for NMS
    Returns:
        keep: array of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Sort boxes by score
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU of the picked box with the rest
        ious = np.array([box3d_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Remove boxes with IoU > threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep 