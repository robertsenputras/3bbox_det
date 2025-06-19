import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# File names for the dataset
frustum_file = 'pc_frustum.npy'  # Changed from pc.npy
box_file = 'pc_box.npy'          # Changed from bbox3d.npy
mask_file = 'pc_mask.npy'        # Changed from mask.npy
rgb_file = 'rgb.jpg'             # Kept the same since it's not used in the dataset class

class Classical3DBaseline:
    def __init__(self,
                 voxel_size: float = 0.02,
                 eps: float = 0.05,
                 min_points: int = 10):
        """
        Args:
            voxel_size: float – size (meters) for downsampling
            eps: float – DBSCAN neighborhood radius
            min_points: int – min points per cluster
        """
        self.voxel_size = voxel_size
        self.eps = eps
        self.min_points = min_points

    def _make_pointcloud(self, pcl: np.ndarray, rgb: np.ndarray = None):
        """
        Convert pcl (3,H,W) + optional rgb (H,W,3) → open3d.geometry.PointCloud
        Filters out zero‐depth points.
        """
        # reshape: (3,H,W) → (H*W,3)
        H, W = pcl.shape[1:]
        pts = pcl.reshape(3, -1).T            # → (H*W, 3)
        valid = pts[:, 2] > 0                 # drop invalid (zero or negative depth)
        pts = pts[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def _cluster_and_pca(self, pcd: o3d.geometry.PointCloud):
        """
        DBSCAN‐cluster a point cloud and compute 8 corner points for each box.
        Returns:
            down: downsampled PointCloud
            corners_list: list of (8,3) np.ndarray of box corners
        """
        # downsample
        # down = pcd.voxel_down_sample(self.voxel_size)
        down = pcd

        # DBSCAN
        labels = np.array(
            down.cluster_dbscan(eps=self.eps,
                                min_points=self.min_points,
                                print_progress=False)
        )
        corners_list = []
        for cid in range(labels.max() + 1):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            pts = np.asarray(down.points)[idx]
            centroid = pts.mean(axis=0)
            cov = np.cov((pts - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvecs = eigvecs[:, eigvals.argsort()[::-1]]

            # project to PCA frame
            pts_pca = (pts - centroid) @ eigvecs
            min_b, max_b = pts_pca.min(axis=0), pts_pca.max(axis=0)

            # build the 8 corners in PCA frame
            corners_pca = np.array([
                [min_b[0], min_b[1], min_b[2]],
                [max_b[0], min_b[1], min_b[2]],
                [max_b[0], max_b[1], min_b[2]],
                [min_b[0], max_b[1], min_b[2]],
                [min_b[0], min_b[1], max_b[2]],
                [max_b[0], min_b[1], max_b[2]],
                [max_b[0], max_b[1], max_b[2]],
                [min_b[0], max_b[1], max_b[2]],
            ])  # shape (8,3)

            # rotate & translate back to original coords
            corners_world = (eigvecs @ corners_pca.T).T + centroid
            corners_list.append(corners_world)

        return down, corners_list

    def run(self, pcl: np.ndarray, rgb: np.ndarray = None, use_seg_masks: np.ndarray = None):
        """
        Main entry point.
        Now returns: down_pcd, list_of_corner_arrays (each shape (8,3))
        """
        pcd = self._make_pointcloud(pcl, rgb)
        down_pcd, corners_list = self._cluster_and_pca(pcd)
        return down_pcd, corners_list

def get_hashes(data_path):
    """
    Get all the hash values from the data path.
    :param data_path: Path to the data directory.
    :return: List of hash values.
    """
    hashes = []
    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            hashes.append(dir_name)
    return hashes

def get_sample_data(data_path, hash_value):
    """
    Get sample data for a given hash value.
    :param data_path: Path to the data directory.
    :param hash_value: Hash value to get the sample data for.
    :return: Dictionary with sample data.
    """
    hash_path = os.path.join(data_path, hash_value)
    sample_data = {
        'bbox': np.load(os.path.join(hash_path, box_file)),
        'seg': np.load(os.path.join(hash_path, mask_file)),
        'pcl': np.load(os.path.join(hash_path, frustum_file)),
        'rgb': os.path.join(hash_path, rgb_file)
    }
    return sample_data



class BBoxVisualizer:
    def __init__(self, pcl: np.ndarray, bbox: np.ndarray, rgb: np.ndarray = None):
        H, W = pcl.shape[1], pcl.shape[2]
        pts = pcl.reshape(3, -1).T
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        if rgb is not None:
            colors = (rgb.reshape(-1, 3) / 255.0).astype(np.float64)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.boxes = [self._make_lineset(corners) for corners in bbox]

    def _make_lineset(self, corners: np.ndarray) -> o3d.geometry.LineSet:
        lines = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7],
        ]
        colors = [[1, 0, 0] for _ in lines]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        return ls

    def visualize(self, line_width: float = 5.0):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.pcd)
        for box in self.boxes:
            vis.add_geometry(box)
        opt = vis.get_render_option()
        opt.line_width = line_width  # increase from default (1.0)
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    # --- example usage ---
    # assume you loaded these four arrays from .npy files, etc.
    # bbox_gt: (12, 8, 3)   ← your ground-truth boxes (not used in baseline)
    # seg:     (12, 630,981) ← your instance masks (optional)
    # pcl:     (3, 630,981)  ← your frustum point cloud
    # rgb:     (630,981,3)   ← your RGB image

    data_path = 'data/dl_challenge'
    hashes = get_hashes(data_path)
    print(f"Found {len(hashes)} hashes in the data path.")

    # Get sample data for the first hash
    index = 0
    sample_hash = hashes[index]
    sample_data = get_sample_data(data_path, sample_hash)
    # Print the sample data keys and types
    print("Sample data keys and types:")
    for key, value in sample_data.items():
        print(f"{key}: {type(value)}", f"shape: {value.shape if isinstance(value, np.ndarray) else 'N/A'}")
        
    bbox_gt = sample_data['bbox']  # (12, 8, 3)
    seg = sample_data['seg']        # (12, 630, 981)
    pcl = sample_data['pcl']        # (3, 630, 981
    rgb = plt.imread(sample_data['rgb']) if sample_data['rgb'].endswith('.jpg') else None        # (630, 981, 3)

    # # Create ground truth bounding boxes visualizer
    # bbox_visualizer = BBoxVisualizer(pcl, bbox_gt, rgb)
    # bbox_visualizer.visualize()

    baseline = Classical3DBaseline(
        voxel_size=0.01,
        eps=0.01,
        min_points=20
    )
    down_pcd, corner_list = baseline.run(pcl, rgb, use_seg_masks=seg)

    # visualize the downsampled cloud + your estimated boxes
    bbox_visualizer = BBoxVisualizer(
        pcl=pcl,
        bbox=np.array(corner_list),
        rgb=np.asarray(down_pcd.colors) if down_pcd.has_colors() else None
    )
    bbox_visualizer.visualize()