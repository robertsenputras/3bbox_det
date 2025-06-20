# Create the bounding box in open3d
import numpy as np
import open3d as o3d

class BBoxVisualizer:
    def __init__(self, pcl: np.ndarray, bbox: np.ndarray, rgb: np.ndarray = None):
        """
        pcl: (3, H, W) numpy array of XYZ coordinates
        bbox: (N, 8, 3) numpy array of box corner coordinates
        rgb: (H, W, 3) optional color image for the points
        """
        # flatten point cloud into (num_pts, 3)
        H, W = pcl.shape[1], pcl.shape[2]
        pts = pcl.reshape(3, -1).T  # (H*W, 3)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        if rgb is not None:
            colors = (rgb.reshape(-1, 3) / 255.0).astype(np.float64)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.boxes = []
        for corners in bbox:
            self.boxes.append(self._make_lineset(corners))

    def _make_lineset(self, corners: np.ndarray) -> o3d.geometry.LineSet:
        """
        Given 8 corners in object coord order, build a LineSet connecting edges.
        corners: (8, 3) array
        """
        # Define the 12 edges of a box by index pairs
        lines = [
            [0,1], [1,2], [2,3], [3,0],  # bottom face
            [4,5], [5,6], [6,7], [7,4],  # top face
            [0,4], [1,5], [2,6], [3,7],  # vertical edges
        ]
        colors = [[1, 0, 0] for _ in lines]  # red boxes
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        return ls

    def visualize(self, line_width: float = 5.0):
        print('test')
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.pcd)
        for box in self.boxes:
            vis.add_geometry(box)
        opt = vis.get_render_option()
        opt.line_width = 5.0  # increase from default (1.0)
        vis.run()
        vis.destroy_window()
