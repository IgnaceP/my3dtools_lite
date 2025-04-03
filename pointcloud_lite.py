import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy
import pandas as pd
from tqdm import tqdm


# class to hold tide data
# -------------------------------------------------------------
class PointCloud:
    def __init__(self, ply_path=None, points=None, colors=None, name=None, labels=None):
        """

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Class object to describe dense cloud
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        :param ply_path: path to .ply file (string, Required)
        :param name: name of the point cloud (string, Optional)


        """
        if ply_path != None:
            pcd = o3d.io.read_point_cloud(ply_path)
            self.labels = np.zeros_like(np.arange(len(np.asarray(pcd.points))))

        else:
            pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
            if type(colors) == np.ndarray:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            if type(labels) != np.ndarray:
                if labels == None:
                    self.labels = np.zeros_like(np.arange(len(points)))
            else:
                self.labels = labels

        self.pcd = pcd
        self.arr = np.asarray(pcd.points);
        self.points = self.arr.copy()
        self.points[np.isnan(self.points)] = 0
        self.X = self.points[:, 0]
        self.Y = self.points[:, 1]
        self.Z = self.points[:, 2]

        self.n = len(self.points)
        self.colors = np.asarray(pcd.colors)
        # self.colors[np.isnan(self.points)] = 0

        self.color_dists = {}
        self.point_dists = {}

        self.original_points = self.points.copy()

        self.cluster_params = {'esp': 0.05, 'min_samples': 100}


    def plot(self, colors=None, ax=None, plot_ax_labels=True, every_x_point=1, aspect=None, **kwargs):
        """
        Method to plot
        :param colors: numpy array of n_points length to color the scatter markers (numpy array, Optional, defaults to the color array)
        """

        if type(colors) != np.ndarray:
            if colors == None:
                colors = self.colors

        if ax:
            fig = ax.get_figure()
            gridspecs = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(gridspecs, projection="3d")
        else:
            fig = plt.figure()
            ax = plt.axes(projection="3d")

        if len(colors.shape) > 1:
            # ax.scatter3D(self.points[::every_x_point, 0], self.points[::every_x_point, 1], self.points[::every_x_point, 2], color=colors[::every_x_point],**kwargs)
            ax.scatter(self.points[::every_x_point, 0], self.points[::every_x_point, 1],
                       self.points[::every_x_point, 2], color=colors[::every_x_point], **kwargs)
        else:
            sc = ax.scatter3D(self.points[::every_x_point, 0], self.points[::every_x_point, 1],
                              self.points[::every_x_point, 2], c=colors[::every_x_point], cmap='tab20', **kwargs)
            sc = ax.scatter(self.points[::every_x_point, 0], self.points[::every_x_point, 1],
                            self.points[::every_x_point, 2], c=colors[::every_x_point], cmap='tab20', **kwargs)
            fig.colorbar(sc)

        if plot_ax_labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        if aspect:
            ax.set_aspect(aspect)
        return fig, ax

    def transformPCA(self, return_PCA=False):
        """
        Method to transform the coordinates of the points along its principal components
        """
        pca = PCA(n_components=3)
        pcd_pca = pca.fit_transform(self.points)

        if return_PCA:
            return PointCloud(points=pcd_pca, colors=self.colors, labels=self.labels), pca
        return PointCloud(points=pcd_pca, colors=self.colors, labels=self.labels)

    def sample(self, reduction_factor=100):
        """
        Method to reduce the size of a point cloud.
        :param reduction_factor: only keep 1/reduction_factor of the points

        :return: the reduced point cloud

        !!! Warning, a lot of points are lost !!!
        !!! Only use for test purposes !!!
        """

        return PointCloud(points=self.points[::reduction_factor], colors=self.colors[::reduction_factor],
                          labels=self.labels[::reduction_factor])

    def mask(self, mask):
        return PointCloud(points=self.points[mask, :], colors=self.colors[mask, :], labels=self.labels[mask])

    def getCentroid(self):
        return np.mean(self.points, axis=0)

    def merge(self, pointcloud):
        """
        method to merge two point clouds
        :param pointcloud: other PointCloud object (PointCloud object, Required)
        :returns merged_pointcloud: new merged PointCloud
        """

        merged_points = np.vstack((self.points, pointcloud.points))
        merged_colors = np.vstack((self.colors, pointcloud.colors))
        merged_labels = np.concatenate((self.labels, pointcloud.labels))

        return PointCloud(points=merged_points, colors=merged_colors, labels=merged_labels)

    def copy(self):
        return deepcopy(self)

    def writePLY(self, fn):
        """
        Method to write/save a point cloud to a PLY file
        :param fn: path to file
        """

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.points))
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.io.write_point_cloud(fn, pcd)

    def normalize(self):
        """
        method to normalize the points to their mean
        :return:
        """

        self.X -= np.mean(self.X)
        self.Y -= np.mean(self.Y)
        self.Z -= np.mean(self.Z)

    def fitSurface(self, order=3):

        # the number of parameters to
        ncols = (order + 1) ** 2
        G = np.zeros((self.X.size, ncols))
        ij = itertools.product(range(order + 1), range(order + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = self.X ** i * self.Y ** j
        m, _, _, _ = np.linalg.lstsq(G, self.Z, rcond=None)
        return m

        return m


    def computeDTM(self, res=1, bottom_quantile=.05, top_quantile=.95):
        """
        Compute the Digital Terrain Model (DTM) and Digital Surface Model (DSM)
        from a point cloud using quantiles within grid cells.

        Arguments:
            pcl: PointCloud object
            res: Resolution for grid cell size (default is 1)

        Returns:
            DTM: Digital Terrain Model (bottom quantile)
            DSM: Digital Surface Model (top quantile)
            DEM: Digital Elevation Model (median quantile)
            XYZ: Array of points with computed DTM values
        """

        # Calculate grid dimensions based on resolution
        cols = int((self.X.max() - self.X.min()) // res) + 1
        rows = int((self.Y.max() - self.Y.min()) // res) + 1

        # Initialize DTM, DSM, and DEM arrays
        DTM = np.full([rows, cols], np.nan)
        DSM = np.full([rows, cols], np.nan)
        DEM = np.full([rows, cols], np.nan)

        # Calculate grid indices for points
        x_indices = np.floor((self.X - self.X.min()) / res).astype(int)
        y_indices = np.floor((self.Y - self.Y.min()) / res).astype(int)

        # Filter valid indices
        valid_mask = (x_indices >= 0) & (x_indices < cols) & (y_indices >= 0) & (y_indices < rows)
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        z_values = self.Z[valid_mask]

        # Accumulate Z-values per grid cell
        grid_data = {}
        for x, y, z in zip(x_indices, y_indices, z_values):
            if (x, y) not in grid_data:
                grid_data[(x, y)] = []
            grid_data[(x, y)].append(z)

        # Populate DTM, DSM, and DEM matrices
        for (x, y), z_vals in tqdm(grid_data.items(), desc='Computing DTM, DSM, DEM'):
            z_array = np.array(z_vals)
            DTM[y, x] = np.quantile(z_array, bottom_quantile)
            DSM[y, x] = np.quantile(z_array, top_quantile)
            DEM[y, x] = np.median(z_array)

        # Create XYZ array for DTM points
        X_coords, Y_coords = np.meshgrid(
            np.arange(cols) * res + self.X.min() + res / 2,
            np.arange(rows) * res + self.Y.min() + res / 2
        )

        XYZ = np.column_stack((X_coords.flatten(), Y_coords.flatten(), DTM.flatten()))
        XYZ = pd.DataFrame(XYZ, columns=['x', 'y', 'z'])

        DEM = np.flip(DEM, axis=0)
        DTM = np.flip(DTM, axis=0)
        DSM = np.flip(DSM, axis=0)

        return {'DTM': DTM, 'DSM': DSM, 'DEM': DEM, 'XYZ': XYZ}

