# visualisation of 3D gaussian mixture model of point cloud
# adapted from https://github.com/sitzikbs/gmm_tutorial/blob/master/estimate_gmm_sklearn.py

import numpy as np
import src.gaussian_visualization as vis
from sklearn.mixture import GaussianMixture
import open3d as o3d


def is_positive_definite(matrix):
    # Check if all eigenvalues are positive
    return np.all(np.linalg.eigvals(matrix) > 0)


## Generate synthetic data
N, D = 1000, 3  # number of points and dimenstinality

# load point cloud
cloud_path = "data/test/24102.pcd"
points = np.array(o3d.io.read_point_cloud(cloud_path).points)

# fit the gaussian model
# gmm = GaussianMixture(n_components=128, covariance_type='diag')
gmm = GaussianMixture(n_components=128, covariance_type="full")

gmm.fit(points)

# print(gmm.covariances_)

# visualize
if D == 2:
    vis.visualize_2D_gmm(
        points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T
    )
elif D == 3:
    vis.visualize_3d_gmm(points, gmm.weights_, gmm.means_, gmm.covariances_)
    # vis.visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
