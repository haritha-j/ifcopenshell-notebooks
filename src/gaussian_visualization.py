# adapted from https://github.com/sitzikbs/gmm_tutorial/blob/master/estimate_gmm_sklearn.py and modified for full GMM visualization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os


# plot ellipsoid along any axis
# required for plotting full gmms rather than just the diagonal
def plot_ellipsoid(mean, cov, ax, w=0, sigma_multiplier=10):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    cov = cov * sigma_multiplier

    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j], y[i, j], z[i, j] = (
                np.dot([x[i, j], y[i, j], z[i, j]], np.linalg.cholesky(cov).T) + mean
            )

    # darken the color as the weight increases
    cmap = cmx.ScalarMappable()
    cmap.set_cmap("jet")
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2)


def visualize_3d_gmm(points, w, mu, stdev, export=True):
    """
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    """
    lim = 0.4
    n_gaussians = mu.shape[0]
    # n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection="3d")
    axes.set_xlim([-lim, lim])
    axes.set_ylim([-lim, lim])
    axes.set_zlim([-lim, lim])
    plt.set_cmap("Set1")
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    print(mu.shape, stdev.shape)
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(
            points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i]
        )
        # plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)
        plot_ellipsoid(mu[i], stdev[i], axes, w[i])

    plt.title("3D GMM")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists("images/"):
            os.mkdir("images/")
        plt.savefig("images/3D_GMM_demonstration.png", dpi=100, format="png")
    plt.show()


def plot_sphere(w=0, c=[0, 0, 0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    """
    plot a sphere surface
    Input:
        c: 3 elements list, sphere center
        r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
        subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
        ax: optional pyplot axis object to plot the sphere in.
        sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
    Output:
        ax: pyplot axis object
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[
        0.0 : pi : complex(0, subdev), 0.0 : 2.0 * pi : complex(0, subdev)
    ]
    x = sigma_multiplier * r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier * r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier * r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap("jet")
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax


def visualize_2D_gmm(points, w, mu, stdev, export=True):
    """
    plots points and their corresponding gmm model in 2D
    Input:
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    """
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.set_cmap("Set1")
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(
                    mu[:, i],
                    width=(j + 1) * stdev[0, i],
                    height=(j + 1) * stdev[1, i],
                    fill=False,
                    color=[0.0, 0.0, 1.0, 1.0 / (0.5 * j + 1)],
                )
            )
        plt.title("GMM")
    plt.xlabel("X")
    plt.ylabel("Y")

    if export:
        if not os.path.exists("images/"):
            os.mkdir("images/")
        plt.savefig("images/2D_GMM_demonstration.png", dpi=100, format="png")

    plt.show()
