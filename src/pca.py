import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook as tqdm

from src.visualisation import get_direction_from_trig
from src.plots import plot_error_graph


def get_PCA(cloud):
    pca = PCA(n_components=3)
    pca.fit(cloud)
    V = pca.components_.T
    return V


def visualise_PCA(cloud):
    V = get_PCA(cloud)
    x_pca_axis, y_pca_axis, z_pca_axis = V

    x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)

    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]

    fig = plt.figure(figsize=(16, 8))
    elevs = [-40, 30]
    azims = [-80, 20]
    plt.clf()

    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d", elev=elevs[i], azim=azims[i])
        # ax.set_position([0, 0, 0.95, 1])
        ax.scatter(x, y, z, marker="+", alpha=0.4)
        ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
    plt.show()


def compare_PCA_axis(cloud, params, cat):
    V = get_PCA(cloud)
    pca_x = V[0]
    if cat == "pipe":
        axis = get_direction_from_trig(params, 5)
    elif cat == "elbow":
        axis = get_direction_from_trig(params, 8)
    elif cat == "tee":
        axis = get_direction_from_trig(params, 7)

    # print(pca_x, axis)
    return math.degrees(math.acos(np.dot(pca_x, axis)))


def testset_PCA(cloud_list, inputs_list, testDataLoader, cat):
    for i in range(10):
        visualise_PCA(cloud_list[i].transpose(1, 0))
        dev = compare_PCA_axis(cloud_list[i].transpose(1, 0), inputs_list[i], cat)
        print(dev)

    deviations = []
    for j, data in tqdm(enumerate(testDataLoader), total=len((testDataLoader))):
        points = data["pointcloud"].numpy()
        labels = data["properties"].numpy()

        for i in range(points.shape[0]):
            deviations.append(
                compare_PCA_axis(points[i].transpose(1, 0), labels[i], cat)
            )
            if i == 0:
                visualise_PCA(points[i])

    print("avg", sum(deviations) / len(deviations))
    plot_error_graph(deviations, "PCA deviation")

    return deviations
