import math
import random
import numpy as np
import open3d as o3d
from numpy.random import default_rng
from itertools import islice


# pick a point uniformly in a triangle
def uniform_triangle(u, v):
    while True:
        s = random.random()
        t = random.random()
        in_triangle = s + t <= 1
        p = s * u + t * v if in_triangle else (1 - s) * u + (1 - t) * v
        yield p


# add unifrom random noise to pointcloud
def add_noise(cl, noise_size, rng):
    min_point = [min(cl[:,i]) for i in range(3)]
    max_point = [max(cl[:,i]) for i in range(3)]
    
    noise = np.column_stack([rng.uniform(min_point[i], max_point[i], noise_size) for i in range(3)])
    noisy_points = np.vstack([cl[:len(cl)-noise_size], noise])
    
    return noisy_points


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# uniformly sample points within triangle
def sample_points(p, q, r, n):   
    it = uniform_triangle((q - p), (r - p))
    points = np.array(list(islice(it, 0, n)))
    points += p
    return points


# convert an element to a point cloud
def element_to_cloud(element, save_path=None, density=0):
    shape = element.Representation.Representations[0].Items[0]
    boundaries = np.array(shape.Coordinates.CoordList)
    
    # determine sampling target 
    point_count = len(boundaries)
    samples = 10 if density == 0 else math.ceil(density * 6 / point_count)
        
    # get additional points by sampling from mesh triangles
    limit = point_count -2
    #print (limit)
    centroids = []
    for j in range(0, point_count, 3):
        if j < limit:
            # centroids.append([(boundaries[j][k] + boundaries[j+1][k] 
            # + boundaries[j+2][k])/3 for k in range(3)])
            centroids.extend(sample_points((boundaries[j]), (boundaries[j+1]),
                                           (boundaries[j+2]), samples))
    boundaries = np.concatenate([boundaries, np.array(centroids)])
    #print(len(boundaries))
    
    # downsample to fixed length
    if density > 0:
        boundaries = boundaries[np.random.choice(boundaries.shape[0], density, 
                                                 replace=False), :]
    
    # convert to pointcloud
    if save_path is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(boundaries)
        o3d.io.write_point_cloud(save_path, pcd)
    return (boundaries)


# create smaller cleaned cloud by removing outliers and downsampling
def refine_cloud(file_path, n_points=1000):
    rng = default_rng()

    
    # read points
    f = open(file_path, 'r')
    points = f.readlines()
    points = [p.strip().split(' ')[:3] for p in points]
    for i in range(len(points)):
        points[i] = [float(p) for p in points[i]]
    
    # create cloud
    points = np.array(points)
    #print(points.shape)
    if len(points) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # outlier removal
    # TODO: experiment with parameters and try radius outlier removal
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)

    # further down sampling
    if len(ind) > n_points:
        #print('l', len(ind))
        sub_ind = rng.choice(len(ind), size=n_points, replace=False)
        cl = np.asarray(pcd.points)[sub_ind]

    elif len(ind) > 0:
        sub_ind = rng.choice(len(ind), size=n_points, replace=True)
        cl = np.asarray(pcd.points)[sub_ind]
        
    else:
        cl = cl.points
    #cl = cl.select_by_index(sub_ind)
    # o3d.io.write_point_cloud("sync.ply", cl)
    
    return np.asarray(cl)