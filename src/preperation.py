import numpy as np
import math
import random
import os
import json
import torch
import copy

import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from path import Path


def read_pcd(file):
    pcd = o3d.io.read_point_cloud(str(file))
    return np.asarray(pcd.points)


class Normalize(object):
    def __call__(self, data):
        pointcloud, properties = data[0], data[1]
        assert len(pointcloud.shape)==2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_factor = np.max(np.linalg.norm(norm_pointcloud, axis=1))
        norm_pointcloud /= norm_factor
        properties_norm = properties/norm_factor
        #print(properties,properties_norm)

        return  (norm_pointcloud, properties_norm)


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig


def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    

# trsansform the centerpoint of a cloud to origin
def center_bbox(cloud):
    bbox_max = np.amax(cloud, 0)
    bbox_min = np.amin(cloud, 0)
    print((bbox_min+bbox_max)/2)


class RandRotation_z(object):
    def __call__(self, data):
        pointcloud, properties = data[0], data[1]
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  (rot_pointcloud, properties)
    

class RandomNoise(object):
    def __call__(self, data):
        pointcloud, properties = data[0], data[1]
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  (noisy_pointcloud, properties)


class ToTensor(object):
    def __call__(self, data):
        pointcloud, properties = data[0], data[1]
        assert len(pointcloud.shape)==2

        return (torch.from_numpy(pointcloud).float(), torch.from_numpy(properties).float())