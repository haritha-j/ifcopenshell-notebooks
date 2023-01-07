# viusalize elements / relationships

import json
import math
import uuid

import ifcopenshell
import numpy as np
import open3d as o3d
from OCC.Core.gp import gp_Pnt
from utils.JupyterIFCRenderer import JupyterIFCRenderer

import plotly.graph_objects as go
import plotly.express as px

from src.geometry import vector_norm
from src.elements import *


# visualize ifc model and point cloud simultaneously
def vis_ifc_and_cloud(ifc, cloud, colour="#abe000"):
    viewer = JupyterIFCRenderer(ifc, size=(400, 300))
    gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud]
    viewer.DisplayShape(gp_pnt_list, '#abe000')
    return viewer


# reover axis direction from six trig values starting from index k
def get_direction_from_trig(preds, k):
    return [math.atan2(preds[k], preds[k+1]), 
        math.atan2(preds[k+2], preds[k+3]), 
        math.atan2(preds[k+4], preds[k+5])]  


# reover axis direction from 2 position values starting from index k, j
def get_direction_from_position(preds, k, j):
    dir = [(preds[k] - preds[j]), (preds[k+1] - preds[j+1]), (preds[k+2] - preds[j+2])]
    return vector_norm(dir)


# visualize predictions side by side with ifc
def visualize_predictions(cloud, element, preds_list, blueprint, use_directions = True, visualize=True):
    ifc = setup_ifc_file(blueprint)
    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]

    ifc_info = {"owner_history": owner_history,
        "project": project,
       "context": context, 
       "floor": floor}
    
    for preds in preds_list:
        if element == 'pipe':
            pm = {'r':preds[0], 'l':preds[1] }
            pm['d'] = get_direction_from_trig(preds, 2)
            pm['p'] = [-((pm['l']*pm['d'][i])/2) for i in range(3)]
            #print(pm)
            
            create_IfcPipe(pm['r'], pm['l'], pm['d'], pm['p'], ifc, ifc_info)
            
        elif element == 'elbow':
            pm = {'r':preds[0], 'x':preds[1], 'y':preds[2]}

            theta = math.atan(pm['x']/pm['y'])
            pm['axis_dir'] = [math.cos(theta), math.sin(theta)]
            # pm['p'] = [0.0, 0.0, 0.0]
            pm['a'] = math.atan2(preds[3], preds[4])
            pm['p'] = [preds[5]*1000, preds[6]*1000, preds[7]*1000]
            pm['d'] = get_direction_from_trig(preds, 8)
            
            create_IfcElbow(pm['r'], pm['a'], pm['d'], pm['p'], pm['x'],
                            pm['y'], pm['axis_dir'], ifc, ifc_info)
            
        elif element == 'tee':
            pm = {'r1':preds[0], 'l1':preds[1], 'r2':preds[2],'l2':preds[3]}
            pm['p2'] = [preds[4]*1000, preds[5]*1000, preds[6]*1000]

            if use_directions:
                pm['d1'] = get_direction_from_trig(preds, 7)       
                pm['d2'] = get_direction_from_trig(preds, 13)
                pm['p1'] = (np.array(pm['p2']) - (np.array(pm['d1']) * np.array(pm['l1']) * 0.5)).tolist()
            else:
                pm['d1'] = get_direction_from_position(preds, 7, 4)
                pm['d2'] = get_direction_from_position(preds, 10, 7)
                pm['p2'] = [preds[7]*1000, preds[8]*1000, preds[9]*1000]
            
            create_IfcTee(pm['r1'], pm['r2'], pm['l1'], pm['l2'], pm['d1'], 
                        pm['d2'], pm['p1'], pm['p2'], ifc, ifc_info)

    if visualize:
        return vis_ifc_and_cloud(ifc, cloud), ifc
    else:
        return ifc


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