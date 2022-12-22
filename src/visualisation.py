# viusalize elements / relationships

import json
import math
import uuid

import ifcopenshell
import numpy as np
import open3d as o3d
from OCC.Core.gp import gp_Pnt
from utils.JupyterIFCRenderer import JupyterIFCRenderer

from src.geometry import get_corner, get_oriented_bbox, sq_distance
from src.elements import *


# visualize ifc model and point cloud simultaneously
def vis_ifc_and_cloud(ifc, cloud, colour="#abe000"):
    viewer = JupyterIFCRenderer(ifc, size=(400, 300))
    gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud.points]
    viewer.DisplayShape(gp_pnt_list, '#abe000')
    return viewer


# visualize predictions side by side with ifc
def visualize_predictions(cloud, element, preds, blueprint):
    ifc = setup_ifc_file(blueprint)
    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]

    ifc_info = {"owner_history": owner_history,
        "project": project,
       "context": context, 
       "floor": floor}
    
    if element == 'pipe':
        pm = {'r':preds[0], 'l':preds[1], 'd':[preds[2], preds[3], preds[4]] }
        pm['p'] = [-((pm['l']*pm['d'][i])/2) for i in range(3)]
        #print(pm)
        
        create_IfcPipe(pm['r'], pm['l'], pm['d'], pm['p'], ifc, ifc_info)
        
    elif element == 'elbow':
        pm = {'r':preds[0], 'x':preds[1], 'y':preds[2], 'd':[preds[3], preds[4], preds[5]], 
              'a':preds[6] }
        theta = math.atan(pm['x']/pm['y'])
        pm['axis_dir'] = [math.cos(theta), math.sin(theta)]
        # pm['p'] = [0.0, 0.0, 0.0]
        pm['p'] = [preds[7]*1000, preds[8]*1000, preds[9]*1000]
        print(pm)
        
        create_IfcElbow(pm['r'], pm['a'], pm['d'], pm['p'], pm['x'],
                        pm['y'], pm['axis_dir'], ifc, ifc_info)
        
    elif element == 'tee':
        pm = {'r1':preds[0], 'l1':preds[1], 'r2':preds[2],'l2':preds[3], 
              'd1':[preds[4], preds[5], preds[6]], 
              'd2':[preds[7], preds[8], preds[9]] }
        pm['p1'] = [preds[10]*1000, preds[11]*1000, preds[12]*1000]
        pm['p2'] = (np.array(pm['p1']) + (np.array(pm['d1']) * np.array(pm['l1']) * 0.5)).tolist()
        print(pm)
        
        create_IfcTee(pm['r1'], pm['r2'], pm['l1'], pm['l2'], pm['d1'], 
                      pm['d2'], pm['p1'], pm['p2'], ifc, ifc_info)

    return vis_ifc_and_cloud(ifc, cloud)


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