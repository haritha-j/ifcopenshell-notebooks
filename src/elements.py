import os
import numpy as np
import math
import json
import random
import uuid
import ifcopenshell
import ifcopenshell.geom

from src.visualisation import *


# create a new blank ifc file
def setup_ifc_file(blueprint):

    ifc = ifcopenshell.open(blueprint)
    ifcNew = ifcopenshell.file(schema=ifc.schema)
    
    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]
    
    ifcNew.add(project) 
    ifcNew.add(owner_history) 
    ifcNew.add(context) 
    ifcNew.add(floor)

    return ifcNew


# return axis aligned bbox and centerpoint of any ifc element through mesh conversion (expensive)
def generic_element_bbox(ifc, element_type):    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.WELD_VERTICES, False)
    #settings.set(settings.USE_BREP_DATA, True)
    pipe = ifc.by_type(element_type)[0]
    #print("X")
    shape = ifcopenshell.geom.create_shape(settings, pipe)
    #print("Y")
    verts = shape.geometry.verts
    
    x = [verts[i] for i in range(0, len(verts), 3)]
    y = [verts[i+1] for i in range(0, len(verts), 3)]
    z = [verts[i+2] for i in range(0, len(verts), 3)]
    
    x_max, y_max, z_max = max(x), max(y), max(z)
    x_min, y_min, z_min = min(x), min(y), min(z)
    
    center = ((x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2)
    bbox = ((x_max - x_min), (y_max - y_min), (z_max - z_min))

    return bbox, center
    #return (2.0,2.0,2.0),(0.0,0.0,0.0)


# generate Ifc Pipe fitting from parameters
def create_IfcElbow(r, a, d, p, x, y, axis_dir, ifc, ifc_info, fill=False):
    cross_section = Circle_Section(r=r, ifcfile=ifc, fill=fill)

    beam = CreateElbow(ifc, container=ifc_info['floor'], name="elbow", 
                      section=cross_section, a=a, position=p,
                      direction=d, x=x, y=y, axis_dir=axis_dir, 
                       owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)


# return axis aligned bbox, centerpoint of elbow
def elbow_bbox(r, a, d, p, x, y, axis_dir, blueprint):
    temp = setup_ifc_file(blueprint)
    owner_history = temp.by_type("IfcOwnerHistory")[0]
    project = temp.by_type("IfcProject")[0]
    context = temp.by_type("IfcGeometricRepresentationContext")[0]
    floor = temp.by_type("IfcBuildingStorey")[0]

    temp_info = {"owner_history": owner_history,
        "project": project,
       "context": context, 
       "floor": floor}
    print(r, a, d, p, x, y)
    create_IfcElbow(r, a, d, p, x, y, axis_dir, temp, temp_info, fill=True)
    
    bbox, center = generic_element_bbox(temp, "IfcPipeFitting")
    del temp
    return bbox, center


# generate a random synthetic elbow
def create_elbow(config,  ifc, ifc_info, blueprint):
    # generate parameters
    r = random.uniform(config['radius_range'][0], config['radius_range'][1])
    a = random.uniform(config['angle_range'][0], config['angle_range'][1])
    
    d = []
    for ax in config['axis_direction_range']:
        d.append(random.uniform(ax[0], ax[1]))
    d_np = np.array(d)
    d = (d_np/np.linalg.norm(d_np)).tolist()

    p = []
    for coord in config['coordinate_range']:
        p.append(random.uniform(coord[0], coord[1]))
        
    # generate points on a 2D ring from the origin
    axis_placement = random.uniform(config['curvature_range'][0], 
                                    config['curvature_range'][1])*r
    #axis_placement = 50*r
    
    axis_ang = random.uniform(config['axis_angle_range'][0], 
                              config['axis_angle_range'][1])
    x = axis_placement * math.sin(axis_ang)
    y = axis_placement * math.cos(axis_ang)
    axis_dir = (math.cos(axis_ang), math.sin(axis_ang))
    
    # transform points to the elbow center and normalize
    bbox, centerpoint = elbow_bbox(r, a, d, p, x, y, axis_dir, blueprint)
    bbox_l2 = math.sqrt(bbox[0]*bbox[0] + bbox[1]*bbox[1] + bbox[2]*bbox[2])
    print('p', p, 'c', centerpoint)
    #p = [p[i] - centerpoint[i]*10000/bbox_l2 for i in range(3)]
    p = [-1* centerpoint[i]*10000/bbox_l2 for i in range(3)]
    r, x, y = 10*r/bbox_l2, 10*x/bbox_l2, 10*y/bbox_l2
    print('p', p, 'c', centerpoint, r, x)

#     print('bb', bbox, 'c', centerpoint, bbox_l2)
#     bbox2, centerpoint2 = elbow_bbox(r, a, d, p, x, y, axis_dir)
#     print('bb', bbox2, 'c', centerpoint2)

    create_IfcElbow(r, a, d, p, x, y, axis_dir, ifc, ifc_info)
    metadata = {'radius':r, "direction":d, "angle":a, "position":p, 
                'axis_x':x, 'axis_y':y}
    
    return metadata


# generate IfcBeam from parameters
def create_IfcPipe(r, l, d, p, ifc, ifc_info):
    cross_section = Circle_Section(r=r, ifcfile=ifc)

    beam = CreateBeam(ifc, container=ifc_info['floor'], name="pipe", 
                      section=cross_section, L=l, position=p,
                      direction=d, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)


# return axis aligned bbox of pipe
def pipe_bbox(r, l, d):
    l_xz = math.sqrt(d[0]*d[0]+d[2]*d[2])
    l_yz = math.sqrt(d[1]*d[1]+d[2]*d[2])
    l_xy = math.sqrt(d[0]*d[0]+d[1]*d[1])
    
    cos_y = 0 if l_xz == 0 else l_xz/(math.sqrt(l_xz*l_xz + d[1]*d[1]))
    sin_y = 0 if d[1] == 0 else d[1]/(math.sqrt(l_xz*l_xz + d[1]*d[1]))    
    cos_x = 0 if l_yz == 0 else l_yz/(math.sqrt(l_yz*l_yz + d[0]*d[0]))
    sin_x = 0 if d[0] == 0 else d[0]/(math.sqrt(l_yz*l_yz + d[0]*d[0]))    
    cos_z = 0 if l_xy == 0 else l_xy/(math.sqrt(l_xy*l_xy + d[2]*d[2]))
    sin_z = 0 if d[2] == 0 else d[2]/(math.sqrt(l_xy*l_xy + d[2]*d[2]))    

    y = r*cos_y*2 + l*sin_y
    x = r*cos_x*2 + l*sin_x
    z = r*cos_z*2 + l*sin_z
    
    return (x,y,z)
    

# generate a random synthetic pipe
def create_pipe(config,  ifc, ifc_info):
    # generate parameters
    reject = True

    while reject:
        r = random.uniform(config['radius_range'][0], config['radius_range'][1])
        l = random.uniform(config['length_range'][0], config['length_range'][1])
        if (l/r > 2):
            reject = False
    
    d = []
    for ax in config['extrusion_direction_range']:
        d.append(random.uniform(ax[0], ax[1]))
    d_np = np.array(d)
    d = (d_np/np.linalg.norm(d_np)).tolist()

    p = []
    for coord in config['coordinate_range']:
        p.append(random.uniform(coord[0], coord[1]))
        
    # normalize bbox
    bbox = pipe_bbox(r,l,d)
    bbox_l2 = math.sqrt(bbox[0]*bbox[0] + bbox[1]*bbox[1] + bbox[2]*bbox[2])
    r, l = 1000*r/bbox_l2, 1000*l/bbox_l2
    print(bbox_l2)
    #bbox2 = pipe_bbox(r,l,d)
    #print(bbox, bbox2, (bbox2[0]*bbox2[0] + bbox2[1]*bbox2[1] + bbox2[2]*bbox2[2]))

    # center the element
    centerpoint = [(p[i] + (l*d[i])/2) for i in range(3)]
    p = [p[i] - centerpoint[i] for i in range(3)]
    #print('c', p)
    
    #print(r,l,d,p)
    
    create_IfcPipe(r, l, d, p, ifc, ifc_info)
    metadata = {'radius':r, "direction":d, "length":l, "position":p}
    return metadata
