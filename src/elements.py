import os
import numpy as np
import math
import random
import ifcopenshell
import ifcopenshell.geom

from src.ifc import *


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


# generate Ifc Pipe fitting from parameters
def create_IfcElbow(r, a, d, p, x, y, axis_dir, ifc, ifc_info, fill=False, z=(0., 0., 1.)):
    cross_section = Circle_Section(r=r, ifcfile=ifc, fill=fill)

    beam = CreateElbow(ifc, container=ifc_info['floor'], name="elbow", 
                      section=cross_section, a=a, position=p,
                      direction=d, x=x, y=y, axis_dir=axis_dir, 
                       owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], Z=z, colour=None)


# return axis aligned bbox, centerpoint of elbow
def tee_bbox(r1, r2, l1, l2, d1, d2):

    b1 = bounding_box_dimensions(bounding_box_cylinder(r1,l1,d1))
    b2 = bounding_box_dimensions(bounding_box_cylinder(r2,l2,d2))
    #print('b1', b1, 'b2', b2)
    #print('b1', [b1[1][i]-b1[0][i] for i in range(3)])
    #print('b1x', [b1x[1][i]-b1x[0][i] for i in range(3)])
    #translate bbox 2 to its p2
    b2_adjusted = []
    for i in range(2):
        b2_adjusted.append((b2[i] + np.array(d1) * 0.5 * l1).tolist())
    #print('b2 adj', [b2_adjusted[1][i]-b2_adjusted[0][i] for i in range(3)])

    min_point = [min(b1[0][0], b2_adjusted[0][0]), min(b1[0][1], b2_adjusted[0][1]), min(b1[0][2], b2_adjusted[0][2])]
    max_point = [max(b1[1][0], b2_adjusted[1][0]), max(b1[1][1], b2_adjusted[1][1]), max(b1[1][2], b2_adjusted[1][2])]

    centerpoint = [(max_point[0] + min_point[0])/2000, (max_point[1] + min_point[1])/2000, (max_point[2] + min_point[2])/2000]
    bbox = [(max_point[0] - min_point[0])/1000, (max_point[1] - min_point[1])/1000, (max_point[2] - min_point[2])/1000]

    return bbox, centerpoint


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
    #print('INIT PARAMS', r, a, d, p, x, y)
    create_IfcElbow(r, a, d, p, x, y, axis_dir, temp, temp_info, fill=True)
    
    bbox, center = generic_element_bbox(temp, "IfcPipeFitting")
    del temp
    return bbox, center


# generate a random synthetic elbow
def create_elbow(config,  ifc, ifc_info, blueprint, i):
    # generate parameters
    r = random.uniform(config['radius_range'][0], config['radius_range'][1])
    # r = 10.0
    # a = 30.0+10*i 
    a = random.uniform(config['angle_range'][0], config['angle_range'][1])
    # d = (-1., 0., 0.)
    d = []
    for ax in config['axis_direction_range']:
        d.append(random.uniform(ax[0], ax[1]))
    d_np = np.array(d)
    d = (d_np/np.linalg.norm(d_np)).tolist()

    p = []
    for coord in config['coordinate_range']:
        p.append(0.0)
        
    # generate points on a 2D ring from the origin
    axis_placement = random.uniform(config['curvature_range'][0], 
                                    config['curvature_range'][1])*r
    # axis_placement = 20.0 * r
    #axis_ang = math.radians(0)
    # axis_ang = math.radians(0+ i*30)
    #axis_placement = 50*r
    
    axis_ang = random.uniform(config['axis_angle_range'][0], 
                             config['axis_angle_range'][1])
    x = axis_placement * math.sin(axis_ang)
    y = axis_placement * math.cos(axis_ang)
    axis_dir = (math.cos(axis_ang), -1*math.sin(axis_ang))

    # transform points to the elbow center and normalize
    bbox, centerpoint = elbow_bbox(r, a, d, p, x, y, axis_dir, blueprint)
    bbox_l2 = math.sqrt(bbox[0]*bbox[0] + bbox[1]*bbox[1] + bbox[2]*bbox[2])
    #print('p', p, 'c', centerpoint, 'bbx', bbox_l2, bbox)
    #p = [p[i] - centerpoint[i]*10000/bbox_l2 for i in range(3)]
    #p = [-1* centerpoint[i]*1000  for i in range(3)]
    p = [-1* centerpoint[2]*1000/bbox_l2, 1* centerpoint[1]*1000/bbox_l2, 1* centerpoint[0]*1000/bbox_l2]
    print("SD P", p)

    # old_z = (0., 0., 1.)
    # x_axis = np.cross(d, old_z).tolist()
    # y_axis = np.cross(d, x_axis).tolist()

    y_axis = (0., 0., 1.)
    x_axis = np.cross(d, y_axis).tolist()

    p = [(-1*(centerpoint[0]*1000/bbox_l2 * x_axis[i]) + (centerpoint[1]*1000/bbox_l2 * y_axis[i]) + 
          -1*(centerpoint[2]*1000/bbox_l2 * d[i])) for i in range(3)]
    #p = [1* centerpoint[2]*1000/bbox_l2, -1* centerpoint[0]*1000/bbox_l2, 1* centerpoint[1]*1000/bbox_l2]
    #p = [-1* centerpoint[2]*1000/bbox_l2, 0., 0.]

    r, x, y = r/bbox_l2, x/bbox_l2, y/bbox_l2
    print('p', p, 'c', centerpoint, r, x)

    # print('bb befpre', bbox, 'c', centerpoint, bbox_l2)
    # bbox2, centerpoint2 = elbow_bbox(r, a, d, p, x, y, axis_dir,blueprint)
    # print('bb after', bbox2, 'c', centerpoint2)

    create_IfcElbow(r, a, d, p, x, y, axis_dir, ifc, ifc_info)

    #draw_bbox(bbox2, centerpoint2, ifc, ifc_info['floor'], owner_history=ifc_info["owner_history"],
    #                  context=ifc_info["context"])

    metadata = {'radius':r, "direction":d, "angle":a, "position":p, 
                'axis_x':x, 'axis_y':y}
    #print(metadata, axis_ang)
    
    return metadata
    
# generate Ifc tee from parameters
def create_IfcTee(r1, r2, l1, l2, d1, d2, p1, p2, ifc, ifc_info, bp=True):
    cross_section1_filled = Circle_Section(r=r1, ifcfile=ifc, fill=True)
    cross_section1 = Circle_Section(r=r1, ifcfile=ifc, fill=bp)
    cross_section2_filled = Circle_Section(r=r2, ifcfile=ifc, fill=True)
    cross_section2 = Circle_Section(r=r2, ifcfile=ifc, fill=bp)

    beam1_full = CreateBeamGeom(ifc, section=cross_section1, L=l1, position=p1,
                      direction=d1)
    beam1_filled = CreateBeamGeom(ifc, section=cross_section1_filled, L=l1, position=p1,
                      direction=d1)

    beam2_full =  CreateBeamGeom(ifc, section=cross_section2, L=l2, position=p2,
                      direction=d2)
    
    beam2_filled =  CreateBeamGeom(ifc, section=cross_section2_filled, L=l2, position=p2,
                      direction=d2)

    beam1 = CreatePartialBeam(ifc, container=ifc_info['floor'], name="main", 
                      primary_beam=beam2_filled, secondary_beam=beam1_full, 
                      owner_history=ifc_info["owner_history"], context=ifc_info["context"])
    
    beam2 = CreatePartialBeam(ifc, container=ifc_info['floor'], name="secondary", 
                      primary_beam=beam1_filled, secondary_beam=beam2_full, 
                      owner_history=ifc_info["owner_history"], context=ifc_info["context"])
    


# generate Ifc tee without element substraction, only for bbox
def create_IfcTeeGeom(r1, r2, l1, l2, d1, d2, p1, p2, ifc, ifc_info):
    cross_section1 = Circle_Section(r=r1, ifcfile=ifc)
    cross_section2 = Circle_Section(r=r2, ifcfile=ifc)

    beam1 = CreateBeam(ifc, container=ifc_info['floor'], name="main", 
                      section=cross_section1, L=l1, position=p1,
                      direction=d1, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)
    
    beam2 = CreateBeam(ifc, container=ifc_info['floor'], name="secondary", 
                      section=cross_section2, L=l2, position=p2,
                      direction=d2, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)


# generate IfcBeam from parameters
def create_IfcPipe(r, l, d, p, ifc, ifc_info):
    cross_section = Circle_Section(r=r, ifcfile=ifc)

    beam = CreateBeam(ifc, container=ifc_info['floor'], name="pipe", 
                      section=cross_section, L=l, position=p,
                      direction=d, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)


# generate IfcBeam from parameters
def create_IfcFlange(r1, r2, l1, l2, d, p1, p2, ifc, ifc_info, fill=True):
    cross_section1 = Circle_Section(r=r1, ifcfile=ifc, fill=fill)
    cross_section2 = Circle_Section(r=r2, ifcfile=ifc, fill=fill)
    
    beam1 = CreateBeam(ifc, container=ifc_info['floor'], name="f1", 
                      section=cross_section1, L=l1, position=p1,
                      direction=d, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)

    beam2 = CreateBeam(ifc, container=ifc_info['floor'], name="f2", 
                      section=cross_section2, L=l2, position=p2,
                      direction=d, owner_history=ifc_info["owner_history"],
                      context=ifc_info["context"], colour=None)


def bounding_box_cylinder(radius, length, direction):
    # Calculate the half-width and half-height of the bounding box
    half_width = radius
    half_height = radius

    # Calculate the centerpoint of the bounding box
    center_x = 0
    center_y = 0
    center_z = length / 2

    # Calculate the vertices of the bounding box
    vertices = [
        (center_x - half_width, center_y - half_height, center_z - length/2),
        (center_x + half_width, center_y - half_height, center_z - length/2),
        (center_x + half_width, center_y + half_height, center_z - length/2),
        (center_x - half_width, center_y + half_height, center_z - length/2),
        (center_x - half_width, center_y - half_height, center_z + length/2),
        (center_x + half_width, center_y - half_height, center_z + length/2),
        (center_x + half_width, center_y + half_height, center_z + length/2),
        (center_x - half_width, center_y + half_height, center_z + length/2)
    ]

    # Rotate the vertices if the cylinder axis is not aligned with the z-axis
    if direction != (0, 0, 1):

        iA = np.array([1., 0., 0.])
        jA = np.array([0., 1., 0.])
        kA = np.array([0., 0., 1.])

        kB =np.array(direction)
        iB = np.cross(kB, kA)
        iB = iB/np.linalg.norm(iB)
        jB = np.cross(kB, iB)
        jB = jB/np.linalg.norm(jB)

        rotation_matrix = np.array([[np.dot(iA, iB), np.dot(iA, jB), np.dot(iA, kB)], 
        [np.dot(jA, iB), np.dot(jA, jB), np.dot(jA, kB)],
        [np.dot(kA, iB), np.dot(kA, jB), np.dot(kA, kB)]])
        #print(rotation_matrix)

        # Rotate the vertices
        rotated_vertices = []
        for vertex in vertices:
            rotated_vertex = rotation_matrix @ vertex
            rotated_vertices.append(rotated_vertex.tolist())
        vertices = rotated_vertices
    return vertices

def bounding_box_dimensions(vertices):
    # Find the minimum and maximum x, y, and z coordinates of the vertices
    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)
    min_z = min(vertex[2] for vertex in vertices)
    max_z = max(vertex[2] for vertex in vertices)

    return ([min_x, min_y, min_z], [max_x, max_y, max_z])


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

    y = abs(r*cos_y*2 + l*sin_y)
    x = abs(r*cos_x*2 + l*sin_x)
    z = abs(r*cos_z*2 + l*sin_z)
    
    return (x,y,z)


# generate random synthetic tee
def create_tee(config,  ifc, ifc_info, blueprint):
    
    # generate parameters
    reject = True
    while reject:
        r1 = random.uniform(config['radius1_range'][0], config['radius1_range'][1])
        l1 = random.uniform(config['length1_range'][0], config['length1_range'][1])
        if (l1/r1 > 2) and (l1/r1 < 10):
            reject = False

    if (random.uniform(0,1) < config['same_radius_prob']):
        r2 = r1
    else:
        r2 = r1 * random.uniform(config['radius2_percentage_range'][0],
                                  config['radius2_percentage_range'][1])
    
    l2 = l1 * random.uniform(config['length2_percentage_range'][0], 
                             config['length2_percentage_range'][1]) + r1
    
    d1 = []
    for ax in config['extrusion_direction_range']:
        d1.append(random.uniform(ax[0], ax[1]))
    d1_np = np.array(d1)
    d1 = (d1_np/np.linalg.norm(d1_np))

    #d1 = np.array([0.707,0.,0.707])
    #d1 = np.array([0.,0.04,0.999])
    p1 = np.array([0., 0., 0.])

    # calculate position and direction of secondary pipe
    #p2 = [(p[i] + x * d1[i]) for i in range(3)]
    p2 = p1 +  d1 * 0.5 * l1
    #print("p2", p2, 'r1', r1, 'r2', r2, 'l1', l1)

    # generate random direction at any angle only if the secondary tube is short enough
    if (random.uniform(0,1) > config['tee_right_angle_prob']) and ((l2*math.cos(config['tee_placement_angle_range'][0]) 
    + r2*math.sin(config['tee_placement_angle_range'][0]))*1.2 < l1/2):
        #print("not right")
        # generate random direction which is not necessarily perpendicular to extrusion axis
        reject = True
        while reject:
            d2 = []
            for ax in config['extrusion_direction_range']:
                d2.append(random.uniform(ax[0], ax[1]))
            d2_np = np.array(d2)
            d2 = (d2_np/np.linalg.norm(d2_np))

            # tee angle
            cos_tee_angle = np.dot(d1,d2)/np.linalg.norm(d1)/np.linalg.norm(d2)
            if ((cos_tee_angle < math.cos(config["tee_angle_range"][0])) and 
                (cos_tee_angle > math.cos(config["tee_angle_range"][1]))):
                reject = False

    else:
        # generate random direction perpendicular to extrusion axis
        cos_tee_angle = 0.
        tee_placement_angle = random.uniform(config['tee_placement_angle_range'][0], 
                                             config['tee_placement_angle_range'][1])
        random_axis = (math.cos(tee_placement_angle), math.sin(tee_placement_angle), 0.)
        #random_axis = np.array((1.,0.,0.))

        d2 = np.cross(d1, random_axis)
        d2 = d2/np.linalg.norm(d2)
        #print("random axis", random_axis)
        #print("right")

    # tee placement angle
    z_old = (0., 0., 1.)
    x_axis = np.cross(d1, z_old)
    x_axis = (x_axis/np.linalg.norm(x_axis)).tolist()
    y_axis = np.cross(d2, x_axis)
    y_axis = (y_axis/np.linalg.norm(y_axis)).tolist()

    #cos_tee_placement_angle = np.dot(x_axis,d2)/np.linalg.norm(x_axis)/np.linalg.norm(d2)
    d1, d2, p1, p2 = d1.tolist(), d2.tolist(),  p1.tolist(),  p2.tolist()
    #print("d2", d1, d2, cos_tee_angle)

    #print("pipe method", pipe_bbox(r1,l1,d1), pipe_bbox(r2,l2,d2))
    # b1_dims = pipe_bbox(r1,l1,d1)
    # c1 = [((l1*d1[i])/2) for i in range(3)]
    # b1 = [[(c1[i] - b1_dims[i]/2) for i in range(3)], [(c1[i] + b1_dims[i]/2) for i in range(3)]]
    # b2_dims = pipe_bbox(r2,l2,d2)
    # c2 = [((l2*d2[i])/2) for i in range(3)]
    # b2 = [[(c2[i] - b2_dims[i]/2) for i in range(3)], [(c2[i] + b2_dims[i]/2) for i in range(3)]]

    bbox, centerpoint = tee_bbox(r1, r2, l1, l2, d1, d2)
    bbox_l2 = math.sqrt(bbox[0]*bbox[0] + bbox[1]*bbox[1] + bbox[2]*bbox[2])
    #print("bb, center", bbox, centerpoint, bbox_l2)

    #p1 = [-1* centerpoint[2]*1000/bbox_l2, 1* centerpoint[1]*1000/bbox_l2, 1* centerpoint[0]*1000/bbox_l2]
    # print("SD P", p)

    # x_axis = np.cross(d1, y_axis).tolist()

    # z_old = (0., 0., 1.)
    # x_axis = np.cross(d1, z_old)
    # x_axis = (x_axis/np.linalg.norm(x_axis)).tolist()
    # y_axis = np.cross(d2, x_axis)
    # y_axis = (y_axis/np.linalg.norm(y_axis)).tolist()
    
    #create_IfcTee(r1, r2, l1, l2, d1, d2, p1, p2, ifc, ifc_info)

    # p1 = [(-1*(centerpoint[0]*1000/bbox_l2 * x_axis[i]) + (centerpoint[1]*1000/bbox_l2 * y_axis[i]) + 
    #       -1*(centerpoint[2]*1000/bbox_l2 * d1[i])) for i in range(3)]

    p1 = [- centerpoint[i]*1000/bbox_l2 for i in range(3)]
    r1, r2, l1, l2 = r1/bbox_l2, r2/bbox_l2, l1/bbox_l2, l2/bbox_l2
    p2 = (np.array(p1) +  np.array(d1) * 0.5 * l1).tolist()

    create_IfcTee(r1, r2, l1, l2, d1, d2, p1, p2, ifc, ifc_info)

    #TODO: Fix: secondary length too long, only the invisible end is aligned to the center in angled tees
    #TODO: Fix: some tees have missing secondary tubes (usually on zeroth element?), when direction is aligned with axis, check placement angle
          
    metadata = {'radius1':r1, 'radius2':r2, "direction1":d1, "direction2":d2, "length1":l1, 
                "length2":l2, "position1":p1, "position2":p2}
    return metadata


# generate a random synthetic pipe
def create_flange(config, ifc, ifc_info, fill=True):
    # generate parameters

    r2 = random.uniform(config['radius_range'][0], config['radius_range'][1])
    l1 = r2 * random.uniform(config['length_percentage_range'][0], config['length_percentage_range'][1])
    l2 = r2 * random.uniform(config['length_percentage_range'][0], config['length_percentage_range'][1])
    r1 = r2 * random.uniform(config['radius2_percentage_range'][0], config['radius2_percentage_range'][1])
    
    d = []
    for ax in config['extrusion_direction_range']:
        d.append(random.uniform(ax[0], ax[1]))
    d_np = np.array(d)
    d = (d_np/np.linalg.norm(d_np)).tolist()

    p = []
    for coord in config['coordinate_range']:
        p.append(random.uniform(coord[0], coord[1]))
        
    # normalize bbox
    bbox = pipe_bbox(r2, (l1+l2)*2, d)
    bbox_l2 = math.sqrt(bbox[0]*bbox[0] + bbox[1]*bbox[1] + bbox[2]*bbox[2])
    r2, r1, l1, l2 = 1000*r2/bbox_l2, 1000*r1/bbox_l2, 1000*l1/bbox_l2, 1000*l2/bbox_l2

    # center the element
    centerpoint = [(p[i] + ((l1+l2)*d[i])/2) for i in range(3)]
    p = [p[i] - centerpoint[i] for i in range(3)]
    p2 = [(p[i] + l1*d[i]) for i in range(3)]
    #print('c', p)
    
    create_IfcFlange(r1, r2, l1, l2, d, p, p2, ifc, ifc_info, fill)
    metadata = {'radius1':r1, 'radius2':r2, "direction":d, "length1":l1, "length2":l2, "position":p2}
    return metadata


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
