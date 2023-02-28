# geomtric helper operations for ifc visualisation and processing

import math
import numpy as np
import open3d as o3d

from compas.geometry import oriented_bounding_box_numpy
from scipy.spatial import distance


def get_point_along_axis(init_point, axis, half_length, edge_distance):
    #print(half_length)
    return (init_point + axis*(half_length - edge_distance))


def sq_distance(x1, y1, z1, x2, y2, z2):
    return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def sq_dist_vect(v1, v2):
    return ((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2)


def vector_mag(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


def vector_normalise(vec):
    den = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return [vec[0]/den, vec[1]/den, vec[2]/den]


def norm_array(v):
    return np.array(vector_normalise(v))


# identify a candidate for an edge of the relationship visualization
def get_corner(obb, center_other, edge_distance):
    direction = obb[0]
    center = obb[3]

    candidate1 = [get_point_along_axis(center[i], direction[i], 
                                    obb[1], edge_distance) 
               for i in range(3)]
    candidate2 = [get_point_along_axis(center[i], direction[i], 
                                    -1*obb[1], -1*edge_distance) 
               for i in range(3)]
    # print(corner1, corner2)
    
    dist1 = sq_distance(center_other[0], center_other[1],
                   center_other[2], candidate1[0],
                   candidate1[1], candidate1[2])    
    dist2 = sq_distance(center_other[0], center_other[1],
                   center_other[2], candidate2[0],
                   candidate2[1], candidate2[2]) 
    
    if dist1 < dist2:
        return candidate1
    else:
        return candidate2

    
# get points belonging to an ifc element
def get_points(element, ifc):
    shape = element.Representation.Representations[0].Items[0]
    return (np.array(shape.Coordinates.CoordList))   
    
    
# calculate min distance between two ifc elements
def element_distance(element1, element2, ifc):
    points1 = get_points(element1, ifc)
    points2 = get_points(element2, ifc)
    return np.min(distance.cdist(points1, points2, 'sqeuclidean'))


# get bounding box of ifc element
def get_oriented_bbox(element):
    shape = element.Representation.Representations[0].Items[0]
    element_coords = np.array(shape.Coordinates.CoordList)
    #print(element_coords)
    bbox = oriented_bounding_box_numpy(element_coords)
    
    # identify box orientation
    l1 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[1][0], bbox[1][1], bbox[1][2]))
    l2 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[3][0], bbox[3][1], bbox[3][2]))
    l3 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[4][0], bbox[4][1], bbox[4][2]))
    half_lengths = [l1/2, l2/2, l3/2]
    
    dominant_axis = half_lengths.index(max(half_lengths))
    if dominant_axis == 0:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[1][i] 
                                          for i in range(3)])
    elif dominant_axis == 1:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[3][i] 
                                          for i in range(3)])
    else:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[4][i] 
                                          for i in range(3)])

    dominance_ratio = max(half_lengths)/sorted(half_lengths)[-2]
    center = [(bbox[0][i] + bbox[6][i])/2 for i in range(3)]

    #print(dominance_ratio, dominant_direction)
    #print(element_name, half_lengths, dominant_direction, center)

    #print(center)
    return([dominant_direction, max(half_lengths), 
           half_lengths, center])


# get center, dimensions and direction element points (used for graph dataset)
def get_dimensions_points (element_coords):
    bbox = oriented_bounding_box_numpy(element_coords)
    center = [(bbox[0][i] + bbox[6][i])/2 for i in range(3)]
    
    # identify box orientation
    l1 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[1][0], bbox[1][1], bbox[1][2]))
    l2 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[3][0], bbox[3][1], bbox[3][2]))
    l3 = math.sqrt(sq_distance(bbox[0][0], bbox[0][1], bbox[0][2],
                               bbox[4][0], bbox[4][1], bbox[4][2]))
    lengths = [l1, l2, l3]
    
    dominant_axis = lengths.index(max(lengths))
    if dominant_axis == 0:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[1][i] 
                                          for i in range(3)])
    elif dominant_axis == 1:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[3][i] 
                                          for i in range(3)])
    else:
        dominant_direction = vector_normalise([bbox[0][i] - bbox[4][i] 
                                          for i in range(3)])
        
    return(center, lengths, dominant_direction)


# get center, dimensions and direction of ifc element (used for graph dataset)
def get_dimensions (element):
    shape = element.Representation.Representations[0].Items[0]
    element_coords = np.array(shape.Coordinates.CoordList)
    
    return get_dimensions_points(element_coords)


# convert rotational matrix to euler angles
def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


# get bounding box in labelcloud format
def get_labelcloud_bbox(points, label):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    oriented_bounding_box = pcl.get_oriented_bounding_box()

    r = [math.degrees(x) for x in rot2eul(oriented_bounding_box.R)]
    c = oriented_bounding_box.center
    e = oriented_bounding_box.extent

    bbox = {
        'name': label,
        'centroid': {
            'x': c[0],
            'y': c[1],
            'z': c[2]
        },
        'dimensions': {
            'length': e[0],
            'width': e[1],
            'height': e[2]
        },
        'rotations': {
            'x': r[0],
            'y': r[1],
            'z': r[2]
        }
    }
    return bbox