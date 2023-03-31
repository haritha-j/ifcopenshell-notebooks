# general
import math
import numpy as np
from tqdm.notebook import tqdm


from src.geometry import sq_distance, vector_mag, sq_dist_vect, norm_array
from src.visualisation import *
from src.elements import *
from src.chamfer import *


# constants
dist_threshold = 3. # distance threshold for pipe proximity
centerline_angle_threshold = math.radians(10) # angle threshold for two pipes to be considered parallel
centerline_dist_threshold = 0.001 # centerline distance threshold pipes to be to considered connected
ridiculously_large_pipe_threshold = 1.0 # remove ridiculously large pipes
elbow_edge_threshold = 0.0005


# get flange radius - radius is defined as the avg of the two dimensions that are similar to each other
# returns radius, length
def flange_radius(bbox):
    diff01 = abs(bbox[0]-bbox[1])
    diff02 = abs(bbox[0]-bbox[2])
    diff12 = abs(bbox[1]-bbox[2])
    
    if (diff01<diff02) and (diff01<diff12):
        return ((bbox[0]+bbox[1])/4, bbox[2])
    elif (diff02<diff01) and (diff02<diff12):
        return ((bbox[0]+bbox[2])/4, bbox[1])
    #elif (diff12<diff01) and (diff12<diff02):
    else:
        return ((bbox[1]+bbox[2])/4, bbox[0])


def get_radius_from_bbox(bbox):
    sides = bbox[bbox.argsort()[:2]]
    return (sum(sides)/4)


# check percentage deviation between a list of radii
def radius_check(pipe_list, thresh):
    radii = [get_radius_from_bbox(p[2]) for p in pipe_list]
    check = True
    
    for r1 in radii:
        if not check:
            break
        for r2 in radii:
            if (r1/r2 < thresh) or (r2/r1 < thresh):
                check = False
                break
    return check


# find closest pair of edges satisfying a minimum distance criteria
def edge_proximity_criteria(e1, e2, threshold):
    nearby_pair = None
    min_dist = math.inf
    d = sq_dist_vect(e1[0], e2[0])
    if (d < threshold):
        nearby_pair = [0, 0]
        min_dist = d
    d = sq_dist_vect(e1[1], e2[0])
    if (d < threshold and d < min_dist): 
        nearby_pair = [1, 0]
        min_dist = d
    d = sq_dist_vect(e1[0], e2[1])
    if (d < threshold and d < min_dist): 
        nearby_pair = [0, 1]
        min_dist = d    
    d = sq_dist_vect(e1[1], e2[1]) 
    if (d < threshold and d < min_dist): 
        nearby_pair = [1, 1]
        min_dist = d
    return nearby_pair


# find edge pairs that match with each other
def edge_match(a, b):
    e = None
    if (a[0][0] == b[0][0] and a[1][0] == b[1][0]):
        e = 1
    elif (a[0][1] == b[0][0] and a[1][1] == b[1][0]):
        e = 0
    elif (a[0][0] == b[0][1] and a[1][0] == b[1][1]):
        e = 1
    elif (a[0][1] == b[0][1] and a[1][1] == b[1][1]):
        e = 0
    
    if e is not None:
        return ((a[0][e], b[0][0], b[0][1]), (a[1][e], b[1][0], b[1][1]), (a[2], b[2]))
    else:
        return None
    
    
# get angle deviation between two pipe centerlines
def get_centerline_deviation(ad, bd):
    centerline_deviation = np.arccos( np.dot(ad, bd))
    if centerline_deviation > np.pi/2:
        centerline_deviation = np.pi - centerline_deviation
    return centerline_deviation


# get minimum distance between two pipe centerlines
def get_centerline_distance(a, b):
    centerline_connecting_line = np.cross(a[3], b[3])
    center_connecting_line = b[1] - a[1]
    centerline_distance = (abs(np.dot(centerline_connecting_line, 
                                          center_connecting_line)) / 
                           vector_mag(centerline_connecting_line))
    return centerline_distance


# get the distance along centerline of each pipe to the intersection or 
# closest point to the other line
def get_distance_to_intersection(a, b):
    centerline_connecting_line = np.cross(a[3], b[3])
    center_connecting_line = b[1] - a[1]
    centerline_distance = (abs(np.dot(centerline_connecting_line, 
                                          center_connecting_line)) / 
                           vector_mag(centerline_connecting_line))
    sq_mag_ccl = np.dot(centerline_connecting_line, 
                        centerline_connecting_line)
    t1 = np.dot(np.cross(b[3], centerline_connecting_line),
                center_connecting_line) / sq_mag_ccl
    t2 = np.dot(np.cross(a[3], centerline_connecting_line),
                center_connecting_line) / sq_mag_ccl
    
    return t1, t2


# for elbows, check that the extended centerlines do not intersect inside one pipe
def outer_intersection_check(a, b):
    t1, t2 = get_distance_to_intersection(a, b)
    if abs(t1) < max(a[2]) and abs(t2) < max(b[2]):
        return True

    
# check if two pipe segments could be continuous
def pipe_check(a, b):
    # radius check
    if radius_check([a,b], 0.8):
        
        # centerline direction check -
        centerline_deviation = get_centerline_deviation(a[3], b[3])
        if centerline_deviation < centerline_angle_threshold:

            # centerline proximity check        
            centerline_distance = get_centerline_distance(a, b)
            if centerline_distance < centerline_dist_threshold:
                return True
#                 # co-planar check
#                 if coplanar_check(a, b):
#                     return True
    return False


# check if two pipe segments could be connected by an elbow
def elbow_check(a, b, thresh = 0.4, intersection_test=True):
    # radius check
    if radius_check([a,b], thresh):
        
        # centerline direction check -
        centerline_deviation = get_centerline_deviation(a[3], b[3])
        if centerline_deviation > centerline_angle_threshold:

            # centerline proximity check        
            centerline_distance = get_centerline_distance(a, b)
            if centerline_distance < centerline_dist_threshold:
                
                # intersection check
                if intersection_test:
                    if outer_intersection_check(a, b):
                        return True
                else:
                    return True
    return False


def visualise_pipes(pipes, return_type="cloud"):
    clouds = []
    preds = []
    count = 0
    for pipe in tqdm(pipes):
        r = get_radius_from_bbox(pipe[2])
        l = max(pipe[2])
        d = pipe[3].tolist()
        p = pipe[1].tolist()
        params = [r, l] + p

        for i in range(3):
            params.append(math.sin(d[i]))
            params.append(math.cos(d[i]))
        
        preds.append(params)
        if return_type == "cloud":
            cld = np.array(generate_pipe_cloud(params, scale = True))
            count += 1
            clouds.append(cld)

    if return_type == "cloud":
        return np.concatenate(clouds)
    else:
        return preds
    
    
def visualise_tees(tee_connections, blueprint, pipes, pipe_edges, return_type="cloud"):
    # setup ifc
    ifc = setup_ifc_file(blueprint)
    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]
    scale = 1
    clouds = []
    preds = []
    refined_tees = []

    ifc_info = {"owner_history": owner_history,
        "project": project,
       "context": context, 
       "floor": floor}
    
    # calculate tee parameters
    for tee in tqdm(tee_connections):
        pipe_pair_ids = [tee[0][0][tee[1][0]], tee[0][0][tee[1][1]]]
        pipe_pair = (pipes[pipe_pair_ids[0]], pipes[pipe_pair_ids[1]])
        other_id = tee[0][0][tee[2]]
        other = pipes[other_id]
        pipe_edge_ids = [tee[0][1][tee[1][0]], tee[0][1][tee[1][1]]]
        other_edge_id = tee[0][1][tee[2]]
        
        r1 = (get_radius_from_bbox(pipe_pair[0][2]) + 
              get_radius_from_bbox(pipe_pair[1][2]))/2
        r2 = get_radius_from_bbox(other[2])
       # print('r', r1, r2)     

        l1_edges = [pipe_edges[pipe_pair_ids[0]][pipe_edge_ids[0]], 
                    pipe_edges[pipe_pair_ids[1]][pipe_edge_ids[1]]]
        l1 = np.sqrt(sq_dist_vect(l1_edges[0], l1_edges[1]))
        l2_edges = [(l1_edges[0] + l1_edges[1])/2, pipe_edges[other_id][other_edge_id]]
        l2 = np.sqrt(sq_dist_vect(l2_edges[0], l2_edges[1]))
        #print('l', l1, l2)
        
        r1, r2, l1, l2 = r1*scale, r2*scale, l1*scale, l2*scale

        p1 = (l1_edges[0]*scale).tolist()
        p2 = (l2_edges[0]*scale).tolist()
        d1 = vector_normalise(l1_edges[1] - l1_edges[0])
        
        # additional check by calculating d1 again
        d1_alternative = pipe_pair[0][3].tolist()
        d1_deviation =  get_centerline_deviation(d1, d1_alternative)
        if d1_deviation > centerline_angle_threshold:
            continue
        
        d2 = vector_normalise(l2_edges[1] - l2_edges[0])
        #print('pd', p1, p2, d1, d2)
        
        # temporary: format to predictions format to generate visualisation
        params = [r1, l1, r2, l2] + p2
        
        for d in [d1, d2]:
            for i in range(3):
                params.append(math.sin(d[i]))
                params.append(math.cos(d[i]))
        
        #print(len(params), params)
        refined_tees.append(tee)
        
        if return_type == "cloud":
            cld = np.array(generate_tee_cloud(params))
            clouds.append(cld)
        else:
            preds.append(params)
            #create_IfcTee(r1, r2, l1, l2, d1, d2, p1, p2, ifc, ifc_info)

    print(len(refined_tees))
    if return_type == "cloud":
        return np.concatenate(clouds), refined_tees
    else:
        return preds, refined_tees


def get_elbow_params_from_pipes(el, pipes, pipe_edges):
    a = pipes[el[0][0]]
    a_edge = pipe_edges[el[0][0]][el[1][0]]
    b_edge = pipe_edges[el[0][1]][el[1][1]]
    b = pipes[el[0][1]]

    r = (get_radius_from_bbox(a[2]) + 
         get_radius_from_bbox(b[2])) / 2  
    p = a_edge.tolist()

    d1 = a[3]
    d2 = b[3]

#         # flip d1, d2 if they're facing incorrect directions
    d4 = (b_edge - a_edge) # line connecting edges
    if (np.dot(d1, d4) > 0):
        d1 = -1*d1
    if (np.dot(d2, d4) > 0):
        d2 = -1*d2

    ang = np.arccos(np.dot(a[3], b[3]))
    ang_sin, ang_cos = np.sin(ang), np.cos(ang)

    old_z = (0., 0., 1.)
    if np.isclose(np.dot(d1, old_z), 1) or np.isclose(np.dot(d1, old_z), -1):
        old_z = (0., 1., 0.)

    x_axis = vector_normalise(np.cross(d1, old_z))
    y_axis = vector_normalise(np.cross(d1, x_axis))

    #rot_mat = np.transpose(np.array([x_axis, y_axis, d1]))
    rot_mat = np.array([x_axis, y_axis, d1])

#         intersection_distance, _ = get_distance_to_intersection(a, b)
#         r2 = intersection_distance / np.tan(ang/2)

#         centerline_connecting_line = np.cross(a[3], b[3])
#         vector_to_elbow_center = np.cross(a[3], centerline_connecting_line)
#         elbow_center = (pipe_edges[el[0][0]][el[1][0]] + 
#                         (r2 * vector_normalise(vector_to_elbow_center)))

    d3 = -1. *norm_array((norm_array(d2) - norm_array(d1)*ang_cos)/(1 - ang_cos))
    l_d3 = np.sqrt(sq_dist_vect(a_edge, b_edge)) / (np.sin(ang/2) * 2)

    v = d3 * l_d3
    transformed_v = np.matmul(rot_mat, v)
    x, y = transformed_v[0], transformed_v[1]

    params = [r, x, y] + p + [ang_sin, ang_cos]
    for i in range(3):
        params.append(math.sin(d1[i]))
        params.append(math.cos(d1[i]))
        
    return params, b_edge


def visualise_elbows(elbows, pipes, pipe_edges, return_type="cloud"):
    clouds = []
    preds = []
    refined_elbows = []
    for el in tqdm(elbows):
        # get elbow parameters
        params, b_edge = get_elbow_params_from_pipes(el, pipes, pipe_edges)
            
        # additional check to ensure the end of the elbow coincides with the end of the pipe
        elbow_edge, _ = generate_elbow_cloud(params, return_elbow_edge=True)
        if (sq_dist_vect(elbow_edge, b_edge) > elbow_edge_threshold):
            continue
        refined_elbows.append(el)
        
        preds.append(params)
        if return_type == "cloud":
            cld = np.array(generate_elbow_cloud(params))
            clouds.append(cld)
            
    if return_type == "cloud":
        return np.concatenate(clouds), refined_elbows
    else:
        return refined_elbows
    
    

    
