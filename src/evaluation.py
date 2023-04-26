# general
import math
import pickle

import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm
from scipy.spatial import distance

from src.ifc import draw_relationship, setup_ifc_file
from src.graph import process_nodes, process_edges, IndustrialFacilityDataset, get_node_features, get_edges_from_node_info
from src.geometry import sq_dist_vect, vector_mag
from src.centerline import get_centerline_deviation



def check_predictions_fast(preds, point_info, node_info, dist_thresh=0.0002, rough_dist_thresh=0.2):
    refined_preds = []
    discarded_preds = []
    rough_count = 0
    
    for pair in tqdm(preds):
        #rough distance check using two edges
        bb0 = get_edges_from_node_info(node_info[pair[0]])
        bb1 = get_edges_from_node_info(node_info[pair[1]])
        if ((sq_dist_vect(bb0[0], bb1[0]) < rough_dist_thresh) or
            (sq_dist_vect(bb0[0], bb1[1]) < rough_dist_thresh) or
            (sq_dist_vect(bb0[1], bb1[0]) < rough_dist_thresh) or
            (sq_dist_vect(bb0[1], bb1[1]) < rough_dist_thresh)):
            
            # slower, precise check using points
            rough_count +=1
            dist = np.min(distance.cdist(
                point_info[pair[0]], point_info[pair[1]], 'sqeuclidean'))
            if (dist < dist_thresh):
                refined_preds.append(pair)
            else:
                discarded_preds.append(pair)
                
        else:
            discarded_preds.append(pair)
            
    print(len(refined_preds), len(discarded_preds), rough_count)
    return refined_preds, discarded_preds


# check if positive predictions fall within distance threshold
def check_predictions(preds, point_info, dist_thresh=0.002):
    refined_preds = []
    
    for pair in tqdm(preds):
        #print(len(point_info), pair[0], pair[1])        
        dist = np.min(distance.cdist(
            point_info[pair[0]], point_info[pair[1]], 'sqeuclidean'))
        if (dist < dist_thresh):
            refined_preds.append(pair)
    return refined_preds


def get_centerline_deviation(ad, bd):
    centerline_deviation = np.arccos( np.dot(ad, bd))
    #print(math.degrees(centerline_deviation), ad, bd)
    if centerline_deviation > np.pi/2:
        centerline_deviation = np.pi - centerline_deviation
    return centerline_deviation


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


def get_rdp(feat, e):
    r = feat[1+e]
    p = [feat[4+e*3], feat[5+e*3], feat[6+e*3]]
    d = [feat[13+e*3], feat[14+e*3], feat[15+e*3]]
    return r, d, p


# TODO: scale thresholds with radius
def connectivity_refinements(preds, node_info, params_path, dataset, radius_threshold, 
                             centerline_angle_threshold, centerline_dist_threshold):
    # load predicted node params
    features = get_node_features(node_info, params_path, dataset, [])
    features = features.cpu().detach().numpy()
    #rint(features.shape, max(np.array(preds)[:,0]), max(np.array(preds)[:,1]))
    refined_preds = []
    discarded_preds = []
    r_count, dev_count, dist_count = 0, 0, 0
    
    for pair in tqdm(preds):
        accepted = False
        feat1 = features[pair[0]]
        feat2 = features[pair[1]]
        
        # iterate through edges of first element
        for e1 in range(3):
            if accepted:
                break
            r1, d1, p1 = get_rdp(feat1, e1)
            if r1 == 0.:
                continue
            
            # iterate through edges of second element
            for e2 in range(3):
                if accepted:
                    break
                r2, d2, p2= get_rdp(feat2, e2)
                if r2 == 0.:
                    continue

                # radius check
                #print(r1, r2, r1/r2 ,r2/r1 )
                if (r1/r2 > radius_threshold) and (r2/r1 > radius_threshold):
                    r_count += 1

                    # centerline direction check
                    centerline_deviation = get_centerline_deviation(d1, d2)
                    #print(centerline_deviation, centerline_angle_threshold)
                    if centerline_deviation < centerline_angle_threshold:
                        dev_count +=1

                        #print ("dists", math.sqrt(sq_dist_vect(p1, p2)), (r1+r2))
                        if math.sqrt(sq_dist_vect(p1, p2)) < (r1+r2):
                            
                        # centerline proximity check
#                         tn1 = [0, np.array(p1), 0, np.array(d1)] # temp node feature
#                         tn2 = [0, np.array(p2), 0, np.array(d2)] # temp node feature
#                         centerline_distance = get_centerline_distance(tn1, tn2)
#                         if centerline_distance < centerline_dist_threshold:
                            
                            # centerline co-planar check
                            #print("dist", get_distance_to_intersection(tn1, tn2))
                            dist_count += 1
                            refined_preds.append(pair)
                            accepted = True
        if not accepted:
            discarded_preds.append(pair)
    
    print(r_count, dev_count, dist_count)
    return refined_preds, discarded_preds
       
        

# as the graph is bidirected, a single edge has two predictions. 
# This function removes repetitions in a single set of predictions.
def remove_repetitions(preds):
    non_rep = []
    for i, pair in enumerate(tqdm(preds)):
        found = False
        for j, pair2 in enumerate(preds[i:]):
            if pair[0] == pair2[1] and pair[1] == pair2[0]:
                found = True
                #break
        if not found:
            non_rep.append(pair)
    
    return non_rep


# same as above, except for removing repetitions across two sets of predictions
# ex. between true positives and false negatives
def compare_preds(preds1, preds2):
    for i in range(len(preds1)):
        list(preds1[i]).sort()
    for i in range(len(preds2)):
        list(preds2[i]).sort()
    non_rep = []
    
    for i, pair in enumerate(preds1):
        found = False
        for j, pair2 in enumerate(preds2):
            if pair[0] == pair2[0] and pair[1] == pair2[1]:
                found = True
                break
        if not found:
            non_rep.append(pair)
    
    return non_rep


# element type wise precision recall analysis
def sort_type(preds, nodes):
    bins = np.zeros([4,4])
    for p in preds:
        x=nodes[p[0]][0]
        y= nodes[p[1]][0]
        if x == 4:
          x = 3
        if y == 4:
          y = 3
        li = [x,y]
        li.sort()
        x,y = li[0],li[1]
        
        bins[x][y] += 1
    return bins


def analyse_dataset(site):
    data_path = "/content/drive/MyDrive/graph/"
    edge_file = "edges_" + site + "deckbox.pkl"
    node_file = "nodes_" + site + "deckbox.pkl"
    with open(data_path + node_file, 'rb') as f:
        node_info = pickle.load(f)
    with open(data_path + edge_file, 'rb') as f:
        edges = pickle.load(f)

    # get element type counts
    labels = np.array([i[0] for i in node_info[0]])
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # get connection counts
    counts = np.zeros((5,5), dtype=int)

    for edge in edges:
        x = labels[edge[0]]
        y = labels[edge[1]]
        if x == 4:
            x = 3
        if y == 4:
            y = 3
        li = [x,y]
        li.sort()
        x,y = li[0],li[1]
        counts[x][y] += 1

    return(counts)

