# utilities for graph dataset generation from IFC

from ifcopenshell.util.selector import Selector
from tqdm import tqdm
import pickle
import numpy as np

# graph 
import dgl
from dgl.data import DGLDataset
import torch

from src.geometry import get_dimensions, vector_normalise
from src.utils import *

from src.cloud import element_to_cloud
from src.centerline import flange_radius


#TODO: Check these parameters
def get_tee_features(preds):
    pm = {'r1':preds[0], 'l1':preds[1], 'r2':preds[2],'l2':preds[3]}
    pm['p2'] = np.array([preds[4], preds[5], preds[6]])
    pm['d1'] = np.array(get_direction_from_trig(preds, 7))      
    pm['d2'] = np.array(get_direction_from_trig(preds, 13))
    p1 = pm['p2'] - pm['d1'] * pm['l1'] * 0.5
    p2 = pm['p2'] + pm['d1'] * pm['l1'] * 0.5
    p3 = pm['p2'] + pm['d2'] * pm['l2']
    
    return {'r1':pm['r1'], 'r2':pm['r2'], 'r3':pm['r2'],
            'p1':p1, 'p2':p2, 'p3':p3,
            'd1':(-1.*pm['d1']), 'd2':pm['d1'], 'd3':pm['d2']}


def get_elbow_features(preds):
    r = preds[0]
    p1 = np.array([preds[3], preds[4], preds[5]])
    d1 = np.array(get_direction_from_trig(preds, 8))
    p2, p_extended = generate_elbow_cloud(preds, return_elbow_edge=True)
    d2 = vector_normalise(p_extended - p2)
    
    return {'r1':r, 'r2':r, 'r3':0.,
            'p1':p1, 'p2':p2, 'p3':np.array([0.,0.,0.]),
            'd1':(-1.*d1), 'd2':d2, 'd3':np. ([0.,0.,0.])}    


# merge predictions together into one dict, and include ifc element ids from metadata file
def get_features_from_params(path):
    #classes_to_merge = ['tee',]
    classes_to_merge = ['elbow', 'tee']
    node_dict = {}
    for cl in classes_to_merge:
        with open(path/('preds_finetuned_' + cl + '.pkl'), 'rb') as f:
            preds, ids, _ = pickle.load(f)

            # load metadata
            metadata_file = path/("bp_" + cl + "_metadata.json")
            meta_f = open(metadata_file, 'r')
            class_metadata = json.load(meta_f)[cl]

            for i, pred in enumerate(tqdm(preds)):
                element_id = class_metadata[str(ids[i])]['id']
                # undo normalisation on predicted parameters
                original_pred = undo_normalisation(ids[i], cl, pred, path, ".pcd", scale_up=False)
                # tees require an additional level of normalisation since the dataset was 
                # resampled to avoid issues with capped ends
                if cl== 'tee':
                    original_pred = bp_tee_correction(original_pred, class_metadata[str(ids[i])], cl)
                    params = get_tee_features(original_pred)
                if cl =='elbow':
                    params = get_elbow_features(original_pred)

                node_dict[str(element_id)] = params

    return (node_dict)


def get_pipe_and_flange_features(nodes, idx):
    r, l = flange_radius(nodes[0][idx][2])
    d = np.array(nodes[0][idx][3])
    p1 = np.array(nodes[0][idx][1]) - d/2
    p2 = np.array(nodes[0][idx][1]) + d/2
    # get d, p build node feature dict 
    
    return {'r1':r, 'r2':r, 'r3':0.,
            'p1':p1, 'p2':p2, 'p3':np.array([0.,0.,0.]),
            'd1':-1.*d, 'd2':d, 'd3':np.array([0.,0.,0.])}


# derive noad features from predicted parameters
# each element has the following parameters
# class (c)
# radii (r1, r2, (r3))
# directions (d1, d2, (d3))
# positions (p1, p2, (p3))
# the 3rd feature is only present in tees
def get_node_features(nodes, path):   
    # filter nodes by type
    types = ['FLANGE', 'ELBOW', 'TEE', 'TUBE', 'BEND']
    element_node_ids = {}
    for i, t in enumerate(types):
        element_node_ids[t] = [j for j, n in enumerate(nodes[0]) if n[0]==i]
        print(len(element_node_ids[t]))
    print(nodes[0][element_node_ids['FLANGE'][0]])
    
    # temp fix for flanges: get flange params using bboxes. pipes use the same logic
    # TODO: infer flange params using pointnet
    node_features_from_bboxes = {}
    for fl in element_node_ids['FLANGE']:
        node_features_from_bboxes[str(nodes[0][fl][4])] = get_pipe_and_flange_features(nodes, fl)
    for pi in element_node_ids['TUBE']:
        node_features_from_bboxes[str(nodes[0][pi][4])] = get_pipe_and_flange_features(nodes, pi)
        
    node_features_from_params = get_features_from_params(path)
    node_features = {**node_features_from_params, **node_features_from_bboxes}
    
    print("dict lengths", 
          len(node_features_from_params.keys()), 
          len(node_features_from_bboxes.keys()), 
          len(node_features.keys()))
    
    # sort all the features in the order of the original node list
    labels = np.array([i[0] for i in nodes[0]])
    element_ids = np.array([i[4] for i in nodes[0]])
    print(node_features_from_params.keys())
    sorted_features = {}
    missing_keys = []
    keys = ['r1', 'r2', 'r3', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3']
    for key in keys:
        sorted_features[key] =[]
            
    for i, element_id in enumerate(element_ids):
        if str(element_id) in node_features.keys():        
            nf = node_features[str(element_id)]
            for key in keys:
                sorted_features[key].append(nf[key])
        else:
            missing_keys.append((element_id, labels[i]))
    
    feature_list = [labels]
    for key in keys:
        feature_list.append(np.array(sorted_features[key]))
    
    print(len(sorted_features['r1']))
    print("missing", len(missing_keys), missing_keys[0])
    return (torch.from_numpy(np.column_stack(feature_list)))
    
    
    
# define industrial facility graph dataset
class IndustrialFacilityDataset(DGLDataset):

    def __init__(self, data_path, site="westdeckbox", element_params=False, params_path=None):
        self.site= site
        self.data_path = data_path
        self.element_params = element_params
        self.params_path = params_path
        super().__init__(name='industrial_facility')


    def process(self):
        # data loading
        #data_path = "/content/drive/MyDrive/graph/"
        #data_path = '../'
        edge_file = "edges_" + self.site + ".pkl"
        node_file = "nodes_" + self.site + ".pkl"
        with open(self.data_path + node_file, 'rb') as f:
            node_info = pickle.load(f)
        with open(self.data_path + edge_file, 'rb') as f:
            edges = pickle.load(f)
            
        # node features
        if self.element_params:
            # derive noad features from predicted parameters
            features = get_node_features(node_info, self.params_path)
            print(features.shape)
        
        else:
            # derive node features using bboxes
            labels = np.array([i[0] for i in node_info[0]])
            centers = np.array([i[1] for i in node_info[0]])
            lengths = np.array([i[2] for i in node_info[0]])
            directions = np.array([i[3] for i in node_info[0]])
            features = torch.from_numpy(np.column_stack((labels, centers, lengths, directions)))
        
        # points = np.array(node_info[1])
        # points = points.reshape((points.shape[0], points.shape[1]*points.shape[2]))
        # features = torch.from_numpy(points)
        # print(np.array([i[0] for i in node_info[0]]).shape, np.array([i[1] for i in node_info[0]]).shape, 
        #       np.array([i[2] for i in node_info[0]]).shape, np.array([i[3] for i in node_info[0]]).shape)



        #b = np.random.randn(*labels.shape)
        #features = torch.from_numpy(np.column_stack((labels, b)))
        print(features.shape)
        
        # edges
        edges = np.array(edges)
        print(len(edges))
        edges_src = torch.from_numpy(edges[:,0])
        edges_dst = torch.from_numpy(edges[:,1])

        # create graph
        self.graph = dgl.to_bidirected(dgl.graph((edges_src, edges_dst), num_nodes = len(node_info[0])))
        self.graph.ndata['feat'] = features
        # self.graph.ndata['centers'] = centers
        # self.graph.ndata['directions'] = directions
        # self.graph.ndata['lengths'] = lengths
        # self.graph.ndata['label'] = labels
        #self.graph.ndata['points'] = points
        #self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
#         n_nodes = nodes_data.shape[0]
#         n_train = int(n_nodes * 0.6)
#         n_val = int(n_nodes * 0.2)
#         train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         train_mask[:n_train] = True
#         val_mask[n_train:n_train + n_val] = True
#         test_mask[n_train + n_val:] = True
#         self.graph.ndata['train_mask'] = train_mask
#         self.graph.ndata['val_mask'] = val_mask
#         self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



# get node info
def process_nodes(ifc, types):
    # load elements
    element_type1 = 'IFCPIPESEGMENT'
    element_type2 = 'IFCPIPEFITTING'
    selector = Selector()
    segments = selector.parse(ifc, '.' + element_type1)
    fittings = selector.parse(ifc, '.' + element_type2)
    elements = segments + fittings
    
    # generate node features
    nodes = []
    points = []
    error_count = 0
    for i, el in enumerate(tqdm(elements)):
        try:
            center, lengths, dominant_direction = get_dimensions(el)
            
            # find element type
            found = False
            for j, t in enumerate(types):
                if t in el.Name:
                    el_type = j
                    found = True
                    break
            if not found:
                el_type = len(types)

            element_points = element_to_cloud(el, None, 1000)
            node = [el_type, center, lengths, dominant_direction, 
                          el.id()]
            
            nodes.append(node)
            points.append(element_points)
            
        except Exception as e:
            error_count += 1
            print(el.Name)
            print(e)
    
    print(error_count)
    return([nodes, points])


# derive edges from node and relationship information
def process_edges(ifc, nodes, rels):
    selector = Selector()
    pipe_type = 'IFCPIPESEGMENT'
    fitting_type = 'IFCPIPEFITTING'

    pipe_selector = Selector()
    fitting_selector = Selector()
    pipes = pipe_selector.parse(ifc, '.' + pipe_type)
    fittings = fitting_selector.parse(ifc, '.' + fitting_type)
    elements = pipes + fittings
    element_names = [e.Name for e in elements]
    print(len(element_names))
    edges = []
    error_count = 0
    
    # lookup matching index from nodes
    for rel in tqdm(rels):
        element1 = element_names.index(rel[0][0])
        element1_id = elements[element1].id() 
        element2= element_names.index(rel[1][0])
        element2_id = elements[element2].id()
        
        element1_found, element2_found = False, False
        for i, node in enumerate(nodes):
            if element1_id == node[4]:
                element1_index = i
                element1_found = True
            if element2_id == node[4]:
                element2_index = i
                element2_found = True
            if (element1_found and element2_found):
                edges.append([element1_index, element2_index])
                break
        if not (element1_found and element2_found):
            #print(rel[0][0], rel[1][0])
            error_count += 1
                        
    print(error_count, len(edges))
    return (edges)
