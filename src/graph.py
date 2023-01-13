# utilities for graph dataset generation from IFC

from ifcopenshell.util.selector import Selector
from tqdm import tqdm
import pickle
import numpy as np

# graph 
import dgl
from dgl.data import DGLDataset
import torch

from src.geometry import get_dimensions
from src.cloud import element_to_cloud


# define industrial facility graph dataset
class IndustrialFacilityDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='industrial_facility')

    def process(self):
        # data loading
        site = 'cloi'
        #data_path = "/content/drive/MyDrive/graph/"
        data_path = '../'
        edge_file = "edges_" + site + ".pkl"
        node_file = "nodes_" + site + ".pkl"
        with open(data_path + node_file, 'rb') as f:
            node_info = pickle.load(f)
        with open(data_path + edge_file, 'rb') as f:
            edges = pickle.load(f)
        
        # node features
        # points = np.array(node_info[1])
        # points = points.reshape((points.shape[0], points.shape[1]*points.shape[2]))
        # features = torch.from_numpy(points)
        # print(np.array([i[0] for i in node_info[0]]).shape, np.array([i[1] for i in node_info[0]]).shape, 
        #       np.array([i[2] for i in node_info[0]]).shape, np.array([i[3] for i in node_info[0]]).shape)

        labels = np.array([i[0] for i in node_info[0]])
        centers = np.array([i[1] for i in node_info[0]])
        lengths = np.array([i[2] for i in node_info[0]])
        directions = np.array([i[3] for i in node_info[0]])
        features = torch.from_numpy(np.column_stack((labels, centers, lengths, directions)))

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
    for i, el in tqdm(enumerate(elements)):
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
