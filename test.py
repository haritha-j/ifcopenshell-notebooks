#import open3d as o3d
import os
import numpy as np
from tqdm import tqdm as tqdm
import math
import json
import random
import uuid
import ifcopenshell
from ifcopenshell import template
import ifcopenshell.geom

from src.ifc import *
from src.elements import create_pipe, create_elbow, create_tee, setup_ifc_file


density = 1024
sample_size = 4096
config_path = "config/pipeline.json"
pcd_path = "/home/haritha/downloads/blender-2.79-linux-glibc219-x86_64/output/pcd/"
blueprint = 'data/sample.ifc'
num_scans = 16

#random.seed(0)

def synthetic_dataset(config, sample_size, element_class, output_base, blueprint, start=0):
    # setup
    f = open(config, 'r')
    config_data  = json.load(f)
    output_dir = os.path.join(output_base, element_class)
    #os.makedirs(output_dir)

    metadata = {}
    for i in tqdm(range(start, sample_size+start)):
        #print("iteration", i)
        # generate ifc file
        ifc = setup_ifc_file(blueprint)
        owner_history = ifc.by_type("IfcOwnerHistory")[0]
        project = ifc.by_type("IfcProject")[0]
        context = ifc.by_type("IfcGeometricRepresentationContext")[0]
        floor = ifc.by_type("IfcBuildingStorey")[0]
        
        ifc_info = {"owner_history": owner_history,
            "project": project,
           "context": context, 
           "floor": floor}
        
        # generate ifc element
        if element_class == 'pipe':
            e = create_pipe(config_data[element_class], ifc, ifc_info)
        elif element_class == 'elbow':
            e = create_elbow(config_data[element_class], ifc, ifc_info, blueprint, i)
        elif element_class == 'tee':
            e = create_tee(config_data[element_class], ifc, ifc_info, blueprint)
    

        metadata[str(i)] = e
        ifc.write(os.path.join(output_dir, '%d.ifc' % i))
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        

synthetic_dataset(config_path, sample_size, "tee", 'output', blueprint, 0)