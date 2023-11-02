## Industrial Facility Relationship Detection

This repository performs relationship inference on industrial facility datasets using a GNN. It is compatible with NWD/IFC design files for training/evaluation and labelled point cloud datasets (CLOI) for evaluation. For relationship annotation on point cloud datasets, see this fork of [labelcloud.](https://github.com/haritha-j/labelCloud/tree/rel)

The primary notebook for this repository is *graph_experiments.ipynb*. 
It contains the following sections:
1. Relationship identification: Identify, extract and (visualize) 
aggregation and connectivity relationships.
2. Visalization: Deprecated
3. IFC to cloud: Create point clouds from each element in IFC file
4. Graph dataset: Create a graph dataset from relationship information
5. Evaluate GNN: Evaluate predictions from GNN
6. Visalize predictions: Draw IFC element to visualize FP,TP,FNs
7. Analyze dataset and results: Repetition removal, element category analysis

Training and inference using the GNN is contained in *link_prediction.ipynb*. This fits between steps 4 and 5 of the above notebook.

CLOI dataset preperation is contained in *CLOI.ipynb*.

Additionally, scripts for exctracting facility structure from NWD files (*total.py*), downsampling element point clouds (*downsample.py*), merging IFC elements (*ifcmerge.py*) and cleaning up IFC files extracted from NWD files (*IFCPARSE.py*) are available in the *utils* folder.

Source code for processing ifc geometry, graphs, point clouds, industrial facility structure and visualisation are available in the *src* folder.

This repository is dependent on a custom version of [chamferdist](https://github.com/haritha-j/chamferdist)


### Dataset 

The dataset is available [here](https://drive.google.com/file/d/14MYRz4-1RuoHHstqzGsiybucTb4wpWs6/view?usp=sharing)


This dataset consists of two sub-sections.
1. Synthetic dataset
2. Industrial facility dataset

Point clouds for each dataset are available in both regular occclusion (3 camera viewpoints) and high occlusion (2 camera viewpoints) variants.

The datasets contain pipe, elbow, flange and tee classes.


##### Synthetic dataset

clouds sampled from 4000+ synthetically generated elements for each class. IFC, OBJ and sampled point clouds are available for each element, as well as metadata describing the underlying geoemtric parameters.


#### Industrial facility dataset

Point clouds sampled from two subsections of an offshore LNG hub.
Metadata contains relationship information for piping branches.
Geometry predictions for all elements from our proposed methods have been included.
Obj files are also included.
