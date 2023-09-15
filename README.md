## Industrial Facility Relationship Detection

This repository contains the codebase for `Learnable Geometry and Connectivity Modelling of BIM Objects`, accepted for BMVC 2023.

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

#### Dataset
The [dataset](https://drive.google.com/drive/folders/16rZGQSLgTGVj9BREb3tM1WkX26WDTaan) of industrial facility elements includes pipes, tees, elbows and flanges.
Two versions are available. The regular version contains clouds with moderate occlusions, sampled using 3 viewpoints. The occluded version has high occlusions, and is sampled using 2 viewpoints.
The scans are generated using 4096 IFC elements for each class.
