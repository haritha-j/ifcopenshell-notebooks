# generate ifc from obj mesh

import numpy as np
import ifcopenshell
import pywavefront
import uuid


# load obj file
obj_file = "../wall_4_1 - Cloud.obj"
blueprint = '../data/sample.ifc'

scene = pywavefront.Wavefront(obj_file, collect_faces=True)
f = scene.mesh_list[0].faces
v = scene.vertices

fx = []
for i in f:
    fx.append([i[0]+1, i[1]+1, i[2]+1])
print(len(v), len(f))

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

# create blueprint
new_ifc = setup_ifc_file(blueprint)
owner_history = new_ifc.by_type("IfcOwnerHistory")[0]
project = new_ifc.by_type("IfcProject")[0]
context = new_ifc.by_type("IfcGeometricRepresentationContext")[0]
floor = new_ifc.by_type("IfcBuildingStorey")[0]
        
ifc_info = {"owner_history": owner_history,
            "project": project,
           "context": context, 
           "floor": floor}

def create_guid(): return ifcopenshell.guid.compress(uuid.uuid1().hex)
owner_history=ifc_info["owner_history"]
container=ifc_info['floor']
context=ifc_info["context"]
project=ifc_info["project"]
name= "wall"

Z = 0., 0., 1.
# print('length', L)
B1 = new_ifc.createIfcWindow(create_guid(), owner_history, name)
#B1.ObjectType = 'beam'

#create mesh 
point_list = new_ifc.createIfcCartesianPointList3D()
point_list.CoordList = v

faceset = new_ifc.createIfcTriangulatedFaceSet()
faceset.CoordIndex = fx
faceset.Coordinates = point_list
#faceset.Closed = False

# placement
direction = (1.0,0.0,0.0)
B1_Point = new_ifc.createIfcCartesianPoint((0.0,0.0,0.0))
# B1_Point =ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) )
B1_Axis2Placement = new_ifc.createIfcAxis2Placement3D(B1_Point)
B1_Axis2Placement.Axis = new_ifc.createIfcDirection(direction)
B1_Axis2Placement.RefDirection = new_ifc.createIfcDirection(
    np.cross(direction, Z).tolist())

B1_Placement = new_ifc.createIfcLocalPlacement(
    container.ObjectPlacement, B1_Axis2Placement)
B1.ObjectPlacement = B1_Placement

B1_Repr = new_ifc.createIfcShapeRepresentation()
B1_Repr.ContextOfItems = context
B1_Repr.RepresentationIdentifier = 'Body'
B1_Repr.RepresentationType = 'Tessellation'
B1_Repr.Items = [faceset]

B1_DefShape = new_ifc.createIfcProductDefinitionShape()
B1_DefShape.Representations = [B1_Repr]
B1.Representation = B1_DefShape

Flr1_Container = new_ifc.createIfcRelContainedInSpatialStructure(
    create_guid(), owner_history)
Flr1_Container.RelatedElements = [B1]
Flr1_Container.RelatingStructure = container

new_ifc.write("../output_file.ifc")