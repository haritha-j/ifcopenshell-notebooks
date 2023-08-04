import pymeshlab as ml
import ifcopenshell


ifc = ifcopenshell.open("../deckboxtee.ifc")

element_type = 'IFCPIPEFITTING'
selector = Selector()
tees = selector.parse(ifc, '.' + element_type)
print(tees[0])


for element in tqdm(tees):
    # create pymeshlab mesh
    #print(element)
    shape = element.Representation.Representations[0].Items[0]
    element_faces = [np.array([i[0]-1, i[1]-1, i[2]-1]) for i in shape.CoordIndex]
    element_coords = np.array(shape.Coordinates.CoordList)
    print(len(element_faces))

ifc.write('../added.ifc') 


## meshlab downsampling -DOESNT WORK
"""


mesh = ml.Mesh(element_coords, element_faces)
ms = ml.MeshSet()
ms.add_mesh(mesh, "x")
#ms.save_current_mesh("../output.ply")

# downsample and reassign
ms.apply_filter('simplification_clustering_decimation', threshold=ml.Percentage(2))
m = ms.current_mesh()
shape.CoordIndex = m.face_matrix().tolist()
shape.Coordinates.CoordList = m.vertex_matrix().tolist()
    
print('input mesh has', mesh.vertex_number(), 'vertex and', mesh.face_number(), 'faces')


#Estimate number of faces to have 100+10000 vertex using Euler

#Simplify the mesh. Only first simplification will be agressive
#ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=False)
ms.apply_filter('simplification_clustering_decimation', threshold=ml.Percentage(5))
print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")


m = ms.current_mesh()
print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
ms.save_current_mesh('../output.ply')

"""