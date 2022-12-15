# viusalize relationships by drawing ifc elements

import math
import numpy as np
import ifcopenshell
import uuid
import json
import open3d as o3d
from OCC.Core.gp import gp_Pnt
from utils.JupyterIFCRenderer import JupyterIFCRenderer

from src.geometry import get_oriented_bbox, get_corner, sq_distance

create_guid = lambda: ifcopenshell.guid.compress(uuid.uuid1().hex)

# create an IFC BEAM
def CreateBeam(ifcFile, container, name, section, L, position, 
               direction, owner_history, context, colour=None):
    Z = 0.,0.,1.
    #print('length', L)
    B1 = ifcFile.createIfcBeam(create_guid(),owner_history , name)
    B1.ObjectType ='beam'
    
    #print(type(position[0]))
    B1_Point =ifcFile.createIfcCartesianPoint ( tuple(position) ) 
    #B1_Point =ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) ) 
    B1_Axis2Placement = ifcFile.createIfcAxis2Placement3D(B1_Point)
    B1_Axis2Placement.Axis = ifcFile.createIfcDirection(direction)
    B1_Axis2Placement.RefDirection =ifcFile.createIfcDirection(
        np.cross(direction, Z).tolist())

    B1_Placement = ifcFile.createIfcLocalPlacement(
        container.ObjectPlacement,B1_Axis2Placement)
    B1.ObjectPlacement=B1_Placement
    #B1Point = ifcFile.createIfcCartesianPoint ( tuple(position) )
    B1Point = ifcFile.createIfcCartesianPoint ((0.0,0.0,0.0) )
    B1_ExtrudePlacement = ifcFile.createIfcAxis2Placement3D(B1Point)
    #print (B1Point, B1_ExtrudePlacement, B1_Placement)
    
    B1_Extruded=ifcFile.createIfcExtrudedAreaSolid()
    B1_Extruded.SweptArea=section
    B1_Extruded.Position=B1_ExtrudePlacement
    B1_Extruded.ExtrudedDirection = ifcFile.createIfcDirection(Z)
    B1_Extruded.Depth = L
    
    # add colour
    if colour is not None:
        shade = ifcFile.createIfcSurfaceStyleRendering(colour)
        surfaceStyle = ifcFile.createIfcSurfaceStyle(colour.Name, "BOTH",(shade,))
        presStyleAssign = ifcFile.createIfcPresentationStyleAssignment((surfaceStyle,))
        ifcFile.createIfcStyledItem(B1_Extruded, (presStyleAssign,), colour.Name)


    B1_Repr=ifcFile.createIfcShapeRepresentation()
    B1_Repr.ContextOfItems=context
    B1_Repr.RepresentationIdentifier = 'Body'
    B1_Repr.RepresentationType = 'SweptSolid'
    B1_Repr.Items = [B1_Extruded]
    
    B1_DefShape=ifcFile.createIfcProductDefinitionShape()
    B1_DefShape.Representations=[B1_Repr]
    B1.Representation=B1_DefShape
    
    Flr1_Container = ifcFile.createIfcRelContainedInSpatialStructure(
        create_guid(),owner_history)
    Flr1_Container.RelatedElements=[B1]
    Flr1_Container.RelatingStructure= container

# create an IFC elbow
def CreateElbow(ifcFile, container, name, section, a, x, y, axis_dir, position, 
               direction, owner_history, context, colour=None):
    Z = 0.,0.,1.
    X = 1.,0.,0.
    #print('length', L)
    B1 = ifcFile.createIfcPipeFitting(create_guid(),owner_history , name)
    B1.ObjectType ='elbow'
    
    #print(type(position[0]))
    B1_Point =ifcFile.createIfcCartesianPoint ( tuple(position) ) 
    #B1_Point =ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) ) 
    B1_Axis2Placement = ifcFile.createIfcAxis2Placement3D(B1_Point)
    B1_Axis2Placement.Axis = ifcFile.createIfcDirection(direction)
    B1_Axis2Placement.RefDirection =ifcFile.createIfcDirection(
        np.cross(direction, Z).tolist())


    B1_Placement = ifcFile.createIfcLocalPlacement(
        container.ObjectPlacement,B1_Axis2Placement)
    B1.ObjectPlacement=B1_Placement
    #B1Point = ifcFile.createIfcCartesianPoint ( position )
    B1Point = ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) )
    B1_ExtrudePlacement = ifcFile.createIfcAxis2Placement3D(B1Point)
    #print (B1Point, B1_ExtrudePlacement, B1_Placement)

    B1Point2 = ifcFile.createIfcCartesianPoint ( (x,y,0.) )
    B1_Axis1Placement = ifcFile.createIfcAxis1Placement(B1Point2)
    B1_Axis1Placement.Axis = ifcFile.createIfcDirection((axis_dir[0], axis_dir[1], 0.))
    #B1_Axis1Placement.Axis = ifcFile.createIfcDirection((axis_dir[0], axis_dir[1], 0.))
    
    B1_Extruded=ifcFile.createIfcRevolvedAreaSolid()
    B1_Extruded.SweptArea=section
    B1_Extruded.Position=B1_ExtrudePlacement
    B1_Extruded.Axis = B1_Axis1Placement
    B1_Extruded.Angle = a
    
    # add colour
    if colour is not None:
        shade = ifcFile.createIfcSurfaceStyleRendering(colour)
        surfaceStyle = ifcFile.createIfcSurfaceStyle(colour.Name, "BOTH",(shade,))
        presStyleAssign = ifcFile.createIfcPresentationStyleAssignment((surfaceStyle,))
        ifcFile.createIfcStyledItem(B1_Extruded, (presStyleAssign,), colour.Name)


    B1_Repr=ifcFile.createIfcShapeRepresentation()
    B1_Repr.ContextOfItems=context
    B1_Repr.RepresentationIdentifier = 'Body'
    B1_Repr.RepresentationType = 'SweptSolid'
    B1_Repr.Items = [B1_Extruded]
    
    B1_DefShape=ifcFile.createIfcProductDefinitionShape()
    B1_DefShape.Representations=[B1_Repr]
    B1.Representation=B1_DefShape
    
    Flr1_Container = ifcFile.createIfcRelContainedInSpatialStructure(
        create_guid(),owner_history)
    Flr1_Container.RelatedElements=[B1]
    Flr1_Container.RelatingStructure= container
    
    
def Circle_Section(r, ifcfile, fill=False):
    B1_Axis2Placement2D =ifcfile.createIfcAxis2Placement2D( 
                          ifcfile.createIfcCartesianPoint( (0.,0.,0.) ) )
    if fill:
        B1_AreaProfile = ifcfile.createIfcCircleProfileDef("AREA")
    else:
        B1_AreaProfile = ifcfile.createIfcCircleHollowProfileDef("AREA")
        B1_AreaProfile.WallThickness  = 2

    B1_AreaProfile.Position = B1_Axis2Placement2D 
    B1_AreaProfile.Radius = r
    return B1_AreaProfile    
    
def Rectangle_Section(ifcfile, x, y, fill=False):
    B1_Axis2Placement2D =ifcfile.createIfcAxis2Placement2D( 
                          ifcfile.createIfcCartesianPoint( (0.,0.,0.) ) )
    if fill:
        B1_AreaProfile = ifcfile.createIfcRectangleProfileDef("AREA")
    else:
        B1_AreaProfile = ifcfile.createIfcRectangleProfileDef("AREA")
        B1_AreaProfile.WallThickness  = 2

    B1_AreaProfile.Position = B1_Axis2Placement2D 
    B1_AreaProfile.XDim = x*1000
    B1_AreaProfile.YDim = y*1000
    return B1_AreaProfile


def draw_bbox(bbox, center, ifc, floor, owner_history, context):
    sectionC1 = Rectangle_Section(ifc, bbox[0], bbox[1], True)

    name='bb'
    print("draw bbox", bbox, center)
    ConnectingBeam_1 = CreateBeam(ifc, container=floor, name=name, 
                                  section= sectionC1, L=bbox[2]*1000,
                                  position=([-bbox[2]*500, 0., 0.]),
                                  direction=(1., 0., 0.),
                                  owner_history=owner_history, context=context, colour=None)

    
def draw_cylinder(p1, p2, radius, colour,  element_name1, element_name2, 
ifc, floor, owner_history, context):
    sectionC1 = Circle_Section(r=radius, ifcfile=ifc)
    name='rel '+element_name1 + ' x ' + element_name2
    ConnectingBeam_1 = CreateBeam(ifc, container=floor, name=name, 
                                  section= sectionC1, 
                                  L=math.sqrt(sq_distance(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2])),
                                  position=([p2[0].item(), p2[1].item(), p2[2].item()]),
                                  direction=((p1[0] - p2[0]).item(), (p1[1] - p2[1]).item(), 
                                             (p1[2] - p2[2]).item()),
                                  owner_history=owner_history, context=context, colour=colour)
#     ifc.create_entity('IfcPipeSegment', GlobalId=ifcopenshell.guid.new(), 
#                       Name='rel '+element_name1 + ' x ' + element_name2)
    
# def draw_cylinder(p1, p2, radius, colour, viewer):
#     x = (p1.X() + p2.X())/2
#     y = (p1.Y() + p2.Y())/2
#     z = (p1.Z() + p2.Z())/2
#     p = gp_Pnt(p2.X(), p2.Y(), p2.Z())
#     v = gp_Dir((p1.X() - p2.X()), (p1.Y() - p2.Y()), (p1.Z() - p2.Z()))
#     ax = gp_Ax2(p, v)
#     h = math.sqrt(sq_distance(p1.X(), p1.Y(), p2.Z(), 
#                               p2.X(), p2.Y(), p2.Z()))
#     print(x,y,z,p,v,ax,h)
    
#     cylinder = BRepPrimAPI_MakeCylinder(ax, radius, h).Shape()
#     viewer.DisplayShape(cylinder, shape_color = colour, 
#                         transparency=True, opacity=0.8)
    

# draw a visual indication of a topological relationship    
def draw_relationship (element_name1, element1, element_name2,
                       element2, ifc, floor, owner_history, context, colour=None):
    # define params
    #radius = 0.03
    radius_expansion = 1.3
    threshold = 0.1
    edge_distance = 0.1
    
    # get bboxes
    obb1 = get_oriented_bbox(element1)
    obb2 = get_oriented_bbox(element2)
    
    # if bboxes are roughly square, then use centerpoint, 
    # otherwise use a point closer to the edge
    if obb1[1] < threshold:
        corner1 = obb1[3]
    else:
        corner1 = get_corner(obb1, obb2[3], edge_distance)
    if  obb2[1] < threshold:
        corner2 = obb2[3]
    else:
        corner2 = get_corner(obb2, obb1[3], edge_distance)
    #print(corner1.X(),corner1.Y(),corner1.Z(),corner2.X(),corner2.Y(),corner1.Z())
    #print('rel', corner1, corner2, element_name1, element_name2)
    
    #dynamically scale radius
    radius = max(min(obb1[2]), min(obb2[2])) * radius_expansion
    
    draw_cylinder(corner1, corner2, radius, colour, 
                  element_name1, element_name2, ifc, floor, owner_history, context)
#     draw_sphere(corner1, radius, colour, viewer)
#     draw_sphere(corner2, radius, 'red', viewer)



# visualize ifc model and point cloud simultaneously
def vis_ifc_and_cloud(ifc, cloud, colour="#abe000"):
  viewer = JupyterIFCRenderer(ifc, size=(400,300))
  gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud.points]
  viewer.DisplayShape(gp_pnt_list, '#abe000')
  return viewer