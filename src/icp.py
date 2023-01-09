import numpy as np
import math
import os
import copy
import subprocess
import open3d as o3d

from src.visualisation import *
from src.geometry import  vector_norm


def run_command(cmds):
    popen = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    return (output)


def save_registration_result(source, transformation, save_path):
    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp],
#                                       zoom=0.4459,
#                                       front=[0.9288, -0.2951, -0.2242],
#                                       lookat=[1.6784, 2.0612, 1.4451],
#                                       up=[-0.3402, -0.9189, -0.1996])
    o3d.io.write_point_cloud( save_path, source_temp)
    
    
def icp(source, target, threshold, trans_init, save_path=None):

    #draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    
    if save_path is not None:
        save_registration_result(source, reg_p2p.transformation, save_path)
    source_temp = copy.deepcopy(source)
    return (reg_p2p.transformation, source_temp.transform(reg_p2p.transformation).points)
    #return (reg_p2p.transformation)


def transform_position(params, t, start_index):
    indices = [i for i in range(start_index, start_index+3)]
    p = [params[ind] for ind in indices]
    p.append(1.)
    transformed_p = np.matmul(t, np.array(p)).tolist()

    for i, ind in enumerate(indices):
        params[ind] = transformed_p[i]
    return params


def transform_direction(params, t, start_index,):
    d = get_direction_from_trig(params, start_index)
    d.append(1.)
    transformed_d = np.matmul(t, np.array(d))[:-1]
    origin = [0.,0.,0.,1.]
    transformed_origin = np.matmul(t, np.array(origin))[:-1]
    new_d = transformed_d - transformed_origin
    new_d = vector_norm(new_d)
    
    for i, axis in enumerate(new_d):
        params[start_index + i*2] = math.sin(axis)
        params[start_index + i*2 + 1] = math.cos(axis)
    return params


def transform_z_axis(t):
    z = [0., 0., 1., 1.]
    transformed_z = np.matmul(t, np.array(z))[:-1]
    origin = [0.,0.,0.,1.]
    transformed_origin = np.matmul(t, np.array(origin))[:-1]
    z_axis = transformed_z - transformed_origin
    print("transformed z", z_axis)
    return z_axis


def transform_params(params, t, cat):
    #t = np.transpose(t)
    print(params)
    z= (0., 0., 1.)
    if cat == "tee":
        #pass
        params = transform_position(params, t, 4)
        params = transform_direction(params, t, 7)
        params = transform_direction(params, t, 13)

    if cat == "elbow":
        params = transform_position(params, t, 5)
        params = transform_direction(params, t, 8)
        z = transform_z_axis(t)

    return params, z


def icp_finetuning(pcd, pcd_id, cat, preds, blueprint, temp_dir, target_dir, ifcConvert_executable,
                   cloudCompare_executable, sample_size, threshold):
    trans_init  = np.identity(4)

    # get ifc file
    ifc = visualize_predictions(pcd, cat, [preds], blueprint, visualize=False)
    tmp_ifc = os.path.join(temp_dir, 'tmp.ifc')
    tmp_obj = os.path.join(temp_dir, 'tmp.obj')
    tmp_mtl = os.path.join(temp_dir, 'tmp.mtl')
    tmp_pcd = os.path.join(temp_dir, 'tmp.pcd')
    ifc.write(tmp_ifc)

    # convert to obj
    cmds = (ifcConvert_executable, tmp_ifc, tmp_obj)
    result = run_command(cmds)
    #print("obj conversion", result)

    # sample points
    cmds = (cloudCompare_executable, "-SILENT", "-AUTO_SAVE", "OFF", "-O", 
            tmp_obj, "-SAMPLE_MESH", "POINTS", str(sample_size), 
            "-C_EXPORT_FMT", "PCD", "-SAVE_CLOUDS", "FILE", tmp_pcd)
    result = run_command(cmds)
    #print("point sampling", result)

    # perform ICP
    source = o3d.io.read_point_cloud(tmp_pcd)
    #target = o3d.io.read_point_cloud(os.path.join(target_dir, str(pcd_id)+".pcd"))
    target = o3d.geometry.PointCloud()
    target.points = pcd
    transformation, transformed_pcd = icp(source, target, threshold, trans_init)

    # transform parameters
    pred_copy = copy.deepcopy(preds)
    transformed_preds, z = transform_params(preds, transformation, cat)
    viewer, ifc = visualize_predictions([pcd], cat, [transformed_preds], blueprint, visualize=True)
    #viewer, ifc = visualize_predictions([pcd, source.points], cat, [pred_copy], blueprint, visualize=True, z=z)

    # cleanup
    os.remove(tmp_ifc)
    os.remove(tmp_obj)
    os.remove(tmp_mtl)
    os.remove(tmp_pcd)
    
    return viewer, ifc
    